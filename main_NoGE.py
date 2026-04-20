import numpy as np
import torch
import time
import argparse
import scipy.sparse as sp
import wandb
import os
import json
from load_data import Data
from collections import defaultdict
from utils_NoGE import *
from model_NoGE import *
import random

torch.manual_seed(1337)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)
np.random.seed(1337)


class NoGE:
    """ Node Coherence-based Graph Neural Networks for Knowledge Graph Link Prediction """

    def __init__(self, encoder="QGNN", decoder="QuatE", num_iterations=3000, batch_size=1024, learning_rate=0.01,
                 label_smoothing=0.1,
                 hidden_dim=128, emb_dim=128, num_layers=1, variant="N", semantic_type="I", pretrain_dir="./cybert",
                 eval_step=1, eval_after=1,
                 a_h: float = 0, a_t: float = 0, a_hr: float = 0, a_tr: float = 0, temperature=0.9):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.label_smoothing = label_smoothing
        self.eval_step = eval_step
        self.eval_after = eval_after
        self.encoder = encoder
        self.decoder = decoder
        self.hid_dim = hidden_dim
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.variant = variant
        self.semantic_type = semantic_type
        self.pretrain_dir = pretrain_dir
        self.a_h = a_h
        self.a_t = a_t
        self.a_hr = a_hr
        self.a_tr = a_tr
        self.temperature = temperature

    """ Functions are adapted from https://github.com/ibalazevic/TuckER for using 1-N scoring strategy """

    def get_data_idxs(self, data):
        # entity_idxs和relation_idx都是按顺序排的id，一一对应
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], self.entity_idxs[data[i][2]]) for i
                     in range(len(data))]
        return data_idxs

    # 实体预测：(h,r)→t
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx + self.batch_size]  # 一个batch的(h，r)，要去掉落单的样本（bn要求多于一个样本）
        # 这里把 len(d.entities) 换成了 len(self.entity_idxs)，严格基于字典大小
        targets = np.zeros((len(batch), len(self.entity_idxs)))
        batch_examples = []
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.  # er_vocab[pair]是列表，targets是g-hot
            # er_vocab[pair]是list
            for t in er_vocab[pair]:
                batch_examples.append([pair[0], pair[1], t])
        targets = torch.FloatTensor(targets)
        return np.array(batch), targets.to(device), np.array(batch_examples)

    # evaluation
    def evaluate(self, model, data, lst_indexes):
        model.eval()
        with torch.no_grad():  # 不会更新梯度
            hits = []
            ranks = []
            for i in range(10):
                hits.append([])

            test_data_idxs = self.get_data_idxs(data)
            er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))
            print("Number of data points: %d" % len(test_data_idxs))

            for i in range(0, len(test_data_idxs), self.batch_size):
                data_batch, _, _ = self.get_batch(er_vocab, test_data_idxs, i)
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)
                r_idx = torch.tensor(data_batch[:, 1]).to(device)
                e2_idx = torch.tensor(data_batch[:, 2]).to(device)

                predictions = model.forward(e1_idx, r_idx, lst_indexes).detach()

                for j in range(data_batch.shape[0]):
                    filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
                    target_value = predictions[j, e2_idx[j]].item()
                    predictions[j, filt] = 0.0
                    predictions[j, e2_idx[j]] = target_value

                sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

                sort_idxs = sort_idxs.cpu().numpy()
                for j in range(data_batch.shape[0]):
                    rank = np.where(sort_idxs[j] == e2_idx[j].item())[0][0]
                    ranks.append(rank + 1)
                    for hits_level in range(10):
                        if rank <= hits_level:
                            hits[hits_level].append(1.0)
                        else:
                            hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9]) * 100))
        print('Hits @3: {0}'.format(np.mean(hits[2]) * 100))
        print('Hits @1: {0}'.format(np.mean(hits[0]) * 100))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))

        return np.mean(hits[9]) * 100, np.mean(hits[2]) * 100, np.mean(hits[0]) * 100, np.mean(ranks), np.mean(
            1. / np.array(ranks))

    # 对比学习，1、构造字典
    def get_dict(self, examples):
        dic_tr = {}
        dic_hr = {}
        dic_t = {}
        dic_h = {}
        # 创建空字典
        for i in examples:
            dic_tr[i[0]] = []  # 共享头实体,d
            dic_hr[i[2]] = []  # 共享尾实体,c
            dic_t[(i[0], i[1])] = []  # 共享头实体、关系,b
            dic_h[(i[2], i[1])] = []  # 共享尾实体、关系,a
        for i in examples:  # (h,r,t)→(h,t,r)
            dic_tr[i[0]].append((i[2], i[1]))  # 列表改成元组
            dic_hr[i[2]].append((i[0], i[1]))
            dic_t[(i[0], i[1])].append(i[2])
            dic_h[(i[2], i[1])].append(i[0])
        return dic_tr, dic_hr, dic_h, dic_t

    # 2、根据字典获取正样本
    def get_pos(self, actual_examples, dic_hr=None, dic_tr=None, dic_h=None, dic_t=None):
        if dic_hr is not None:
            p_hr = []
            for i in actual_examples:
                hr_sample = dic_hr[i[2]]
                random.shuffle(hr_sample)
                p_hr.append(hr_sample[0])
            return p_hr
        if dic_tr is not None:
            p_tr = []
            for i in actual_examples:
                tr_sample = dic_tr[i[0]]
                random.shuffle(tr_sample)
                p_tr.append(tr_sample[0])
            return p_tr
        if dic_h is not None:
            p_h = []
            for i in actual_examples:
                h_sample = dic_h[(i[2], i[1])]
                random.shuffle(h_sample)
                p_h.append(h_sample[0])
            return p_h
        if dic_t is not None:
            p_t = []
            for i in actual_examples:
                t_sample = dic_t[(i[0], i[1])]
                random.shuffle(t_sample)
                p_t.append(t_sample[0])
            return p_t

    # 3、根据a_hr、a_tr、a_h、a_t超参数，调用get_pos方法获取正样本
    def get_p(self, batch_examples, dic_hr, dic_tr, dic_h, dic_t):
        p_hr, p_tr, p_h, p_t = None, None, None, None
        # 根据batch中三元组，获取对应的正样本
        if self.a_hr != 0:  # 共享t
            p_hr = self.get_pos(batch_examples, dic_hr=dic_hr)
            p_hr = torch.tensor(p_hr)
        if self.a_tr != 0:  # 共享h
            p_tr = self.get_pos(batch_examples, dic_tr=dic_tr)
            p_tr = torch.tensor(p_tr)
        if self.a_h != 0:  # 共享rt
            p_h = self.get_pos(batch_examples, dic_h=dic_h)  # 返回头实体id的list
            p_h = torch.tensor(p_h)
        if self.a_t != 0:  # 共享hr
            p_t = self.get_pos(batch_examples, dic_t=dic_t)
            p_t = torch.tensor(p_t)
        return p_hr, p_tr, p_h, p_t

    # training and evaluating
    def train_and_eval(self):
        # 彻底回到原始状态：根据内存数据动态分配ID，完美包容 _reverse
        self.entity_idxs = {d.entities[i]: i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]: i for i in range(len(d.relations))}

        # 取出训练集。排在前面，按训练集大小取即可
        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        print("Creating the new weighted Adj matrix!")
        # 根据训练集构造权重矩阵,节点数*节点数
        adj = compute_weighted_adj_matrix(d.train_data, self.entity_idxs, self.relation_idxs).to(device)
        print('len of adj:', len(adj))

        if self.encoder.lower() == "gcn":
            print("Training with the GCN encoder")
            model = NoGE_GCN_QuatE(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs), len(self.relation_idxs),
                                   self.num_layers).to(device)
            print("and the customized QuatE decoder...")

        else:
            print("Training with the QGNN encoder")
            if self.decoder.lower() == "quate":
                model = NoGE_QGNN_QuatE(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs), len(self.relation_idxs),
                                        self.entity_idxs, self.relation_idxs, self.num_layers, self.variant,
                                        self.semantic_type, self.pretrain_dir).to(device)
                cl_net = CL(self.emb_dim, temperature=0.9, hidden_size=self.hid_dim).to(device)
            else:
                model = NoGE_QGNN_DistMult(self.emb_dim, self.hid_dim, adj, len(self.entity_idxs),
                                           len(self.relation_idxs)).to(device)
                print("and the customized DistMult decoder...")

        print("Using Adam optimizer")
        # 不衰减
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # 修改点：gnn的节点数=实体数+关系数，这里改为使用 self.entity_idxs 和 self.relation_idxs 保证匹配
        lst_indexes = torch.LongTensor([i for i in range(len(self.entity_idxs) + len(self.relation_idxs))]).to(device)

        # er_vocab：key：（h，r）元组，value：尾实体的id列表。维度：（h，r）类别数 * 以（h，r）开头的三元组数量
        # er_vocab_pairs： 去掉尾实体以后的（h，r）——用于尾实体预测
        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())
        max_valid = 0.0
        final_test_h10 = 0.0
        final_test_h3 = 0.0
        final_test_h1 = 0.0
        final_test_mr = 0.0
        final_test_mrr = 0.0
        best_epoch = 0

        patience = 8  # ❗新增：最大容忍次数
        patience_counter = 0  # ❗新增：当前连续未提升的次数

        print("Starting training...")

        # 每个epoch
        for it in range(1, self.num_iterations + 1):
            model.train()
            losses = []
            np.random.shuffle(er_vocab_pairs)

            # 4、对比学习模块
            # 共享h、t、tr、hr
            dic_tr, dic_hr, dic_h, dic_t = self.get_dict(train_data_idxs)  # 训练集

            # 引入对比学习模块
            cl_net = CL(temperature=self.temperature).cuda()

            # 每个批次，注意是按照er_vocab_pairs来分批，不是按照三元组样本来分批
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                # data_batch：（h，r）
                # targets：批次大小*实体数量，每一行是g-hot的向量，标签
                # input_batch则为批次内所有三元组数量，＞data_batch
                data_batch, targets, input_batch = self.get_batch(er_vocab, er_vocab_pairs, j)

                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:, 0]).to(device)  # 头实体id
                r_idx = torch.tensor(data_batch[:, 1]).to(device)  # 关系id

                # 头实体+关系，预测尾实体； lst_indexes：0~节点数-1
                predictions = model.forward(e1_idx, r_idx, lst_indexes)  # 头实体+关系共现的id

                if self.label_smoothing:
                    targets = ((1.0 - self.label_smoothing) * targets) + (1.0 / targets.size(1))
                pre_loss = model.loss(predictions, targets)  # BCE loss

                # 计算对比损失，分别共享t、h、tr、hr
                # 如果超参数都为0，则不计算对比损失
                p_hr, p_tr, p_h, p_t = self.get_p(input_batch, dic_hr, dic_tr, dic_h, dic_t)
                pos_hr_loss, pos_tr_loss, pos_h_loss, pos_t_loss = 0, 0, 0, 0
                if p_hr is not None:
                    p_tail = p_hr.cuda()  # h、r的id
                    # self_hr：批次的hr, pos_t_predictions：正例的hr
                    labels1 = torch.tensor(input_batch[:, 0])  # 头实体id
                    labels2 = torch.tensor(input_batch[:, 1])  # 关系id
                    self_hr, pos_t_predictions = model.forward(labels1, labels2, lst_indexes, p_tail=p_tail, mod=0)
                    pos_hr_loss = cl_net(self_hr, pos_t_predictions, labels1=labels1, labels2=labels2)

                if p_h is not None:
                    p_h_pos = p_h.cuda()
                    self_h_e = model.embeddings(torch.tensor(input_batch[:, 0]).cuda())
                    p_h_e = model.embeddings(p_h_pos)
                    labels1 = torch.tensor(input_batch[:, 2])
                    pos_h_loss = cl_net(self_h_e, p_h_e, labels1)  # 正例的对比损失

                CL_loss = self.a_hr * pos_hr_loss + self.a_tr * pos_tr_loss + self.a_h * pos_h_loss + self.a_t * pos_t_loss

                loss = pre_loss + CL_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # prevent the exploding gradient problem
                opt.step()
                losses.append(loss.item())

            print("Epoch: {}".format(it), " --> Loss: {:.4f}".format(np.sum(losses)))  # 每个epoch：batch内step相加
            # 输出梯度
            for name, parms in model.named_parameters():
                if torch.isnan(parms).any() or parms.equal(torch.zeros(parms.shape).to(device)):
                    print('-->name:', name, '-->grad_requirs:', parms.requires_grad, \
                          ' -->grad_value:', parms.grad)
                    breakpoint()

            print(it)
            print(np.mean(losses))
            # evaluation：多少个epoch之后验证测试
            if it > self.eval_after and it % self.eval_step == 0:
                print("Validation:")
                tmp_hit10, tmp_hit3, tmp_hit1, tmp_mr, tmp_mrr = self.evaluate(model, d.valid_data, lst_indexes)

                if max_valid < tmp_mrr:
                    max_valid = tmp_mrr
                    best_epoch = it
                    patience_counter = 0  # ❗新增：如果有提升，计数器清零

                    print("Test:")
                    final_test_h10, final_test_h3, final_test_h1, final_test_mr, final_test_mrr = self.evaluate(model,
                                                                                                                d.test_data,
                                                                                                                lst_indexes)

                    # ===== 保存最终embedding（用于MuRP初始化）=====
                    print("Saving entity embeddings...")

                    model.eval()
                    with torch.no_grad():
                        entity_ids = torch.arange(len(self.entity_idxs)).to(device)

                        # 🔥 关键：不同模型写法不一样，这里做兼容
                        if hasattr(model, "embeddings"):
                            try:
                                entity_emb = model.embeddings(entity_ids)
                            except:
                                entity_emb = model.embeddings[0](entity_ids)
                        else:
                            raise ValueError("Model has no embeddings attribute!")

                        entity_emb = entity_emb.detach().cpu()

                    save_path = "./noge_entity_embeddings.pt"
                    torch.save(entity_emb, save_path)
                    print(f"Embedding saved to {save_path}")

                    # ===== 保存entity2id（防止MuRP错位）=====
                    # 依然保存在同级目录下，但确保包含最新的数据映射
                    # with open(os.path.join(dict_dir, "entity2id.json"), "w", encoding='utf-8') as f:
                    #     json.dump(self.entity_idxs, f, ensure_ascii=False, indent=4)

                    # print("Entity mapping saved.")

                    print("Best valid epoch", best_epoch, " --> Final test results: ", final_test_h10, final_test_h3,
                          final_test_h1, final_test_mr, final_test_mrr)
                else:
                    # ❗新增：如果这次验证集没有破记录，计数器 +1
                    patience_counter += 1
                    print(f"EarlyStopping counter: {patience_counter} out of {patience}")

                    # ❗新增：达到容忍上限，触发早停
                if patience_counter >= patience:
                    print(f"✅ Early stopping triggered! 连续 {patience} 次验证集未提升，提前结束训练。")
                    break  # 直接跳出整个 for 循环


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="APT-Hier", nargs="?", help="1, 2, and 3")
    parser.add_argument("--num_iterations", type=int, default=3000, nargs="?", help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=128, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.005, nargs="?", help="Learning rate.")
    parser.add_argument("--hidden_dim", type=int, default=256, nargs="?", help="")
    parser.add_argument("--emb_dim", type=int, default=256, nargs="?", help="")
    parser.add_argument("--num_layers", type=int, default=1, nargs="?", help="Number of layers")
    parser.add_argument("--encoder", type=str, default="QGNN", nargs="?")
    parser.add_argument("--decoder", type=str, default="QuatE", nargs="?")
    parser.add_argument("--variant", type=str, default="D", nargs="?", help="N: QGNN, D: Dual QGNN")
    parser.add_argument("--eval_step", type=int, default=100, nargs="?")
    parser.add_argument("--eval_after", type=int, default=0, nargs="?")
    # semantic embeddings dim = emb_dim
    parser.add_argument("--semantic_type", type=str, default="None", nargs="?",
                        help="I: initial, CT: concat train, C:concat")
    parser.add_argument("--pretrain_dir", type=str, default="./cybert", nargs="?", help="regional pretrained model")
    parser.add_argument('--a_h', default=0, type=float, help="a_h")
    parser.add_argument('--a_t', default=0, type=float, help="a_t")
    parser.add_argument('--a_hr', default=0, type=float, help="a_hr")
    parser.add_argument('--a_tr', default=0, type=float, help="a_tr")
    parser.add_argument('--temperature', default=0.9, type=float, help="temperature")
    args = parser.parse_args()

    dataset = args.dataset
    # 修改点：将主入口的读取路径也换为了05_data_重构
    data_dir = "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/04_data/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    d = Data(data_dir=data_dir)

    gnnkge = NoGE(encoder=args.encoder, decoder=args.decoder, num_iterations=args.num_iterations,
                  batch_size=args.batch_size,
                  learning_rate=args.lr, hidden_dim=args.hidden_dim, emb_dim=args.emb_dim, num_layers=args.num_layers,
                  eval_step=args.eval_step, eval_after=args.eval_after, variant=args.variant,
                  semantic_type=args.semantic_type, pretrain_dir=args.pretrain_dir,
                  a_h=args.a_h, a_t=args.a_t, a_hr=args.a_hr, a_tr=args.a_tr, temperature=args.temperature)

    gnnkge.train_and_eval()