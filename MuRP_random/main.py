import numpy as np
import torch
import time
import argparse
import os
import json
from collections import defaultdict
from load_data_hier import Data
from model import *
from rsgd import *


class Experiment:

    def __init__(self, learning_rate=50, dim=200, nneg=50, model="poincare",
                 num_iterations=5, batch_size=128, cuda=False, data_dir=""):
        self.model = model
        self.learning_rate = learning_rate
        self.dim = dim
        self.nneg = nneg
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.cuda = cuda
        self.data_dir = data_dir  # 新增：接收傳入的資料集路徑

        # 初始化对比损失
        # self.contrastive_loss = NoGEContrastiveLoss(temperature=0.1)

        # [新增] 初始化最佳评估指标和对应的 epoch
        self.best_metric = 0.0  # 因为后面是用 Hits@1，所以初始值设为 0
        self.best_epoch = 0
        self.best_model_state = None  # ⭐ 新增

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def get_er_vocab(self, data, idxs=[0, 1, 2]):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[idxs[0]], triple[idxs[1]])].append(triple[idxs[2]])
        return er_vocab

    def evaluate(self, model, data):
        hits = [[] for _ in range(10)]
        ranks = []

        test_data_idxs = self.get_data_idxs(data)
        sr_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))

        # 修改點：統一使用 self.entity_idxs 的長度作為實體總數
        num_entities = len(self.entity_idxs)

        for i in range(len(test_data_idxs)):
            data_point = test_data_idxs[i]

            e1_idx = torch.tensor(data_point[0])
            r_idx = torch.tensor(data_point[1])
            e2_idx = torch.tensor(data_point[2])

            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()

            # 修改點：將原本的 len(d.entities) 改為字典的實體總數，防止維度不對齊
            predictions = model.forward(
                e1_idx.repeat(num_entities),
                r_idx.repeat(num_entities),
                range(num_entities)
            )

            filt = sr_vocab[(data_point[0], data_point[1])]
            target_value = predictions[e2_idx].item()

            predictions[filt] = -np.inf
            predictions[e1_idx] = -np.inf
            predictions[e2_idx] = target_value

            _, sort_idxs = torch.sort(predictions, descending=True)
            sort_idxs = sort_idxs.cpu().numpy()

            rank = np.where(sort_idxs == e2_idx.item())[0][0] + 1
            ranks.append(rank)

            for k in range(10):
                hits[k].append(1.0 if rank <= k + 1 else 0.0)

        # ⭐ 五个指标
        hits10 = np.mean(hits[9])
        hits3 = np.mean(hits[2])
        hits1 = np.mean(hits[0])
        mr = np.mean(ranks)
        mrr = np.mean(1.0 / np.array(ranks))

        print(f"Hits @10: {hits10}")
        print(f"Hits @3: {hits3}")
        print(f"Hits @1: {hits1}")
        print(f"Mean Rank: {mr}")
        print(f"MRR: {mrr}")

        return hits10, hits3, hits1, mr, mrr

    def train_and_eval(self):
        print("Training the %s model..." % self.model)

        # 修改點：不再動態生成，而是從資料目錄直接讀取 entity2id.json 和 relation2id.json
        print(f"Loading dictionaries from: {self.data_dir}")
        with open(os.path.join(self.data_dir, "entity2id.json"), "r", encoding="utf-8") as f:
            self.entity_idxs = json.load(f)

        with open(os.path.join(self.data_dir, "relation2id.json"), "r", encoding="utf-8") as f:
            self.relation_idxs = json.load(f)

        # ====== 🌟 核心修复区：动态补齐反向关系 ======
        # 遍历原始所有的关系，如果存在正向关系，就为其分配一个 _reverse 反向关系的 ID
        original_rels = list(self.relation_idxs.keys())
        for rel in original_rels:
            # 避免重复添加 _reverse_reverse
            if not rel.endswith("_reverse"):
                rev_rel = rel + "_reverse"
                if rev_rel not in self.relation_idxs:
                    # 为新的反向关系分配递增的最新 ID
                    self.relation_idxs[rev_rel] = len(self.relation_idxs)
        # ============================================

        # 覆蓋載入資料時的 d.entities 和 d.relations 以免傳入外部模型(MuRP)時發生維度不匹配的問題
        d.entities = list(self.entity_idxs.keys())
        d.relations = list(self.relation_idxs.keys())

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        model = MuRP(d, self.dim)

        # ❗❗❗ 新增这三行：如果启用了 cuda，就把模型送进 GPU ❗❗❗
        if self.cuda:
            model = model.cuda()
            print("🚀 Model successfully moved to GPU!")

        param_names = [name for name, param in model.named_parameters()]
        opt = RiemannianSGD(model.parameters(), lr=self.learning_rate, param_names=param_names)

        er_vocab = self.get_er_vocab(train_data_idxs)

        print("Starting training...")
        # ⭐ 新增：用于记录每个 epoch 的平均 Loss 的列表
        epoch_losses_history = []

        # （把这段放进循环开始前）
        patience = 5  # 容忍多少次验证集不提升就停
        patience_counter = 0

        for it in range(1, self.num_iterations + 1):
            start_train = time.time()
            model.train()

            losses = []  # 这个是记录当前 epoch 内各个 batch 的 loss
            np.random.shuffle(train_data_idxs)
            for j in range(0, len(train_data_idxs), self.batch_size):
                data_batch = np.array(train_data_idxs[j:j + self.batch_size])
                # 负样本变量 (基於 self.entity_idxs 取得所有的實體ID)
                negsamples = np.random.choice(list(self.entity_idxs.values()),
                                              size=(data_batch.shape[0], self.nneg))

                e1_idx = torch.tensor(np.tile(np.array([data_batch[:, 0]]).T, (1, negsamples.shape[1] + 1)))
                r_idx = torch.tensor(np.tile(np.array([data_batch[:, 1]]).T, (1, negsamples.shape[1] + 1)))
                e2_idx = torch.tensor(np.concatenate((np.array([data_batch[:, 2]]).T, negsamples), axis=1))
                # e2_idx = e2_idx[:, :self.nneg + 1]  # 确保负样本数为 nneg + 1

                targets = np.zeros(e1_idx.shape)
                targets[:, 0] = 1
                targets = torch.DoubleTensor(targets)

                # ❗❗❗ 新增这一段：把张量转移到 GPU ❗❗❗
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    r_idx = r_idx.cuda()
                    e2_idx = e2_idx.cuda()
                    targets = targets.cuda()

                opt.zero_grad()

                # Forward pass
                predictions = model.forward(e1_idx, r_idx, e2_idx)

                # Compute losses
                # 原有损失
                loss = model.loss(predictions, targets)

                loss.backward()
                opt.step()
                losses.append(loss.item())

            # 计算并保存当前 epoch 的平均 loss
            current_epoch_mean_loss = np.mean(losses)
            epoch_losses_history.append(current_epoch_mean_loss)  # ⭐ 加入总列表

            print(it)
            print(time.time() - start_train)
            print(np.mean(losses))

            model.eval()
            with torch.no_grad():
                # 前 50 轮每 10 轮看一次，后面每 50 轮看一次
                # （把下面这段替换掉你原来的 Eval 保存逻辑）
                if (it <= 90 and it % 10 == 0) or (it > 90 and it % 50 == 0):
                    print(f"Validation at epoch {it}:")
                    hits10, hits3, hits1, mr, mrr = self.evaluate(model, d.valid_data)
                    current_hits = hits10  # 用 Hits@10 作为监控指标

                    if current_hits > self.best_metric:
                        self.best_metric = current_hits
                        self.best_epoch = it
                        self.best_model_state = model.state_dict()
                        patience_counter = 0  # ❗有提升，计数器清零
                        print(f"--> [New Best] Model updated at epoch {it} with Hits@10: {current_hits}")
                    else:
                        patience_counter += 1  # ❗没提升，计数器+1
                        print(f"--> No improvement. Early stopping counter: {patience_counter} / {patience}")

                    # ❗触发早停
                    if patience_counter >= patience:
                        print(f"🚀 Early stopping triggered at epoch {it}! Best epoch was {self.best_epoch}.")
                        break  # 强行跳出 500 轮的 for 循环

        # 训练结束后输出最佳epoch的信息
        print(f"Best valid epoch: {self.best_epoch}, Best Valid Hits@10: {self.best_metric}")

        # ⭐ 新增：训练结束后，将包含 Epoch Loss 的列表保存为 npy 文件
        if not os.path.exists("results"):
            os.makedirs("results")
        np.save("results/loss_random.npy",
                np.array(epoch_losses_history))  # 注意：如果你还没建 results 文件夹，请把路径改回 "loss_random.npy"
        print("✅ 训练 Loss 已成功保存至 results/loss_random.npy")

        # ⭐ 加载 best model
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

        hits10, hits3, hits1, mr, mrr = self.evaluate(model, d.test_data)

        print("\nFinal Test Results:")
        print(f"Hits@10: {hits10}")
        print(f"Hits@3:  {hits3}")
        print(f"Hits@1:  {hits1}")
        print(f"MR:      {mr}")
        print(f"MRR:     {mrr}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="APT-Hier", nargs="?",
                        help="Which dataset to use: FB15k-237 or WN18RR.")
    parser.add_argument("--model", type=str, default="poincare", nargs="?",
                        help="Which model to use: poincare or euclidean.")
    parser.add_argument("--num_iterations", type=int, default=500, nargs="?",
                        help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=1024, nargs="?",
                        help="Batch size.")
    parser.add_argument("--nneg", type=int, default=50, nargs="?",
                        help="Number of negative samples.")
    parser.add_argument("--lr", type=float, default=50, nargs="?",
                        help="Learning rate.")
    parser.add_argument("--dim", type=int, default=256, nargs="?",
                        help="Embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=False, nargs="?",
                        help="Whether to use cuda (GPU) or not (CPU).")

    args = parser.parse_args()
    dataset = args.dataset

    # 修改點：這裡的路徑改為 05_data_重构
    data_dir = "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/05_data_重构/%s/" % dataset

    torch.backends.cudnn.deterministic = True
    seed = 1337
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed)
    d = Data(data_dir=data_dir)

    experiment = Experiment(learning_rate=args.lr, batch_size=args.batch_size,
                            num_iterations=args.num_iterations, dim=args.dim,
                            cuda=args.cuda, nneg=args.nneg, model=args.model,
                            data_dir=data_dir)  # 修改點：傳遞正確的 data_dir 進去
    experiment.train_and_eval()