import numpy as np

""" The file is taken from https://github.com/ibalazevic/TuckER"""

class Data:

    def __init__(self, data_dir="/Users/yulii/Desktop/实验室/实验/model/MuRP/data/", reverse=False):
        self.train_data = self.load_data(data_dir, "train", reverse=reverse)
        self.valid_data = self.load_data(data_dir, "valid", reverse=reverse)
        self.test_data = self.load_data(data_dir, "test", reverse=reverse)
        self.data = self.train_data + self.valid_data + self.test_data  # 训练集+验证集+测试集
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        # 关系去重：训练集出现的关系+验证集出现但训练集未出现+测试集出现但验证集未出现
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="test", reverse=False):
        # ❗❗❗ 在这里定义你真正关心的“攻击偏好”关系名称（请务必核对你的真实数据拼写！）
        PREFERENCE_RELS = {"use", "target", "exploit"}

        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            lines = f.read().strip().split("\n")

            data = []
            for line in lines:
                # 1. 过滤 warning
                if line.startswith("Warning"):
                    continue

                # 2. 分割
                if '\t' in line:
                    parts = line.split('\t')
                else:
                    parts = line.split(' ')

                # 3. 必须是三元组
                if len(parts) != 3:
                    continue

                rel = parts[1]

                # ❗4. 【核心修改：分片逻辑隔离】
                if "train" in data_type:
                    '''
                    # 【训练集逻辑】：保留所有图谱知识，但对核心偏好边进行过采样
                    if rel != "subClassOf":
                        data.extend([parts, parts, parts])  # 核心边 * 3
                    else:
                        data.append(parts)  # 层级边只加 1 次
                    '''
                    data.append(parts)  # 训练集全部保留，给模型提供完整的图结构背景
                else:
                    # 【验证/测试集逻辑】：严格过滤，只考“攻击偏好”！
                    if rel in PREFERENCE_RELS:
                        data.append(parts)
                    # 如果不是偏好关系，直接抛弃，不进入测试集
                # data.append(parts)

            # 5. 反向边（针对上面筛选出的干净数据生成反向边）
            data += [[i[2], i[1] + "_reverse", i[0]] for i in data]

        return data

    # 返回列表：关系名
    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data]))) # 关系去重，排序
        return relations

    # 返回列表：实体名
    def get_entities(self, data):
        # 打印长度不足的元素
        for i, d in enumerate(data):
            if len(d) != 3:
                print(f"Invalid element at index {i}: {d}, Length: {len(d)}")

        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data]))) # sorted打乱头实体、尾实体顺序？
        return entities
