class Data:

    def __init__(self, data_dir="/Users/yulii/LocalFiles/实验室/实验/data/APT/6/"):
        self.train_data = self.load_data(data_dir, "train")
        self.valid_data = self.load_data(data_dir, "valid")
        self.test_data = self.load_data(data_dir, "test")
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.valid_relations = self.get_relations(self.valid_data)
        self.test_relations = self.get_relations(self.test_data)
        self.relations = self.train_relations + [i for i in self.valid_relations \
                if i not in self.train_relations] + [i for i in self.test_relations \
                if i not in self.train_relations]

    def load_data(self, data_dir, data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            raw_lines = f.read().strip().split("\n")

            data = []
            for line_idx, line in enumerate(raw_lines):
                # 1. 过滤掉完全空白的行
                if not line.strip():
                    continue

                # 2. 严格按 Tab 切分
                parts = line.split('\t')

                # 3. 检查切分后的长度是不是完美的三元组 (头, 关系, 尾)
                if len(parts) == 3:
                    data.append(parts)
                else:
                    # 打印出有问题的行，让你知道数据哪里脏了，但不中断程序
                    print(f"⚠️ 警告: {data_type}.txt 第 {line_idx + 1} 行格式异常，已被过滤 -> {parts}")

            # 4. 关系反向增强
            if reverse:
                data += [[i[2], i[1] + "_reverse", i[0]] for i in data]

        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
