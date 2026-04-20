import os


def generate_dicts(data_dir):
    entities = set()
    relations = set()

    # 遍历你的数据集文件提取所有节点和边
    for split in ['train.txt', 'valid.txt', 'test.txt']:
        file_path = os.path.join(data_dir, split)
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    entities.add(parts[0])
                    relations.add(parts[1])
                    entities.add(parts[2])

    # 写入 entities.dict (格式: ID \t 实体名)
    with open(os.path.join(data_dir, 'entities.dict'), 'w', encoding='utf-8') as f:
        for idx, ent in enumerate(sorted(list(entities))):
            f.write(f"{idx}\t{ent}\n")

    # 写入 relations.dict (格式: ID \t 关系名)
    with open(os.path.join(data_dir, 'relations.dict'), 'w', encoding='utf-8') as f:
        for idx, rel in enumerate(sorted(list(relations))):
            f.write(f"{idx}\t{rel}\n")

    print(f"✅ {data_dir} 字典生成完毕! 包含 {len(entities)} 个实体, {len(relations)} 种关系。")


# 一次性为你两份数据生成字典
generate_dicts("./dataset/APT-Flat")
generate_dicts("./dataset/APT-Hier")