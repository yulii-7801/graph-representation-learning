import os
import json


def generate_json_dicts(data_dir):
    entities = set()
    relations = set()

    # 1. 遍历三份原始数据，提取所有节点和关系
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

    # 2. 强行按字母顺序排序，保证每次生成的 ID 映射绝对一致
    entity_list = sorted(list(entities))
    relation_list = sorted(list(relations))

    # 3. 构建字典：{"实体名": ID}
    entity2id = {ent: idx for idx, ent in enumerate(entity_list)}
    relation2id = {rel: idx for idx, rel in enumerate(relation_list)}

    # 4. 写入 json 文件
    with open(os.path.join(data_dir, 'entity2id.json'), 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False, indent=4)

    with open(os.path.join(data_dir, 'relation2id.json'), 'w', encoding='utf-8') as f:
        json.dump(relation2id, f, ensure_ascii=False, indent=4)

    print(f"✅ {data_dir} 处理完毕: {len(entity2id)} 个实体, {len(relation2id)} 个关系")


# 执行生成
generate_json_dicts(
    "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/04_data/APT-Flat")
generate_json_dicts(
    "/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/04_data/APT-Hier")