import os
import re


def nuke_data(data_dir):
    entities = set()
    relations = set()

    for split in ['train.txt', 'valid.txt', 'test.txt']:
        file_path = os.path.join(data_dir, split)
        if not os.path.exists(file_path): continue

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            line = line.strip('\n')
            if not line: continue

            # 严格按照 Tab 拆开
            parts = line.split('\t')
            if len(parts) >= 3:
                # \s+ 能匹配所有隐形空白字符（全角空格、\xa0、回车等），强行转为下划线
                h = re.sub(r'\s+', '_', parts[0].strip())
                r = re.sub(r'\s+', '_', parts[1].strip())
                t = re.sub(r'\s+', '_', parts[2].strip())

                entities.add(h)
                relations.add(r)
                entities.add(t)

                new_lines.append(f"{h}\t{r}\t{t}\n")

        # 直接覆盖写入最纯净的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

    # 同步重写字典，保证 100% 匹配
    with open(os.path.join(data_dir, 'entities.dict'), 'w', encoding='utf-8') as f:
        for idx, ent in enumerate(sorted(list(entities))): f.write(f"{idx}\t{ent}\n")
    with open(os.path.join(data_dir, 'relations.dict'), 'w', encoding='utf-8') as f:
        for idx, rel in enumerate(sorted(list(relations))): f.write(f"{idx}\t{rel}\n")


print("--- 正在进行核聚变级清洗 ---")
nuke_data('./dataset/APT-Flat')
nuke_data('./dataset/APT-Hier')
print("✅ 所有隐形空白字符已被彻底剿灭，字典已完美同步！")