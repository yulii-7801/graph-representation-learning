# extract_targets.py
import os

def load_triples(file_path):
    triples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                triples.append((parts[0], parts[1], parts[2]))
    return triples

# 1. 假设你的层级约束关系叫 'subClassOf' (如果叫别的请修改)
HIER_RELATION = 'subClassOf'

# 2. 读取数据 (路径替换为你实际的路径)
hier_train = load_triples('APT-Hier/train.txt')
hier_valid = load_triples('APT-Hier/valid.txt')
hier_test = load_triples('APT-Hier/test.txt')

all_hier_triples = hier_train + hier_valid + hier_test

# 3. 筛选候选实体 E
candidates_E = set()
for h, r, t in all_hier_triples:
    if r == HIER_RELATION:
        candidates_E.add(h)
print(f"✅ 找到 {len(candidates_E)} 个层级头实体 (E)")

# 4. 在测试集中构造预测任务
target_tasks = []
for h, r, t in hier_test:  # 必须从 test 里面找
    if t in candidates_E and r != HIER_RELATION:
        target_tasks.append(f"{h}\t{r}\t{t}\n")

# 5. 保存下来，供后续打分使用
with open('target_tasks.txt', 'w', encoding='utf-8') as f:
    f.writelines(target_tasks)
print(f"✅ 已将 {len(target_tasks)} 个目标任务保存至 target_tasks.txt")