import os
from collections import Counter

# 1. 你刚发我的本地绝对路径
base_path = '/Users/yulii/LocalFiles/Projects_各种实验项目毕设_代码区/Grad_毕设/PythonProject_重来/04_data/APT-Base'

# 2. 自动拼接三个数据集文件的完整路径
file_paths = [
    os.path.join(base_path, 'train.txt'),
    os.path.join(base_path, 'valid.txt'),
    os.path.join(base_path, 'test.txt')
]

relations = []
total_triples = 0

print("🔍 开始扫描图谱数据...\n")

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"⚠️ 找不到文件：{file_path}，将跳过。")
        continue

    print(f"正在读取 {os.path.basename(file_path)} ...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 3:
                    # 【重要提醒】：标准的知识图谱大多是 (头实体, 关系, 尾实体)
                    # 如果你的数据格式是 (头实体, 尾实体, 关系)，请把下面的 parts[1] 改成 parts[2]
                    relation = parts[1]
                    relations.append(relation)
                    total_triples += 1
    except Exception as e:
        print(f"读取 {os.path.basename(file_path)} 时出错：{e}")

# 3. 统计并排序
counter = Counter(relations)
sorted_relations = counter.most_common()

# 4. 打印最终结果
print("\n=== 🎯 APT2K 全局真实关系分布（最终版） ===")
for rel, count in sorted_relations:
    print(f"{rel}: {count}")

print("-" * 40)
print(f"总三元组数量: {total_triples}")
print(f"总关系种类数: {len(sorted_relations)}")
print("=========================================")