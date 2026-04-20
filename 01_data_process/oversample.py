import os

# 1. 配置文件路徑
input_file = "train.txt"
output_file = "train_oversampled.txt"

# 2. 設置規則：不需要放大的關係（即你新增的層級關係）
target_relation = "subClassOf"
# 其他原始業務關係的放大倍數
oversample_ratio = 3

# 3. 執行重採樣
with open(input_file, 'r', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8') as f_out:
    for line in f_in:
        line = line.strip()
        if not line:
            continue

        # 解析三元組 (嚴格依賴 Tab 鍵分割)
        parts = line.split('\t')

        if len(parts) == 3:
            head, relation, tail = parts

            # 如果是 subClassOf，只寫入 1 次；否則寫入 3 次
            if relation == target_relation:
                f_out.write(line + '\n')
            else:
                for _ in range(oversample_ratio):
                    f_out.write(line + '\n')
        else:
            # 異常行直接保留
            f_out.write(line + '\n')

print(f"✅ 重採樣完成！非 '{target_relation}' 的關係已放大 {oversample_ratio} 倍。")
print(f"請將原來的 {input_file} 備份，並將 {output_file} 重命名為 {input_file} 供模型使用。")