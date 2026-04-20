import os
import shutil


def wash_data(directory):
    files_to_clean = ['train.txt', 'valid.txt', 'test.txt']

    for filename in files_to_clean:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            continue

        backup_path = filepath + ".bak"
        shutil.copy(filepath, backup_path)

        valid_lines = []
        warning_count = 0
        fixed_space_count = 0  # 新增：記錄把空格成功轉為 Tab 的數量

        with open(backup_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 嚴格按 Tab 切割
                parts = line.split('\t')

                # 如果 Tab 切割出來不是 3 塊，說明可能是用空格分割的
                if len(parts) != 3:
                    parts = line.split()  # 退化為按任何空格切割
                    if len(parts) == 3:
                        fixed_space_count += 1  # 成功透過空格切割救回

                # 最終檢查與強行 Tab 拼裝
                if len(parts) == 3:
                    clean_line = f"{parts[0]}\t{parts[1]}\t{parts[2]}\n"
                    valid_lines.append(clean_line)
                else:
                    warning_count += 1
                    print(f"[Warning] 丟棄 {filename} 第 {line_num} 行異常數據: {line}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)

        print(f"✅ {filename} 清洗完畢！保留 {len(valid_lines)} 條，丟棄 {warning_count} 條。")
        print(f"   🔧 其中有 {fixed_space_count} 條數據成功從「空格分割」被統一修復為「Tab 分割」。")


print("--- 開始清洗 APT-Hier ---")
wash_data(".")