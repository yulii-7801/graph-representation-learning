import os


def fix_spaces_to_underscore(directory):
    files_to_clean = ['train.txt', 'valid.txt', 'test.txt']

    for filename in files_to_clean:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            # 严格按照 Tab 拆开
            parts = line.strip('\n').split('\t')
            if len(parts) == 3:
                # 把每部分内部的空格替换为下划线
                h = parts[0].strip().replace(' ', '_')
                r = parts[1].strip().replace(' ', '_')
                t = parts[2].strip().replace(' ', '_')
                fixed_lines.append(f"{h}\t{r}\t{t}\n")
            else:
                fixed_lines.append(line)

        # 覆写原文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)

        print(f"✅ {filename} 处理完毕！实体内部的空格已全部替换为下划线。")


fix_spaces_to_underscore(".")