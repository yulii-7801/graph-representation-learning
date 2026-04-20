import os

# ================= 配置区域 =================
INPUT_FILE = 'train.txt'
OUTPUT_FILE = 'hierarchy_edges.txt'
# ===========================================

FINE_TO_META_MAP = {
    'Apt': 'Strategic_Core', 'Campaign': 'Strategic_Core', 'Secteam': 'Strategic_Core',
    'Time': 'Context_Dimension', 'Loc': 'Context_Dimension', 'Field': 'Context_Dimension',
    'Os': 'Target_System',
    'Act': 'Methodology', 'Vulname': 'Methodology', 'Vulid': 'Methodology',
    'Mal': 'Arsenal_Heavy', 'Tool': 'Arsenal_Heavy', 'File': 'Arsenal_Heavy',
    'Ip': 'Infra_Network', 'Dom': 'Infra_Network', 'Url': 'Infra_Network', 'Email': 'Infra_Network',
    'Prot': 'Infra_Network',
    'Md5': 'Digital_Trace', 'Sha1': 'Digital_Trace', 'Sha2': 'Digital_Trace'
}


def generate_hierarchy_debug():
    print(f"[*] (V4 终极版) 开始扫描文件: {INPUT_FILE}")

    if not os.path.exists(INPUT_FILE):
        print(f"[!] 错误: 找不到文件 {INPUT_FILE}，请确认它和脚本在同一个文件夹下！")
        return

    generated_count = 0
    seen_entities = set()

    print("-" * 30)
    print("[显微镜模式] 正在检查前 5 行的原始内容...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue

            # 1. 尝试 Tab 切分
            parts = line.split('\t')

            # [调试] 打印前5行的解析结果，帮我们看清到底发生了什么
            if line_num <= 5:
                print(f"行 {line_num} 切分后 ({len(parts)}段): {parts}")

            if len(parts) != 3:
                continue

            # 2. 关键修复：对所有部分都做 .strip()，去除 Tab 旁边的潜在空格
            head = parts[0].strip()
            relation = parts[1].strip()  # <--- 这里之前可能读到了 " hasType "
            tail = parts[2].strip()

            # [调试] 如果这行看起来像 hasType 但没匹配上，告诉我是为什么
            if 'hastype' in relation.lower() and relation != 'hasType':
                if line_num <= 10:
                    print(
                        f"[发现潜在问题] 行 {line_num} 的关系列是 '{relation}'，不等于 'hasType' (可能是大小写或特殊符号)")

            # 3. 匹配逻辑
            if relation == 'hasType':
                if tail in FINE_TO_META_MAP:
                    meta_type = FINE_TO_META_MAP[tail]

                    if head not in seen_entities:
                        f_out.write(f"{head}\tsubClassOf\t{meta_type}\n")
                        seen_entities.add(head)
                        generated_count += 1
                else:
                    # 如果有 hasType 但类型不在字典里（比如大小写不匹配）
                    if line_num <= 20:  # 只报前20个错
                        # print(f"[跳过] 未知类型: '{tail}' (在行 {line_num})")
                        pass

    print("-" * 30)
    print(f"[*] 处理完成!")
    print(f"[*] 成功生成: {generated_count} 条数据")

    if generated_count > 0:
        print(f"[*] 输出文件: {OUTPUT_FILE}")
        # 预览前3行结果
        print("[预览结果]")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            for i in range(3):
                line = f.readline()
                if line: print(f"  {line.strip()}")
    else:
        print("[!] 依然是 0 条？请把上面 [显微镜模式] 的输出截图发给我！")


if __name__ == '__main__':
    generate_hierarchy_debug()