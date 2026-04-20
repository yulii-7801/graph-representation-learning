import matplotlib.pyplot as plt

# 1. 真实数据
relations = ['hasType', 'target', 'exploit', 'indicate', 'use',
             'hasAttackTime', 'hasAlias', 'has', 'foundBy', 'belongTo', 'evolveTo']
counts = [1964, 314, 195, 147, 108, 67, 60, 48, 48, 26, 8]

# 2. 解决 Mac 系统中文显示问题
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 3. 创建画布，稍微增加高度给下方文字留空间 (10宽, 6高)
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 4. 绘制柱状图
colors = ['#5b9bd5'] * len(relations)
colors[0] = '#c00000'
bars = ax.bar(relations, counts, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)

# ================= 关键修复区 =================

# 修复一：强制把 Y 轴的上限拉高到 2200，给“1964”留出充足的“头顶空间”！
ax.set_ylim(0, 2200)

# 修复二：在柱子上方添加数值，稍微调小字号使其更精致
ax.bar_label(bars, padding=3, fontsize=10, color='black')

# 修复三：强制显式添加横轴和纵轴的说明文字
ax.set_xlabel('关系类型', fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel('数 量', fontsize=12, fontweight='bold', labelpad=10)

# ==============================================

# X轴标签旋转 45 度，对齐刻度
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(fontsize=11)

# 添加横向网格线
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True)

# 修复四：在保存前，强制系统重新计算所有元素的边界，把外面被切掉的字全拉回画布里
plt.tight_layout()

# 保存图片
save_path = 'apt2k_relation_distribution_final.png'
plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"完美图表已生成：{save_path}，头顶不破框，底下不切字！")