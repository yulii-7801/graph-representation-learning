import numpy as np
import matplotlib.pyplot as plt

# 读取你刚才跑出来的两个 Loss 数据
loss_random = np.load("results/loss_random.npy")
loss_euc_init = np.load("results/loss_euc_init.npy")

epochs = range(1, len(loss_random) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_random, label="MuRP (Random Init)", color='red', linestyle='--', alpha=0.8)
plt.plot(epochs, loss_euc_init, label="Ours (NoGE + Exp Map Init)", color='blue', linewidth=2)

plt.title("Training Loss Convergence: Random vs. Euclidean Initialization", fontsize=14)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Training Loss", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

# 保存高清图片，直接贴进你的论文和 PPT
plt.savefig("figures/loss_convergence.png", dpi=300)
plt.show()