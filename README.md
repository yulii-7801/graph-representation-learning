# APT Attack Preference Prediction (APT攻击偏好预测实验)

## 📌 项目简介
本项目针对高级持续性威胁（APT）场景，构建了一套**基于两阶段表示学习（Two-stage Representation Learning）的APT攻击偏好预测**模型框架。

与传统的通用图谱任务不同，本项目严格聚焦于**攻击偏好预测**，通过引入**高层抽象节点（High-level Abstract Node）**，并结合双空间（欧几里得空间与双曲空间）的几何特性，对 APT 组织的攻击路径和偏好进行建模。

*注：本项目代码为核心逻辑展示，在当前实验设置下，模型在攻击偏好预测任务上验证了双空间表示的有效性。*

## 🛠 技术栈
* **核心语言与框架:** Python 3.12.13, PyTorch 2.10 (CUDA 12.8)
* **图谱与表示学习:** PyKEEN, NetworkX
* **数据处理与分析:** Pandas, NumPy
