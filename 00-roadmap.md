# 深度学习系统学习路线

面向背景：大数据开发，了解基础概念，希望系统掌握深度学习的核心机制、训练流程和工程实践。

## 总目标

- 能解释常见深度学习概念，而不是只背定义。
- 能用 PyTorch 独立完成数据加载、模型定义、训练、评估和保存。
- 能写基础测试判断模型训练流程是否正常。
- 能把实验结果沉淀成可复查的文档。
- 能把深度学习和大数据场景结合，例如推荐、特征处理、离线预测。

## 阶段安排

| 周次 | 主题 | 核心概念 | 实验产物 |
| --- | --- | --- | --- |
| Week 01 | 线性回归与梯度下降 | tensor、loss、梯度、学习率 | 手写线性回归 |
| Week 02 | 二分类与逻辑回归 | sigmoid、交叉熵、分类阈值 | 逻辑回归二分类 |
| Week 03 | PyTorch 基础 | autograd、Module、optimizer、DataLoader | PyTorch 训练模板 |
| Week 04 | MLP 与 MNIST | 全连接层、激活函数、反向传播 | MLP 手写数字分类 |
| Week 05 | 正则化与训练稳定性 | Dropout、BatchNorm、初始化、过拟合 | 对比正则化效果 |
| Week 06 | CNN 基础 | 卷积、池化、特征图、感受野 | CIFAR-10 分类 |
| Week 07 | 迁移学习 | 预训练、微调、冻结参数 | 图像分类微调 |
| Week 08 | 文本分类 | token、embedding、序列建模 | 文本情感分类 |
| Week 09 | Attention 与 Transformer | attention、self-attention、位置编码 | 简化 Transformer 实验 |
| Week 10 | 推荐系统入门 | sparse feature、embedding、召回/排序 | MovieLens 推荐实验 |
| Week 11 | 模型服务 | checkpoint、推理接口、批量预测 | FastAPI 推理服务 |
| Week 12 | 综合项目 | 数据、训练、评估、部署闭环 | 一个完整小项目 |

## 每周工作流

1. 阅读本周主题资料。
2. 在 `docs/` 下写概念笔记。
3. 在 `labs/weekXX-*` 下完成最小实验。
4. 在 `tests/` 下补充 shape、loss、checkpoint 等测试。
5. 在 `experiments/YYYY-MM-DD-*` 下记录实验配置和指标。
6. 在 `reports/weekly/` 下写周复盘。

## 验收标准

每周结束时至少回答这些问题：

- 这个模型或概念解决什么问题？
- 输入和输出的 shape 是什么？
- loss 为什么这样设计？
- loss 是否真的下降？
- 训练集和验证集表现是否一致？
- 结果能否复现？
- 哪些地方还没理解透？

## 推荐资料

- PyTorch Tutorials: https://docs.pytorch.org/tutorials/index.html
- Dive into Deep Learning: https://en.d2l.ai/
- CS231n: https://cs231n.stanford.edu/
- fast.ai: https://course.fast.ai/
