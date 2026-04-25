# Week 01 线性回归与梯度下降

## 目标

理解模型参数、预测值、损失函数、梯度和参数更新之间的关系。

## 练习

- 用 NumPy 或 PyTorch tensor 构造一组线性数据。
- 手写均方误差。
- 手写梯度下降更新参数。
- 观察 learning rate 对收敛的影响。

## 运行

```bash
python3 labs/week01-linear-regression/scripts/train.py
```

知识点文档：`docs/01-数学基础/01-linear-regression-gradient-descent.md`

## 验收标准

- loss 能稳定下降。
- 学到的参数接近真实参数。
- 能解释学习率过大和过小时的现象。
