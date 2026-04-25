# 线性回归与梯度下降

## 一句话理解

线性回归用一条直线拟合输入和输出之间的关系，梯度下降负责一步步调整参数，让预测误差变小。

## 核心知识点

- 模型：`y_hat = w * x + b`
- 参数：`w` 是斜率，`b` 是偏置。
- 损失函数：均方误差 `MSE = mean((y_hat - y)^2)`。
- 梯度：损失函数对参数的变化方向。
- 参数更新：`param = param - learning_rate * gradient`。

## 重点理解

学习率过小，loss 下降慢；学习率过大，loss 可能震荡甚至发散。训练不是“模型自己变聪明”，而是每一步根据梯度修正参数。

## 对应实验

```bash
python3 labs/week01-linear-regression/scripts/train.py
```

实验会生成线性数据，手写 MSE 和梯度下降，并把指标写入：

```text
labs/week01-linear-regression/outputs/metrics.json
```

## 验收问题

- `w` 和 `b` 分别代表什么？
- 为什么 MSE 可以衡量回归误差？
- 为什么参数要沿负梯度方向更新？
- learning rate 改成 `0.1` 或 `0.0001` 会发生什么？
