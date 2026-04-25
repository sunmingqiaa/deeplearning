# 逻辑回归与二分类

## 一句话理解

逻辑回归把线性模型的输出通过 sigmoid 转成概率，再根据阈值判断类别。

## 核心知识点

- 线性打分：`z = w1 * x1 + w2 * x2 + b`
- 概率映射：`p = sigmoid(z)`
- 损失函数：二元交叉熵。
- 阈值：默认 `p >= 0.5` 判为正类。
- 指标：accuracy、precision、recall、F1、混淆矩阵。

## 重点理解

accuracy 不一定够用。比如告警、风控、故障预测这些场景，precision 和 recall 的取舍更重要。调阈值不是改模型本身，而是改业务决策边界。

## 对应实验

```bash
python3 labs/week02-logistic-regression/scripts/train.py
```

实验会输出 `0.5` 和 `0.7` 两种阈值下的指标，结果写入：

```text
labs/week02-logistic-regression/outputs/metrics.json
```

## 验收问题

- sigmoid 为什么能表示概率？
- 交叉熵和 MSE 在分类任务上有什么差异？
- precision 高但 recall 低意味着什么？
- 阈值从 `0.5` 提到 `0.7` 后，指标为什么会变？
