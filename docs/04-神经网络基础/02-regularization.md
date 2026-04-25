# 正则化与训练稳定性

## 一句话理解

正则化通过限制模型过度记忆训练数据，提升模型在新数据上的表现。

## 核心知识点

- 过拟合：训练集好，验证集差。
- 欠拟合：训练集和验证集都差。
- L2/weight decay：惩罚过大的权重。
- Dropout：训练时随机关闭一部分神经元。
- Early stopping：验证集不再提升时停止训练。

## 重点理解

正则化通常会让训练集指标略差，但可能让验证集指标更稳。它不是为了让训练 loss 最低，而是为了让模型泛化更好。

## 对应实验

```bash
python3 labs/week05-regularization/scripts/compare_regularization.py
```

实验会比较 baseline 和 `dropout + L2` 两组结果。

## 验收问题

- 为什么训练集准确率高不一定是好事？
- Dropout 在训练和推理阶段有什么区别？
- weight decay 惩罚的是什么？
- 如何从 train/val 曲线判断过拟合？
