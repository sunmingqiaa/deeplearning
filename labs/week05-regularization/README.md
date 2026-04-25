# Week 05 正则化与训练稳定性

## 目标

理解过拟合以及常见缓解方式。

## 练习

- 对比无正则化、Dropout、weight decay。
- 对比 BatchNorm 前后的训练曲线。
- 尝试在小数据集上制造过拟合。

## 运行

```bash
python3 labs/week05-regularization/scripts/compare_regularization.py
```

知识点文档：`docs/04-神经网络基础/02-regularization.md`

## 验收标准

- 能识别 train/val 曲线中的过拟合。
- 能说明 Dropout 在训练和评估时的差异。
- 能记录至少两组配置对比。
