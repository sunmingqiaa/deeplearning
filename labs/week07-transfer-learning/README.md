# Week 07 迁移学习

## 目标

理解预训练模型、冻结参数和微调。

## 练习

- 使用 torchvision 预训练模型。
- 替换分类头。
- 对比冻结 backbone 和全量微调。

## 运行

```bash
python3 labs/week07-transfer-learning/scripts/transfer_demo.py
```

知识点文档：`docs/05-cnn/02-transfer-learning.md`

## 验收标准

- 能说明哪些参数参与训练。
- 能保存最佳验证集模型。
- 能解释小数据集上迁移学习的优势。
