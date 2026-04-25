# Week 03 PyTorch Autograd 与训练模板

## 目标

掌握 PyTorch 标准训练流程。

## 练习

- 使用 `torch.nn.Module` 定义模型。
- 使用 `Dataset` 和 `DataLoader` 加载数据。
- 编写训练循环和评估循环。
- 保存和加载 checkpoint。

## 运行

```bash
python3 labs/week03-pytorch-autograd/scripts/tiny_autograd.py
```

知识点文档：`docs/03-pytorch基础/01-autograd-training-loop.md`

## 验收标准

- 能解释 `zero_grad()`、`backward()`、`step()`。
- checkpoint 加载后推理结果一致。
- 训练和评估逻辑分开。
