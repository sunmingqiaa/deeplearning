# Autograd 与训练循环

## 一句话理解

Autograd 会记录前向计算图，`backward()` 根据链式法则自动计算每个参数的梯度。

## 核心知识点

- 前向传播：根据当前参数计算预测和 loss。
- 反向传播：从 loss 出发计算每个参数的梯度。
- 清空梯度：避免上一轮梯度累加到这一轮。
- 参数更新：optimizer 根据梯度更新参数。
- 训练循环顺序：`zero_grad -> forward -> loss -> backward -> step`。

## 重点理解

PyTorch 里的 `requires_grad=True` 表示这个 tensor 需要参与梯度计算。`loss.backward()` 不是在训练模型，它只是在计算梯度；真正改变参数的是 `optimizer.step()`。

## 对应实验

当前环境还没有安装 PyTorch，所以先用一个标准库 tiny autograd 理解机制：

```bash
python3 labs/week03-pytorch-autograd/scripts/tiny_autograd.py
```

输出写入：

```text
labs/week03-pytorch-autograd/outputs/metrics.json
```

## 验收问题

- 为什么每轮训练前要清空梯度？
- `backward()` 和参数更新是什么关系？
- 如果不调用 `step()`，参数会不会变化？
- 链式法则在计算图里起什么作用？
