# Week 06 CNN 与 CIFAR-10

## 目标

理解卷积、池化和图像特征提取。

## 练习

- 训练一个小型 CNN。
- 打印每层输出 shape。
- 对比 MLP 和 CNN 在图像任务上的表现。

## 运行

```bash
python3 labs/week06-cnn-cifar10/scripts/train.py
```

知识点文档：`docs/05-cnn/01-convolution-pooling.md`

## 验收标准

- 能计算卷积层输出尺寸。
- 能解释 channel、kernel、stride、padding。
- CNN 指标应明显优于同等简单 MLP。
