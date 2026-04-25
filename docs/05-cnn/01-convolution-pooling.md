# CNN 卷积与池化

## 一句话理解

卷积用局部窗口提取图像特征，池化压缩特征图并保留关键信息。

## 核心知识点

- 卷积核：在局部区域上滑动的小矩阵。
- stride：卷积核每次移动的步长。
- padding：在边缘补值，控制输出尺寸。
- channel：输入或输出的通道数。
- 特征图：卷积后的二维响应。
- 池化：常见是 max pooling。

## 输出尺寸

不考虑 dilation 时：

```text
out = floor((input + 2 * padding - kernel) / stride) + 1
```

## 对应实验

```bash
python3 labs/week06-cnn-cifar10/scripts/train.py
```

当前实验用合成 8x8 图像演示卷积特征提取和分类，重点看卷积输出 shape 和特征维度。

## 验收问题

- 8x8 输入经过 2x2 valid convolution 后输出多大？
- pooling 为什么会降低空间尺寸？
- CNN 为什么比纯 MLP 更适合图像？
- filter 学到的通常是什么类型的局部模式？
