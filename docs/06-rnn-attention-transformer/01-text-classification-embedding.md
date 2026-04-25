# 文本分类与 Embedding

## 一句话理解

文本分类先把词转换成向量表示，再把句子表示输入分类器。

## 核心知识点

- token：文本切分后的基本单位。
- vocab：训练语料中的 token 到 id 的映射。
- embedding：每个 token 对应的可训练向量。
- pooling：把多个 token 向量合成一个句子向量。
- 分类头：根据句子向量输出类别概率。

## 重点理解

Embedding 不是人工写死的词义表，而是在训练中根据任务目标学出来的向量。相同词在不同任务中可能学到不同含义。

## 对应实验

```bash
python3 labs/week08-text-classification/scripts/train.py
```

实验会训练一个 tiny embedding 文本分类器，输出词表大小、embedding 维度、loss 和 accuracy。

## 验收问题

- 为什么文本不能直接输入神经网络？
- vocab 和 embedding matrix 的关系是什么？
- pooling 会丢失什么信息？
- 未登录词应该如何处理？
