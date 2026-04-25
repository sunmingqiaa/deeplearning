# Attention 与 Transformer

## 一句话理解

Attention 让每个 token 根据相关性从其他 token 中取信息，Transformer 则把这种机制堆叠成强大的序列建模结构。

## 核心知识点

- Query：当前位置想查什么。
- Key：每个位置提供什么索引。
- Value：每个位置真正被聚合的信息。
- Attention score：Query 和 Key 的相似度。
- Softmax：把相似度转成权重。
- Self-Attention：Q、K、V 都来自同一个序列。

## 核心公式

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

## 对应实验

```bash
python3 labs/week09-attention-transformer/scripts/attention_demo.py
```

实验会输出 Q/K/V shape、attention matrix shape 和 attention weights。

## 验收问题

- 为什么要除以 `sqrt(d_k)`？
- attention weights 每一行为什么和为 1？
- self-attention 和普通 seq2seq attention 有什么区别？
- 多头注意力为什么要拆成多个 head？
