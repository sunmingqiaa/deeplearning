# Week 09 Attention 与 Transformer

## 目标

理解 attention 机制和 Transformer Encoder 的基本结构。

## 练习

- 实现或调用一个简化 self-attention。
- 打印 Q、K、V 和 attention score 的 shape。
- 用 Transformer Encoder 做小型文本分类。

## 运行

```bash
python3 labs/week09-attention-transformer/scripts/attention_demo.py
```

知识点文档：`docs/06-rnn-attention-transformer/02-attention-transformer.md`

## 验收标准

- 能解释 attention score 的含义。
- 能说明多头注意力为什么要拆分 head。
- 能完成一次端到端训练。
