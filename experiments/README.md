# Experiments

每次实验新建一个目录，命名建议：

```text
YYYY-MM-DD-topic-short-name/
```

示例：

```text
2026-05-01-mlp-mnist-baseline/
```

## 实验目录结构

```text
YYYY-MM-DD-topic/
  config.yaml
  metrics.json
  notes.md
  plots/
  checkpoints/
```

## 记录要求

- 配置必须能复现主要结果。
- 指标要记录最终值和最好值。
- notes 里记录异常现象和判断。
- 不要把大模型文件直接混在文档目录里。
