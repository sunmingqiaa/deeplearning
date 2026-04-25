# Week 11 模型服务与批量预测

## 目标

完成训练模型到推理使用的闭环。

## 练习

- 加载 checkpoint 做单条推理。
- 封装 FastAPI 接口。
- 编写批量预测脚本。

## 运行

批量预测：

```bash
python3 labs/week11-model-serving/scripts/batch_predict.py
```

HTTP 推理服务：

```bash
python3 labs/week11-model-serving/scripts/serve_linear_model.py --port 8000
```

知识点文档：`docs/09-部署与推理/01-model-serving.md`

## 验收标准

- 单条推理结果稳定。
- HTTP 接口能返回预测结果。
- 批量预测能写出结果文件。
