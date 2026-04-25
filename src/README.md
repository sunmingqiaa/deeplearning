# Source Code

这里放可复用代码，不放一次性实验脚本。

## 模块建议

- `dl_playground/data/`：数据加载、预处理。
- `dl_playground/models/`：模型结构。
- `dl_playground/training/`：训练循环、checkpoint。
- `dl_playground/evaluation/`：指标计算、评估逻辑。
- `dl_playground/utils/`：随机种子、日志、配置工具。

## 约定

先在 `labs/` 中验证想法，稳定后再抽到 `src/`。
