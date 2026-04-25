# Tests

这里存放 pytest 测试，用于验证训练流程中的关键假设。

## 重点测试

- 输入输出 shape。
- Dataset 和 DataLoader 是否返回预期字段。
- loss 是否能在小 batch 上下降。
- checkpoint 保存和加载是否一致。
- 推理接口是否返回合法结果。

## 示例命令

不安装 pytest 时，先跑标准库 smoke test：

```bash
python3 tests/run_smoke_tests.py
```

安装 pytest 后，可以再补充正式测试并运行：

```bash
pytest -q
```
