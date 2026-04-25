# 推荐系统与 Embedding

## 一句话理解

推荐模型常把用户、物品、上下文等稀疏 ID 映射成 embedding，再学习它们之间的匹配关系。

## 核心知识点

- 稀疏特征：用户 ID、物品 ID、类目 ID 等。
- User embedding：用户偏好向量。
- Item embedding：物品属性向量。
- 矩阵分解：用用户向量和物品向量的内积预测评分或偏好。
- 召回：从海量物品中找候选。
- 排序：对候选物品精排。

## 重点理解

Embedding 本质上是把离散 ID 变成可学习参数。它能把“用户喜欢什么”和“物品像什么”压缩到同一个向量空间。

## 对应实验

```bash
python3 labs/week10-recommender-movielens/scripts/train.py
```

实验会训练一个合成评分数据上的矩阵分解模型，并输出 RMSE 和推荐结果。

## 验收问题

- user embedding 和 item embedding 的维度为什么要一致？
- 内积高代表什么含义？
- 召回和排序的目标有什么区别？
- 大数据离线训练时，样本构造最容易出什么问题？
