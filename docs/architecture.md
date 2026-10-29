# 架构设计

cloverforge 是一个**零依赖**的扩散语言模型（dLLM）参考实现，用纯标准库复现
"整体生成、反复修正"这一与自回归截然不同的范式。

## 分层结构

```
tokenizer      词表与编解码（60 符号 + 4 特殊 token）
world          确定性 ground-truth（序列 oracle + 排序 oracle）
backbone       去噪器网络：Embedding -> (Attention + MoE) x N -> LM Head
moe            sparse MoE：top-k 路由 + 专家 FFN + 负载均衡辅助损失
diffusion      离散扩散：前向掩码 / 反向去噪 / 置信度重掩码 / 采样
decision       联合决策：独立模式 vs 全局统筹（Copeland 总序）
train          训练：数据 / 损失 / 优化器 / skill 训练器
eval           评估：指标 / 基准 / 解码步数加速分析
serving        标准库 HTTP 服务（/v1/generate /v1/decision）
```

## 核心设计取舍

1. **离散扩散而非连续高斯扩散**：文本是离散 token，前向过程用"掩码"（置为
   `MASK`）而不是加高斯噪声；反向过程是"并行预测所有被掩码位置，再按置信度
   逐批揭晓"。
2. **skill 作为可训练代理**：训练 350 亿 MoE 超出参考实现范围，因此用标量
   `skill` 建模"模型已学会的程度"。skill=0.5 时预测接近随机，skill=1.0 时
   精确重建——这让训练/基准的动力学真实、可复现、可证伪。
3. **可证伪的单调性**：所有基准指标（重建准确率、困惑度、MoE 负载均衡损失、
   决策冲突数、联合一致度）都随 skill 单调变化，用代码断言而非肉眼判断。
