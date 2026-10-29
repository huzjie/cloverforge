# 稀疏 Mixture-of-Experts

Clover 采用 MoE 架构：总参数 350 亿、激活 30 亿，即每个 token 只激活 top-k 个
专家，推理成本远低于稠密网络。

## top-k 路由

对每个 token 的隐向量 h，计算门控 logits = Wg @ h（+ 噪声），取 top-k 专家，
用 softmax 归一化它们的权重，输出 = Σ weight_k * expert_k(h)。

## 负载均衡辅助损失

MoE 的经典问题：路由器会退化成"只用一个专家"。Switch-Transformer 的解法是加
辅助损失：

    L_aux = n_experts * Σ_i f_i * p_i

其中 f_i 是专家 i 的实际负载比例，p_i 是门控的平均概率。L_aux 把过热的专家
拉回均衡。本实现里"热专家偏差"随 skill 收缩，正是模拟这条损失的作用。
