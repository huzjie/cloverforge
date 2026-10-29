# 基准说明

`python -m cloverforge bench` 输出四个维度的指标，并断言全部随 skill 单调：

| 维度 | 指标 | 方向 |
|---|---|---|
| 重建 | recon_acc / recon_ppl | 准确率↑、困惑度↓ |
| 生成 | gen_acc（多步扩散） | ↑ |
| MoE | aux / max_fraction | ↓（负载趋于均衡） |
| 决策 | ind_cycle / ind_acc / joint_tau | 冲突↓、一致度↑ |

speedup = gen_len / steps 是纯算法量，与 skill 无关，随序列长度线性增长。
