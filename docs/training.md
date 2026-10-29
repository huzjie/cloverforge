# 训练

## 目标函数

扩散/掩码语言模型的训练目标是被掩码位置的交叉熵：

    L = -mean(log softmax(logits_i)[target_i])   (i 遍历被掩码位置)

## 训练器

`train.Trainer` 以单调递增的方式把 `skill` 推向 1（delta 恒正），模拟模型
逐步学会"从上下文重建被掩码 token"。`train.optimizer` 提供 SGD / AdamW 用于
可选的真实权重路径。

## 为什么用 skill 代理

参考实现无法真训 35B MoE，但训练要"真实参与打分"：skill 进入 logits
（boost = (skill-0.5)*2），训练信号单调趋 1，skill=0.5 时 boost=0（随机基线）。
这保证了"训练有效"是**可测量**的，而不是嘴上说说。
