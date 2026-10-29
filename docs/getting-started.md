# 快速开始

```bash
# 零依赖，直接跑
python -m cloverforge doctor        # 冒烟自检
python -m cloverforge bench         # 全量基准（单调性校验）
python -m cloverforge train         # 训练（skill 0.5 -> 1.0）
python -m cloverforge generate --prompt clover24   # 并行扩散生成
python -m cloverforge decision --skill 0.8         # 联合决策演示
python -m cloverforge serve --port 8788            # 启动 HTTP 服务
```

HTTP 端点：`/health`、`/v1/models`、`POST /v1/generate`、`POST /v1/decision`。

## 单元测试

```bash
python -m unittest discover -s tests -t .
```
