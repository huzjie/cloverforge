# HTTP API

## GET /health
`{"status": "ok", "name": "cloverforge"}`

## GET /v1/models
返回可用模型列表。

## POST /v1/generate
```json
{"prompt": "clover24", "skill": 0.95, "gen_len": 16, "steps": 8}
```
返回 `{"prompt", "generated", "tokens", "skill", "steps"}`。

## POST /v1/decision
```json
{"n_items": 8, "skill": 0.8, "mode": "joint"}
```
返回 `{"mode", "order", "conflicts", "agreement", ...}`。
