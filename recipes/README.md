# 配方（recipes）

recipe 是覆盖默认配置的 YAML 子集，用法：

```bash
python -m cloverforge bench -c recipes/fast.yaml
python -m cloverforge generate -c recipes/quality.yaml --prompt clover24
```

内置极简 YAML 解析器（`cloverforge/utils/yamlish.py`），无需 pyyaml。
