# Contributing

欢迎提交 issue / PR。请保持：

1. 零第三方运行时依赖（标准库优先）。
2. 任何新基准指标必须**随 skill 单调**，并在 `bench` 中加断言。
3. 生成器脚本在 `scripts/`，成品在 `_build/`（见仓库根 README）。
