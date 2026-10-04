# apibench

给**任何 OpenAI-compatible API** 做基准测试的小工具：延迟、首 token 时间（TTFT）、tokens/秒、多模型横向对比、持续压测。纯 Python 标准库，零依赖，一个文件拎走即用。

## 快速开始

```bash
# 非流式：总延迟 + token 统计
python -m apibench --base-url https://api.openai.com/v1 --model gpt-4o-mini "讲个笑话"

# 流式：TTFT / tokens/s / token 间隔 p50/p95
python -m apibench --base-url http://localhost:8000/v1 --model qwen3 --stream --n 5

# 多模型对比（Markdown 表格输出，可存文件）
python -m apibench --compare gpt-4o-mini,gpt-4o --stream --md result.md

# 列出服务端可用模型
python -m apibench --base-url http://localhost:8000/v1 --models

# 持续压测 60 秒：成功率 + 延迟分位数 + ASCII 直方图
python -m apibench --soak 60 --stream

# 没 key 也能看输出格式
python -m apibench --dry-run --compare m1,m2 --stream
```

API key 优先级：`--api-key` > 环境变量 `OPENAI_API_KEY`。报错和输出里 key 会被涂成 `sk-***abcd`，不会明文出现。

## 输出示例

```
$ python -m apibench --compare m1,m2 --stream --n 3

| 模型 | 成功 | TTFT | 速度 | 间隔 p50 | 间隔 p95 | 总耗时 |
|---|---|---|---|---|---|---|
| m1 | 3/3 | 42 ms | 61.2 tok/s | 15 ms | 21 ms | 312 ms |
| m2 | 3/3 | 118 ms | 33.7 tok/s | 28 ms | 44 ms | 590 ms |
```

```
$ python -m apibench --soak 10

成功率   : 38/38 (100.0%)
延迟 p50 : 251 ms   p95 : 268 ms   max : 290 ms
延迟分布:
    248 ms | ############ 4
    252 ms | ######################################## 12
    ...
```

## 参数

| 参数 | 说明 |
|---|---|
| `--base-url` | API 地址（默认 `https://api.openai.com/v1`）|
| `--api-key` | API key（默认读 `OPENAI_API_KEY`）|
| `--model` | 模型名（默认 `gpt-4o-mini`）|
| `--n` | 每个模型跑几次取平均（默认 3）|
| `--max-tokens` | `max_tokens`（默认 200）|
| `--timeout` | 单次超时秒数（默认 60）|
| `--stream` | 流式模式：测 TTFT / tokens/s |
| `--compare m1,m2` | 多模型对比 |
| `--md FILE` | 对比表格存成 Markdown 文件 |
| `--models` | `GET /models` 列出可用模型 |
| `--soak N` | 持续压测 N 秒 |
| `--dry-run` | 模拟数据演示输出格式，无需 key |

## 要求

Python 3.10+，零第三方依赖。

## License

MIT
