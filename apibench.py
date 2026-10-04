#!/usr/bin/env python3
"""apibench — 给任何 OpenAI-compatible API 做基准测试的小工具。

只依赖 Python 标准库。支持：
- 非流式：总延迟 + usage 里的 token 统计
- 流式 (--stream)：首 token 时间 (TTFT)、token 间隔 p50/p95、tokens/秒
- --compare m1,m2：多模型横向对比，输出 Markdown 表格
- --models：GET /models 列出可用模型
- --soak N：持续压测 N 秒，给出成功率、延迟分位数和 ASCII 直方图
- --dry-run：无 key 也能看完整输出格式（模拟数据）

用法示例：
    apibench --base-url https://api.openai.com/v1 --model gpt-4o-mini "讲个笑话"
    apibench --base-url http://localhost:8000/v1 --api-key test --model m1 --stream --n 3
    apibench --compare gpt-4o-mini,gpt-4o --stream --md result.md
    apibench --soak 60 --stream
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

VERSION = "0.1.0"
DEFAULT_PROMPT = "用一句话介绍你自己。"
DEFAULT_MAX_TOKENS = 200

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")


def redact(text: str) -> str:
    """把文本里的 key 涂掉：sk-***abcd。所有对外输出/报错都走这里。"""
    return _KEY_RE.sub(lambda m: "sk-***" + m.group(0)[-4:], text)


def die(msg: str, code: int = 1) -> "NoReturn":
    print(f"error: {redact(msg)}", file=sys.stderr)
    raise SystemExit(code)


def percentile(xs: list[float], q: float) -> float:
    """0<=q<=100，线性插值分位数。"""
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def ms(t: float) -> str:
    return f"{t * 1000:.0f} ms"


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------

class BenchError(Exception):
    pass


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"apibench/{VERSION}",
    }


def _post_json(base_url: str, api_key: str, payload: dict, timeout: float):
    """POST /chat/completions，返回 (status, body_bytes, headers)。"""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers=_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:500]
        if e.code == 401:
            raise BenchError("认证失败 (401)：API key 无效或已过期，请检查 --api-key")
        raise BenchError(f"HTTP {e.code}：{body or '无返回内容'}")
    except urllib.error.URLError as e:
        reason = getattr(e.reason, "strerror", None) or str(e.reason)
        raise BenchError(f"连接失败：{reason}（base-url 对吗？服务在跑吗？）")
    except TimeoutError:
        raise BenchError(f"请求超时（>{timeout:.0f}s），可加大 --timeout")
    except OSError as e:  # socket.timeout 等
        raise BenchError(f"请求超时或网络错误：{e}")


def _get_json(url: str, api_key: str, timeout: float):
    req = urllib.request.Request(url, headers=_headers(api_key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise BenchError("认证失败 (401)：API key 无效或已过期，请检查 --api-key")
        raise BenchError(f"HTTP {e.code}")
    except urllib.error.URLError as e:
        raise BenchError(f"连接失败：{e.reason}")
    except OSError as e:
        raise BenchError(f"请求超时或网络错误：{e}")


# ---------------------------------------------------------------------------
# 单次测量
# ---------------------------------------------------------------------------

def run_once(base_url, api_key, model, prompt, max_tokens, timeout, stream):
    """跑一次请求。返回 dict：ok / error / latency_s / ttft_s / tokens / tok_per_s /
    gap_p50_ms / gap_p95_ms / prompt_tokens / completion_tokens。"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    t0 = time.perf_counter()
    if not stream:
        try:
            status, body, _ = _post_json(base_url, api_key, payload, timeout)
        except BenchError as e:
            return {"ok": False, "error": str(e)}
        dt = time.perf_counter() - t0
        try:
            obj = json.loads(body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"ok": False, "error": "返回不是合法 JSON"}
        usage = obj.get("usage") or {}
        return {
            "ok": True,
            "latency_s": dt,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }

    # ---- 流式：手动解析 SSE ----
    payload["stream"] = True
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers=_headers(api_key),
        method="POST",
    )
    token_times: list[float] = []
    first_token_at: float | None = None
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"ok": False, "error": "认证失败 (401)：API key 无效或已过期，请检查 --api-key"}
        return {"ok": False, "error": f"HTTP {e.code}"}
    except (urllib.error.URLError, OSError) as e:
        return {"ok": False, "error": f"连接失败：{e}"}
    try:
        with resp:
            while True:
                if time.perf_counter() - t0 > timeout:
                    return {"ok": False, "error": f"流式读取超时（>{timeout:.0f}s）"}
                # readline 按行返回，服务端每 flush 一个 SSE 事件就唤醒一次，
                # token 时间戳才准；read(N) 会攒满 N 字节才返回，时间全挤在一起。
                raw = resp.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_s)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                delta = (choices[0].get("delta") if choices else {}) or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                    token_times.append(now)
    except OSError as e:
        return {"ok": False, "error": f"流式读取中断：{e}"}
    finally:
        try:
            resp.close()
        except Exception:
            pass

    dt = time.perf_counter() - t0
    n = len(token_times)
    if n == 0:
        return {"ok": False, "error": "流式响应里没有收到任何 token（空回复？）"}
    gaps = [(b - a) * 1000 for a, b in zip(token_times, token_times[1:])]
    span = token_times[-1] - token_times[0]
    return {
        "ok": True,
        "latency_s": dt,
        "ttft_s": (first_token_at - t0) if first_token_at else None,
        "tokens": n,
        "tok_per_s": n / span if span > 0 else float(n),
        "gap_p50_ms": percentile(gaps, 50) if gaps else 0.0,
        "gap_p95_ms": percentile(gaps, 95) if gaps else 0.0,
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_single(result: dict, model: str, stream: bool, n: int):
    if not result["ok"]:
        die(result["error"])
    if stream:
        print(f"模型: {model}（流式，{n} 次取平均）")
        print(f"  TTFT（首 token） : {ms(result['ttft_s'])}")
        print(f"  速度             : {result['tok_per_s']:.1f} tokens/s")
        print(f"  token 间隔 p50   : {result['gap_p50_ms']:.0f} ms")
        print(f"  token 间隔 p95   : {result['gap_p95_ms']:.0f} ms")
        print(f"  总耗时           : {ms(result['latency_s'])}（{result['tokens']} tokens）")
    else:
        print(f"模型: {model}（非流式，{n} 次取平均）")
        print(f"  总延迟           : {ms(result['latency_s'])}")
        for k, label in (("prompt_tokens", "prompt"), ("completion_tokens", "补全"),
                         ("total_tokens", "总计")):
            v = result.get(k)
            print(f"  token {label:<6}: {v if v is not None else '（服务端未返回 usage）'}")


def aggregate(results: list[dict]) -> dict:
    ok = [r for r in results if r["ok"]]
    errs = [r["error"] for r in results if not r["ok"]]
    agg: dict = {"runs": len(results), "ok": len(ok), "errors": errs}
    if not ok:
        return agg
    agg["latency_s"] = statistics.mean(r["latency_s"] for r in ok)
    if all("ttft_s" in r and r["ttft_s"] is not None for r in ok):
        agg["ttft_s"] = statistics.mean(r["ttft_s"] for r in ok)  # type: ignore[arg-type]
        agg["tok_per_s"] = statistics.mean(r["tok_per_s"] for r in ok)
        agg["gap_p50_ms"] = statistics.mean(r["gap_p50_ms"] for r in ok)
        agg["gap_p95_ms"] = statistics.mean(r["gap_p95_ms"] for r in ok)
        agg["tokens"] = statistics.mean(r["tokens"] for r in ok)
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        vals = [r[k] for r in ok if r.get(k) is not None]
        if vals:
            agg[k] = statistics.mean(vals)
    return agg


def compare_table(rows: list[tuple[str, dict]], stream: bool) -> str:
    """rows: [(model, agg)]。返回 Markdown 表格。"""
    if stream:
        head = "| 模型 | 成功 | TTFT | 速度 | 间隔 p50 | 间隔 p95 | 总耗时 |"
        sep = "|---|---|---|---|---|---|---|"
        lines = [head, sep]
        for model, a in rows:
            if a["ok"] == 0:
                lines.append(f"| {model} | 0/{a['runs']} ❌ | - | - | - | - | - |")
                continue
            lines.append(
                f"| {model} | {a['ok']}/{a['runs']} | {ms(a['ttft_s'])} | "
                f"{a['tok_per_s']:.1f} tok/s | {a['gap_p50_ms']:.0f} ms | "
                f"{a['gap_p95_ms']:.0f} ms | {ms(a['latency_s'])} |"
            )
    else:
        head = "| 模型 | 成功 | 总延迟 | prompt tok | 补全 tok |"
        sep = "|---|---|---|---|---|"
        lines = [head, sep]
        for model, a in rows:
            if a["ok"] == 0:
                lines.append(f"| {model} | 0/{a['runs']} ❌ | - | - | - |")
                continue
            pt = f"{a['prompt_tokens']:.0f}" if "prompt_tokens" in a else "-"
            ct = f"{a['completion_tokens']:.0f}" if "completion_tokens" in a else "-"
            lines.append(f"| {model} | {a['ok']}/{a['runs']} | {ms(a['latency_s'])} | {pt} | {ct} |")
    return "\n".join(lines)


def ascii_histogram(latencies: list[float], width: int = 40, bins: int = 12) -> str:
    if not latencies:
        return "（无数据）"
    lo, hi = min(latencies), max(latencies)
    if hi == lo:
        hi = lo + 1e-9
    counts = [0] * bins
    for x in latencies:
        i = min(int((x - lo) / (hi - lo) * bins), bins - 1)
        counts[i] += 1
    peak = max(counts)
    lines = []
    for i, c in enumerate(counts):
        left = lo + (hi - lo) * i / bins
        bar = "#" * max(1, int(c / peak * width)) if c else ""
        lines.append(f"  {left * 1000:7.0f} ms | {bar} {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="apibench",
        description="给 OpenAI-compatible API 做基准测试：延迟 / TTFT / tokens/s / 多模型对比 / 持续压测。",
    )
    p.add_argument("--version", action="version", version=f"apibench {VERSION}")
    p.add_argument("--base-url", default="https://api.openai.com/v1",
                   help="API 地址（默认 https://api.openai.com/v1）")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                   help="API key；不填则读环境变量 OPENAI_API_KEY")
    p.add_argument("--model", default="gpt-4o-mini", help="模型名（默认 gpt-4o-mini）")
    p.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="测试 prompt")
    p.add_argument("--n", type=int, default=3, help="每个模型跑几次取平均（默认 3）")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                   help="max_tokens（默认 200）")
    p.add_argument("--timeout", type=float, default=60, help="单次请求超时秒数（默认 60）")
    p.add_argument("--stream", action="store_true", help="流式模式：测 TTFT / tokens/s")
    p.add_argument("--compare", metavar="m1,m2",
                   help="多模型对比，逗号分隔，如 --compare gpt-4o-mini,gpt-4o")
    p.add_argument("--md", metavar="FILE", help="把对比表格存成 Markdown 文件")
    p.add_argument("--models", action="store_true", help="列出服务端可用模型（GET /models）")
    p.add_argument("--soak", type=float, metavar="SECONDS",
                   help="持续压测 N 秒：成功率、延迟分位数、ASCII 直方图")
    p.add_argument("--dry-run", action="store_true",
                   help="不发真实请求，用模拟数据演示完整输出格式")
    return p


def fake_single(stream: bool) -> dict:
    if stream:
        return {"ok": True, "latency_s": 2.31, "ttft_s": 0.42, "tokens": 87,
                "tok_per_s": 45.5, "gap_p50_ms": 19.0, "gap_p95_ms": 38.0}
    return {"ok": True, "latency_s": 1.85, "prompt_tokens": 12,
            "completion_tokens": 87, "total_tokens": 99}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.models:
        if not args.api_key:
            die("未提供 API key：用 --api-key 或环境变量 OPENAI_API_KEY")
        try:
            _, obj = _get_json(args.base_url.rstrip("/") + "/models", args.api_key, args.timeout)
        except BenchError as e:
            die(str(e))
        ids = [m.get("id", "?") for m in obj.get("data", [])]
        print(f"可用模型（{len(ids)} 个）：")
        for i in ids:
            print(f"  - {i}")
        return 0

    if args.dry_run:
        print("（dry-run：以下为模拟数据，未发真实请求）\n")
        models = args.compare.split(",") if args.compare else [args.model]
        if args.soak:
            print("持续压测: 模拟 60s")
            print("  成功率   : 60/60 (100.0%)")
            print("  延迟 p50 : 1830 ms   p95 : 2210 ms   max : 2480 ms")
            print("  延迟分布:")
            print(ascii_histogram([1.6 + (i % 12) * 0.07 for i in range(60)]))
            return 0
        rows = [(m.strip(), aggregate([fake_single(args.stream)])) for m in models]
        if len(rows) == 1:
            print_single(rows[0][1], rows[0][0], args.stream, 1)
        else:
            table = compare_table(rows, args.stream)
            print(table)
            if args.md:
                with open(args.md, "w", encoding="utf-8") as f:
                    f.write("# apibench 对比（dry-run 模拟数据）\n\n" + table + "\n")
                print(f"\n已保存到 {args.md}")
        return 0

    if not args.api_key:
        die("未提供 API key：用 --api-key 或环境变量 OPENAI_API_KEY")

    models = [m.strip() for m in args.compare.split(",")] if args.compare else [args.model]
    if args.n < 1:
        die("--n 至少为 1")

    # ---- 持续压测 ----
    if args.soak:
        print(f"持续压测 {args.soak:.0f}s（{'流式' if args.stream else '非流式'}，模型 {args.model}）…")
        latencies: list[float] = []
        ok_count = 0
        total = 0
        deadline = time.perf_counter() + args.soak
        while time.perf_counter() < deadline:
            total += 1
            r = run_once(args.base_url, args.api_key, args.model, args.prompt,
                         args.max_tokens, args.timeout, args.stream)
            if r["ok"]:
                ok_count += 1
                latencies.append(r["latency_s"])
            if total % 10 == 0:
                print(f"  …已跑 {total} 次", flush=True)
        print(f"\n成功率   : {ok_count}/{total} ({ok_count / total * 100:.1f}%)" if total else "\n一次都没跑完")
        if latencies:
            print(f"延迟 p50 : {ms(percentile(latencies, 50))}   "
                  f"p95 : {ms(percentile(latencies, 95))}   "
                  f"max : {ms(max(latencies))}")
            print("延迟分布:")
            print(ascii_histogram(latencies))
        return 0 if ok_count else 1

    # ---- 常规基准 / 对比 ----
    rows: list[tuple[str, dict]] = []
    for m in models:
        print(f"测试 {m}（{args.n} 次，{'流式' if args.stream else '非流式'}）…", flush=True)
        results = [
            run_once(args.base_url, args.api_key, m, args.prompt,
                     args.max_tokens, args.timeout, args.stream)
            for _ in range(args.n)
        ]
        rows.append((m, aggregate(results)))

    if len(rows) == 1:
        model, agg = rows[0]
        if agg["ok"] == 0:
            die(f"{model} 全部失败：{agg['errors'][0] if agg['errors'] else '未知错误'}")
        print_single(agg, model, args.stream, args.n)
    else:
        table = compare_table(rows, args.stream)
        print("\n" + table)
        for model, agg in rows:
            for e in agg["errors"][:2]:
                print(f"  ⚠ {model}: {redact(e)}", file=sys.stderr)
        if args.md:
            with open(args.md, "w", encoding="utf-8") as f:
                f.write(f"# apibench 对比\n\nprompt: {args.prompt}\n\n" + table + "\n")
            print(f"\n已保存到 {args.md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
