"""Gemma (via Ollama's native tool calling) writes the prose; TabPFN and pandas supply the numbers.

The only way data reaches the model is `_tool_message`, which serialises a tool result through
`privacy.Guard.check`. Tools return aggregates; the guard refuses anything containing a title/URL.
"""

from __future__ import annotations

import bisect
import ipaddress
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from sloth import tools
from sloth.config import Config
from sloth.privacy import Guard
from sloth.timing import timed

MAX_ROUNDS = 6
SYSTEM = """You are Timesheet Sloth, a friendly, slightly lazy assistant that writes short work summaries \
from the user's own ActivityWatch data.
Rules:
- Get every number from a tool. Never estimate, add up, or invent numbers yourself.
- Copy numbers exactly as the tools give them (you may round to 1 decimal).
- You never see window titles or URLs; don't guess what specific work was done beyond categories and apps.
- focus_conditions suggestions are patterns, not guarantees: say "tends to" and give the session count.
- Be brief and plain. Markdown is fine."""


def is_local(url: str) -> bool:
    host = urlparse(url).hostname or ""
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def banner(cfg: Config, data_dir: object) -> str:
    where = "LOCAL" if is_local(cfg.ollama_url) else "REMOTE"
    return f"LLM: {where} ({cfg.llm_model}) · TabPFN: LOCAL · data dir: {data_dir}"


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    result: dict[str, Any]


@dataclass
class Result:
    text: str
    calls: list[ToolCall] = field(default_factory=list)
    llm_seconds: float = 0.0
    llm_calls: int = 0
    unmatched_numbers: list[str] = field(default_factory=list)


def _tool_message(guard: Guard, name: str, result: dict[str, Any]) -> dict[str, Any]:
    """The single gate between data and the prompt."""
    return {"role": "tool", "tool_name": name, "content": guard.check(result)}


def _chat(client: httpx.Client, cfg: Config, messages: list[dict[str, Any]], use_tools: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.llm_model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0.2, "num_ctx": cfg.llm_num_ctx},
        "keep_alive": cfg.llm_keep_alive,
    }
    if use_tools:
        body["tools"] = tools.schemas()
    with timed("llm.chat", model=cfg.llm_model, messages=len(messages)) as rec:
        r = client.post(f"{cfg.ollama_url}/api/chat", json=body, timeout=600)
        if r.status_code == 400 and "think" in r.text:  # model without a thinking switch
            body.pop("think")
            r = client.post(f"{cfg.ollama_url}/api/chat", json=body, timeout=600)
        r.raise_for_status()
        data = r.json()
        rec["prompt_tokens"] = data.get("prompt_eval_count")
        rec["output_tokens"] = data.get("eval_count")
        rec["load_seconds"] = round(data.get("load_duration", 0) / 1e9, 2)
    return data


def run(
    task: str,
    ctx: tools.Context,
    on_call: Callable[[ToolCall], None] | None = None,
    client: httpx.Client | None = None,
) -> Result:
    """Run the tool loop until Gemma answers in prose (or MAX_ROUNDS is hit)."""
    cfg = ctx.cfg
    guard = Guard(ctx.events)
    own = client is None
    client = client or httpx.Client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    res = Result(text="")
    try:
        for round_ in range(MAX_ROUNDS + 1):
            t0 = time.perf_counter()
            data = _chat(client, cfg, messages, use_tools=round_ < MAX_ROUNDS)
            res.llm_seconds += time.perf_counter() - t0
            res.llm_calls += 1
            msg = data.get("message", {})
            calls = msg.get("tool_calls") or []
            if not calls:
                res.text = (msg.get("content") or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.get("content", ""), "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {})
                name, args = fn.get("name", ""), fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                result = tools.call(ctx, name, args)
                call = ToolCall(name, args, result)
                res.calls.append(call)
                if on_call:
                    on_call(call)
                messages.append(_tool_message(guard, name, result))
    finally:
        if own:
            client.close()
    res.unmatched_numbers = check_numbers(res.text, [c.result for c in res.calls])
    return res


# --- number checker ----------------------------------------------------------------------------

_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")


def _walk(x: Any) -> list[Any]:
    if isinstance(x, dict):
        return [k for k in x] + [v for val in x.values() for v in _walk(val)]
    if isinstance(x, list):
        return [v for item in x for v in _walk(item)]
    return [x]


def _allowed(results: list[dict[str, Any]]) -> tuple[list[float], list[float], set[str]]:
    """Numbers in the tool results: plain values, values that may be written as percentages
    (fractions x 100, and numbers that already carry a % sign), and HH:MM times."""
    plain: list[float] = []
    percent: list[float] = []
    times: set[str] = set()
    for leaf in _walk(results):
        if isinstance(leaf, bool):
            continue
        if isinstance(leaf, int | float):
            v = float(leaf)
            plain.append(v)
            if 0 < abs(v) <= 1:
                percent.append(v * 100)
        elif isinstance(leaf, str):
            times.update(t.zfill(5) for t in _TIME.findall(leaf))
            for m in _NUM.findall(_TIME.sub(" ", _DATE.sub(" ", leaf))):
                (percent if m.endswith("%") else plain).append(float(m.rstrip("%")))
    return sorted(plain), sorted(percent), times


def _near(values: list[float], x: float, tol: float) -> bool:
    i = bisect.bisect_left(values, x - tol)
    return i < len(values) and values[i] <= x + tol


def check_numbers(prose: str, results: list[dict[str, Any]]) -> list[str]:
    """Numbers in the prose that aren't a tool number rounded to the prose's precision.

    "5.3" matches a tool value of 5.25 (|5.25 - 5.3| <= 0.05), whichever way the model rounds
    halves; "5.4" doesn't. A number written with % must match a fraction x 100 (0.41 -> "41%"), so
    "0.41%" for a 41% rate is flagged. Dates are ignored; HH:MM times must match exactly.
    """
    plain, percent, times = _allowed(results)
    unmatched = [t for t in _TIME.findall(prose) if t.zfill(5) not in times]
    text = _TIME.sub(" ", _DATE.sub(" ", prose))
    for m in _NUM.findall(text):
        raw = m.rstrip("%")
        x = float(raw)
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        tol = 0.5 * 10**-decimals + 1e-9
        if not _near(percent if m.endswith("%") else plain, x, tol):
            unmatched.append(m)
    return unmatched
