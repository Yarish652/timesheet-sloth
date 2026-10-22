#!/usr/bin/env python3
"""
sloth.export - Step 1 of Timesheet Sloth.

Pulls your ActivityWatch history (AFK time removed) from the local server and
writes two CSVs. Nothing leaves your machine.

  events.csv  one row per window event: start, end, duration_s, app, title, url, category
  blocks.csv  one row per time block (default 30 min) with features for TabPFN

Standard library only. Usage:
  python -m sloth.export                  # last 120 days, 30-min blocks, into $SLOTH_DATA_DIR
  python -m sloth.export --days 60 --block 60
  python -m sloth.export --url http://localhost:5600 --categories aw-category-export.json

Category rules come from the server settings ('classes') when the server has them. ActivityWatch
v0.13 keeps them in the web UI instead; export them there (Settings > Categorization > Export)
and pass the JSON with --categories (default: $SLOTH_DATA_DIR/categories.json if present).
"""

import argparse
import csv
import json
import os
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from pathlib import Path

UNCAT = "Uncategorized"
EVENT_FIELDS = ["start", "end", "duration_s", "app", "title", "url", "category"]
BLOCK_FIELDS = [
    "block_start", "date", "weekday", "hour", "minute", "active_min", "switches",
    "distinct_apps", "top_app", "top_app_share", "second_app",
    "top_category", "uncategorized_share",
]  # fmt: skip


# ---------- ActivityWatch REST helpers ----------


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())


def post(base, path, body):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def pick_bucket(buckets, btype, hostname=None):
    cands = [b for b in buckets.values() if b.get("type") == btype]
    if hostname:
        cands = [b for b in cands if b.get("hostname") == hostname] or cands
    # prefer the bucket that was updated most recently
    cands.sort(key=lambda b: b.get("last_updated") or b.get("created") or "", reverse=True)
    return cands[0]["id"] if cands else None


def fetch_active_events(base, win_b, afk_b, web_b, start, end):
    """Window events intersected with not-afk periods, fetched one day at a time."""
    q = [
        f'afk = query_bucket("{afk_b}");',
        'afk = filter_keyvals(afk, "status", ["not-afk"]);',
        f'win = query_bucket("{win_b}");',
        "win = filter_period_intersect(win, afk);",
    ]
    if web_b:
        q += [
            f'web = query_bucket("{web_b}");',
            "web = filter_period_intersect(web, afk);",
            'RETURN = {"win": win, "web": web};',
        ]
    else:
        q.append('RETURN = {"win": win, "web": []};')

    win_all, web_all = [], []
    day = start
    while day < end:
        nxt = min(day + timedelta(days=1), end)
        period = f"{day.isoformat()}/{nxt.isoformat()}"
        res = post(base, "/api/0/query/", {"timeperiods": [period], "query": q})[0]
        win_all += res.get("win", [])
        web_all += res.get("web", [])
        day = nxt
        sys.stderr.write(f"\r  fetched up to {nxt.date()}   ")
    sys.stderr.write("\n")
    return win_all, web_all


# ---------- categories (your existing AW rules = free labels) ----------


def _server_classes(base):
    """ActivityWatch keeps category rules in server settings ('classes') on newer versions."""
    try:
        settings = get(base, "/api/0/settings/")
        classes = settings.get("classes") or []
    except Exception:
        try:
            classes = get(base, "/api/0/settings/classes") or []
        except Exception:
            classes = []
    return classes


def _file_classes(path):
    """A web UI category export: {"id": ..., "categories": [...]} or a bare list."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("categories", []) if isinstance(data, dict) else data


def compile_rules(classes):
    rules = []
    for c in classes:
        rule = c.get("rule") or {}
        if rule.get("type") != "regex" or not rule.get("regex"):
            continue
        flags = re.IGNORECASE if rule.get("ignore_case") else 0
        try:
            rx = re.compile(rule["regex"], flags)
        except re.error:
            continue
        name = c.get("name")
        name = " > ".join(name) if isinstance(name, list) else str(name)
        rules.append((name, name.count(">"), rx))
    return rules


def load_rules(base, categories_path=None):
    """Rules from the server, falling back to a web UI export file. Returns (rules, source)."""
    classes = _server_classes(base)
    if classes:
        return compile_rules(classes), "server settings"
    if categories_path and Path(categories_path).exists():
        return compile_rules(_file_classes(categories_path)), str(categories_path)
    return [], "none found"


def categorize(rules, *texts):
    hay = " ".join(t for t in texts if t)
    best = None
    for name, depth, rx in rules:
        if rx.search(hay) and (best is None or depth > best[1]):
            best = (name, depth)
    return best[0] if best else UNCAT


# ---------- writers (shared with sloth.synthetic) ----------


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def write_events(rows, path, extra_fields=()):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*EVENT_FIELDS, *extra_fields])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "start": r["start"].isoformat(), "end": r["end"].isoformat()})


def aggregate_blocks(rows, block_minutes):
    bsec = block_minutes * 60

    def block_key(t):
        midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
        offset = int((t - midnight).total_seconds()) // bsec * bsec
        return midnight + timedelta(seconds=offset)

    blocks = defaultdict(lambda: {"secs": 0.0, "apps": Counter(), "cats": Counter(), "switches": 0})
    prev_app = None
    for r in rows:
        if r["app"] != prev_app:
            blocks[block_key(r["start"])]["switches"] += 1
            prev_app = r["app"]
        # split events that cross block boundaries
        t, remaining = r["start"], r["duration_s"]
        while remaining > 0.01:
            k = block_key(t)
            chunk = min(remaining, (k + timedelta(seconds=bsec) - t).total_seconds())
            b = blocks[k]
            b["secs"] += chunk
            b["apps"][r["app"]] += chunk
            b["cats"][r["category"]] += chunk
            t += timedelta(seconds=chunk)
            remaining -= chunk
    return blocks


def write_blocks(blocks, path):
    """Write blocks with at least a minute of activity; returns how many were written."""
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=BLOCK_FIELDS)
        w.writeheader()
        for k in sorted(blocks):
            b = blocks[k]
            if b["secs"] < 60:
                continue
            top = b["apps"].most_common(2)
            cat = b["cats"].most_common(1)[0][0]
            w.writerow(
                {
                    "block_start": k.isoformat(),
                    "date": k.date().isoformat(),
                    "weekday": k.strftime("%a"),
                    "hour": k.hour,
                    "minute": k.minute,
                    "active_min": round(b["secs"] / 60, 1),
                    "switches": b["switches"],
                    "distinct_apps": len(b["apps"]),
                    "top_app": top[0][0],
                    "top_app_share": round(top[0][1] / b["secs"], 2),
                    "second_app": top[1][0] if len(top) > 1 else "",
                    "top_category": cat,
                    "uncategorized_share": round(b["cats"][UNCAT] / b["secs"], 2),
                }
            )
            n += 1
    return n


def print_summary(rows, n_blocks, block_minutes, out):
    """Privacy-safe summary (no titles, no URLs)."""
    total_h = sum(r["duration_s"] for r in rows) / 3600
    days = len({r["start"].date() for r in rows})
    by_cat = Counter()
    for r in rows:
        by_cat[r["category"]] += r["duration_s"] / 3600
    print("\n===== Summary (safe to share) =====")
    print(f"active days: {days}   active hours: {total_h:.0f}   events: {len(rows)}")
    print(f"{block_minutes}-min blocks: {n_blocks}  (TabPFN on CPU is comfortable up to ~5,000 rows)")
    print("hours by category:")
    for c, h in by_cat.most_common(12):
        print(f"  {c:<40} {h:7.1f}h  ({h / max(total_h, 1e-9):5.1%})")
    print(f"\nwrote {out}/events.csv and {out}/blocks.csv")


# ---------- main ----------


def main(argv=None):
    from sloth.config import load_dotenv

    load_dotenv()
    default_out = os.environ.get("SLOTH_DATA_DIR") or "data"
    ap = argparse.ArgumentParser(prog="python -m sloth.export")
    ap.add_argument("--url", default="http://localhost:5600")
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--block", type=int, default=30, help="block size in minutes")
    ap.add_argument("--hostname", default=None)
    ap.add_argument("--out", default=default_out, help="output dir (default: $SLOTH_DATA_DIR or ./data)")
    ap.add_argument(
        "--categories",
        default=None,
        help="web UI category export JSON, used when the server has no rules "
        "(default: <out>/categories.json)",
    )
    a = ap.parse_args(argv)
    base = a.url.rstrip("/")
    Path(a.out).mkdir(parents=True, exist_ok=True)

    try:
        buckets = get(base, "/api/0/buckets/")
    except Exception as e:
        sys.exit(f"Can't reach ActivityWatch at {base} ({e}). Is aw-server running?")

    win_b = pick_bucket(buckets, "currentwindow", a.hostname)
    afk_b = pick_bucket(buckets, "afkstatus", a.hostname)
    web_b = pick_bucket(buckets, "web.tab.current")
    if not (win_b and afk_b):
        sys.exit(f"Missing window or AFK bucket. Found: {list(buckets)}")
    print(f"window bucket: {win_b}\nafk bucket:    {afk_b}\nweb bucket:    {web_b or '(none)'}")

    rules, source = load_rules(base, a.categories or Path(a.out) / "categories.json")
    print(f"category rules found: {len(rules)} ({source})")
    if not rules:
        print(
            "  warning: no category rules, so everything will be Uncategorized. Export them from\n"
            "  the web UI (Settings > Categorization > Export) and pass --categories <file>."
        )

    local_tz = datetime.now().astimezone().tzinfo
    end = datetime.now(UTC)
    start = end - timedelta(days=a.days)
    win, web = fetch_active_events(base, win_b, afk_b, web_b, start, end)

    # index browser events so window events from browsers can borrow the URL
    web_spans = sorted(
        (
            parse_ts(e["timestamp"]),
            parse_ts(e["timestamp"]) + timedelta(seconds=e["duration"]),
            e["data"].get("url", ""),
        )
        for e in web
        if e.get("duration", 0) > 0
    )

    def url_at(t):
        # binary search over browser-tab spans
        lo, hi = 0, len(web_spans)
        while lo < hi:
            mid = (lo + hi) // 2
            if web_spans[mid][0] <= t:
                lo = mid + 1
            else:
                hi = mid
        if lo and web_spans[lo - 1][0] <= t < web_spans[lo - 1][1]:
            return web_spans[lo - 1][2]
        return ""

    rows = []
    for e in win:
        d = e.get("duration", 0)
        if d < 1:
            continue
        t0 = parse_ts(e["timestamp"]).astimezone(local_tz)
        app = e["data"].get("app", "")
        title = e["data"].get("title", "")
        url = url_at(t0.astimezone(UTC)) if web_spans else ""
        rows.append(
            {
                "start": t0,
                "end": t0 + timedelta(seconds=d),
                "duration_s": round(d, 1),
                "app": app,
                "title": title,
                "url": url,
                "category": categorize(rules, app, title, url),
            }
        )
    rows.sort(key=lambda r: r["start"])

    write_events(rows, f"{a.out}/events.csv")
    n_blocks = write_blocks(aggregate_blocks(rows, a.block), f"{a.out}/blocks.csv")
    print_summary(rows, n_blocks, a.block, a.out)


if __name__ == "__main__":
    main()
