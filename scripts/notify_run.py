"""Push a one-line summary of the daily run to a phone or tablet.

Why this exists: the account went from 2026-03-24 to 2026-08-18 without opening
a position. The evidence was there every single day -- "Blocked: portfolio
exposure limit reached" in a CI log nobody opens. A run that silently does
nothing looks exactly like a run that had nothing to do, and only a push
notification distinguishes them at a glance.

Configure either channel via repository secrets / env vars:

  Telegram : TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  Pushover : PUSHOVER_TOKEN, PUSHOVER_USER

With neither set this is a silent no-op, so the workflow step is safe to leave
in place before the secrets exist. Never raises -- a notification failure must
not fail a trading run.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "docs" / "data.json"
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "").strip()


def _echo(text: str) -> None:
    """Print without dying on a Windows console.

    The default Windows code page is cp1252, which cannot encode the warning
    glyph in the title — and an unhandled UnicodeEncodeError here would fail the
    workflow step over a *log line*, after the notification had already been
    sent successfully.
    """
    try:
        print(text)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def _post(url: str, payload: Dict[str, str]) -> bool:
    try:
        data = urllib.parse.urlencode(payload).encode()
        req = urllib.request.Request(url, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:  # noqa: BLE001
        print(f"notify: send failed ({exc})", file=sys.stderr)
        return False


def build_message(d: Dict[str, Any]) -> tuple[str, str, int]:
    """Return (title, body, priority). Priority 1 means something needs a look."""
    run = d.get("latest_run") or {}
    acct = run.get("account") or {}
    summary = d.get("summary") or {}
    signals: List[Dict[str, Any]] = run.get("signals") or []

    passed = [s for s in signals if s.get("decision") == "PASS"]
    submitted = run.get("orders_submitted")
    exposure = acct.get("exposure")

    lines = [
        f"Equity ${acct.get('equity', 0):,.2f}  |  exposure {(exposure or 0) * 100:.1f}%",
        f"{len(signals)} signals, {len(passed)} passed, {submitted or 0} orders submitted",
    ]

    # Anything that means the bot could not act gets called out by name.
    problems: List[str] = []
    if exposure is not None and exposure >= 0.30:
        problems.append(f"exposure {exposure * 100:.1f}% is at/over the 30% cap — buys blocked")
    if passed and not submitted:
        names = ", ".join(s["symbol"] for s in passed)
        problems.append(f"{names} passed but nothing was submitted")
    if run.get("eos_available") is False:
        problems.append("eos algorithms did not load — reduced logic")
    if summary.get("failed"):
        problems.append(f"{summary['failed']} failed order(s) on record")

    if problems:
        lines.append("")
        lines += [f"! {p}" for p in problems]

    if passed:
        lines.append("")
        lines += [
            f"{s['symbol']} {s.get('signal', '')} conf {s.get('confidence', 0):.0%} "
            f"P(up) {s.get('prob_profit', 0):.0%}"
            for s in passed
        ]

    if DASHBOARD_URL:
        lines += ["", DASHBOARD_URL]

    title = f"Trading run {run.get('date') or ''}".strip()
    if problems:
        title = "⚠ " + title
    return title, "\n".join(lines), (1 if problems else 0)


def send(title: str, body: str, priority: int) -> bool:
    sent = False

    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if tg_token and tg_chat:
        sent |= _post(
            f"https://api.telegram.org/bot{tg_token}/sendMessage",
            {"chat_id": tg_chat, "text": f"{title}\n\n{body}", "disable_web_page_preview": "true"},
        )

    po_token = os.environ.get("PUSHOVER_TOKEN", "").strip()
    po_user = os.environ.get("PUSHOVER_USER", "").strip()
    if po_token and po_user:
        sent |= _post(
            "https://api.pushover.net/1/messages.json",
            {
                "token": po_token,
                "user": po_user,
                "title": title,
                "message": body,
                "priority": str(priority),
            },
        )

    return sent


def main() -> None:
    if not DATA.exists():
        _echo("notify: no docs/data.json — run scripts/build_dashboard.py first")
        return
    try:
        payload = json.loads(DATA.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _echo(f"notify: could not read data.json ({exc})")
        return

    title, body, priority = build_message(payload)
    _echo(f"{title}\n{body}")

    if not send(title, body, priority):
        _echo("notify: no channel configured (or all sends failed) — nothing pushed")


if __name__ == "__main__":
    main()
