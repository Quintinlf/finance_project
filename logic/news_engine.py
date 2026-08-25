"""Information diffusion: news events -> affected assets -> shadow signal.

The concrete case this exists for: a headline says a war is disrupting grain
exports. `logic/universe.py` already tags WEAT/CORN/SOYB/DBA with the
"geopolitical_supply_shock" theme specifically so a module like this one could
answer "which of our assets does this touch" -- but until now nothing read
those tags. This module is that reader.

Pipeline, in order:

1. **Ingest** -- Alpaca's News API, bundled with the same credentials already
   in .env. No new signup, no new secret. Confirmed live on 2026-08-25: a
   real headline, "Russia Eyes Grain Duty Pause Through Year End on Ukraine
   Strikes," tagged WEAT.

2. **Classify** -- a keyword lexicon maps headline + summary text to zero or
   more themes from `universe.THEMES`, and to a polarity (bullish / bearish /
   neutral) for the assets it touches. This is NOT NLP: it is word-matching
   against a small, inspectable list, and it says so everywhere it is used.
   Negation, sarcasm, and headline-vs-reality gaps are not handled.

3. **Diffuse** -- for every theme a headline matches, every asset carrying
   that theme (`universe.assets_for_theme`) gets an event, not just the
   symbols Alpaca explicitly tagged. A headline about Ukraine grain exports
   that only tags WEAT still reaches CORN, SOYB, and DBA, which share the real
   exposure. Explicit Alpaca tags and theme-expanded reach are recorded
   separately (`match_type`) because they carry different reliability -- an
   explicit tag is a direct claim about that symbol; an expanded one is an
   inference through a shared theme.

4. **Persist + score** -- every event is written to the `news_events` table
   and graded against next-day return using the exact same (already
   split-corrected) return calculation the model components are graded with.
   No exceptions for a signal just because it is new.

5. **Shadow only** -- `apply_news_context` annotates `signal.meta` with the
   matched events, themes, and a polarity score, but never touches
   `signal_type`, `confidence`, or `prob_profit`. Every other addition to this
   pipeline (the eos algorithms, the edge gate) shipped shadow-first because
   nothing had evidence yet; a keyword lexicon has even less claim to being
   trusted out of the gate.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Union

from logic.sqlite_store import DEFAULT_DB_PATH, connect
from logic.universe import THEMES, assets_for_theme

# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------
# Deliberately small and readable rather than exhaustive. Every phrase here
# should be defensible on its own if someone asks "why did this headline match
# geopolitical_supply_shock." Extend by adding phrases, not by growing logic.

THEME_KEYWORDS: Dict[str, Sequence[str]] = {
    "geopolitical_supply_shock": (
        "war", "invasion", "invades", "strikes", "airstrike", "conflict",
        "sanctions", "sanction", "blockade", "export ban", "export duty",
        "export tax", "trade war", "tariff", "seizes", "attacks", "attack on",
        "military", "ceasefire", "truce",
    ),
    "weather_drought": (
        "drought", "frost", "freeze warning", "flood", "flooding", "heatwave",
        "heat wave", "crop damage", "crop failure", "dry conditions", "wildfire",
    ),
    "opec": (
        "opec", "opec+", "production cut", "output cut", "production quota",
        "supply cut",
    ),
    "crude": ("oil price", "crude oil", "barrel", "refinery"),
    "food_security": ("food security", "grain export", "wheat export", "food crisis"),
    "inflation_hedge": ("inflation", "cpi report", "rate hike", "rate cut", "fed decision"),
    "dollar_weakness": ("dollar weakens", "dollar falls", "dxy", "greenback"),
    "china_demand": ("china demand", "beijing", "pboc", "chinese imports"),
    "industrial_demand": ("industrial demand", "manufacturing pmi", "factory output"),
}

# Words pushing the read toward "this asset goes up" vs "down." Independent of
# theme: a headline can match a theme and still need a direction.
BULLISH_WORDS = (
    "surge", "surges", "soar", "soars", "jump", "jumps", "rally", "rallies",
    "spike", "spikes", "shortage", "halts", "halt exports", "disrupt",
    "disrupts", "cut supply", "supply cut", "ban exports", "export ban",
    "record high", "tightens", "tightening",
)
BEARISH_WORDS = (
    "plunge", "plunges", "tumble", "tumbles", "slump", "slumps", "glut",
    "oversupply", "bumper crop", "record harvest", "resumes exports",
    "resume exports", "ease sanctions", "eases sanctions", "ceasefire",
    "truce", "increase production", "raises output", "output hike",
    "falls", "eases",
)

# A theme key here that doesn't exist in universe.THEMES silently matches
# nothing downstream (assets_for_theme would raise, but only when called) --
# fail at import time instead, the same guarantee Asset.__post_init__ makes
# for asset tagging.
_unknown_theme_keys = set(THEME_KEYWORDS) - set(THEMES)
if _unknown_theme_keys:
    raise ValueError(
        f"THEME_KEYWORDS has unknown theme(s) {_unknown_theme_keys}; "
        f"add them to universe.THEMES first"
    )

NEWS_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS news_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_key TEXT UNIQUE,
    published_at DATETIME,
    headline TEXT,
    source TEXT,
    symbol TEXT,
    match_type TEXT,           -- 'explicit' (Alpaca-tagged) | 'theme' (diffused)
    matched_themes TEXT,       -- comma-joined
    polarity TEXT,             -- 'bullish' | 'bearish' | 'neutral'
    polarity_score REAL,       -- signed strength, roughly [-1, 1]
    next_day_return REAL,
    correct INTEGER
)
"""


def init_news_engine(db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH) -> None:  # type: ignore[name-defined]
    with connect(db_path) as conn:
        conn.execute(NEWS_TABLE_DDL.strip())
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_events_symbol_time ON news_events(symbol, published_at)"
        )


@dataclass
class NewsEvent:
    headline: str
    source: str
    published_at: datetime
    symbol: str
    match_type: str  # 'explicit' | 'theme'
    matched_themes: List[str]
    polarity: str  # 'bullish' | 'bearish' | 'neutral'
    polarity_score: float
    event_key: str = field(default="")

    def __post_init__(self) -> None:
        if not self.event_key:
            ts = self.published_at.isoformat()
            self.event_key = f"{ts}|{self.symbol}|{self.headline[:80]}"


def classify_headline(text: str) -> Dict[str, Any]:
    """Match themes and polarity in a headline (+ optional summary).

    Pure keyword matching against `THEME_KEYWORDS` / `BULLISH_WORDS` /
    `BEARISH_WORDS`, case-insensitive, substring match. Returns which themes
    matched, which literal phrases triggered them, and a polarity verdict.
    """
    lowered = text.lower()

    matched_themes: List[str] = []
    matched_phrases: List[str] = []
    for theme, phrases in THEME_KEYWORDS.items():
        for phrase in phrases:
            if phrase in lowered:
                matched_themes.append(theme)
                matched_phrases.append(phrase)
                break

    bull_hits = [w for w in BULLISH_WORDS if w in lowered]
    bear_hits = [w for w in BEARISH_WORDS if w in lowered]

    score = (len(bull_hits) - len(bear_hits))
    if score > 0:
        polarity = "bullish"
    elif score < 0:
        polarity = "bearish"
    else:
        polarity = "neutral"

    # Normalize to roughly [-1, 1] without pretending false precision beyond
    # "more one-sided phrases matched than the other."
    total_hits = len(bull_hits) + len(bear_hits)
    polarity_score = (score / total_hits) if total_hits else 0.0

    return {
        "themes": matched_themes,
        "matched_phrases": matched_phrases,
        "polarity": polarity,
        "polarity_score": round(polarity_score, 3),
        "bullish_hits": bull_hits,
        "bearish_hits": bear_hits,
    }


def _fetch_raw_news(symbols: Sequence[str], *, lookback_hours: int = 48, limit: int = 50) -> List[Any]:
    """Pull recent news from Alpaca for the given symbols. Empty list on any failure."""
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        from logic.alpaca_exercises import load_alpaca_creds

        creds = load_alpaca_creds()
        client = NewsClient(creds.api_key, creds.secret_key)
        req = NewsRequest(
            symbols=",".join(sorted(set(symbols))),
            start=datetime.now(timezone.utc) - timedelta(hours=lookback_hours),
            limit=limit,
        )
        resp = client.get_news(req)
        data = getattr(resp, "data", resp)
        return list(data.get("news", []) if isinstance(data, dict) else data)
    except Exception as exc:  # noqa: BLE001
        logging.warning("News fetch failed (%s). Continuing without news context.", exc)
        return []


def build_news_events(
    universe_symbols: Sequence[str],
    *,
    lookback_hours: int = 48,
    limit: int = 50,
) -> List[NewsEvent]:
    """Fetch recent news and diffuse it across the tradeable universe.

    One NewsEvent per (headline, affected symbol). A headline explicitly
    tagging WEAT that also matches "geopolitical_supply_shock" produces
    further events for every other universe symbol carrying that theme
    (CORN, SOYB, DBA, ...) with match_type='theme' -- weaker than an explicit
    tag, and always labeled as such.
    """
    universe_set = {s.strip().upper() for s in universe_symbols}
    raw = _fetch_raw_news(universe_set, lookback_hours=lookback_hours, limit=limit)

    events: List[NewsEvent] = []
    seen: set = set()

    for item in raw:
        headline = str(getattr(item, "headline", "") or "")
        summary = str(getattr(item, "summary", "") or "")
        source = str(getattr(item, "source", "") or "")
        published_at = getattr(item, "created_at", None) or datetime.now(timezone.utc)
        if not isinstance(published_at, datetime):
            published_at = datetime.now(timezone.utc)
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=timezone.utc)

        classification = classify_headline(f"{headline} {summary}")
        themes = classification["themes"]
        polarity = classification["polarity"]
        polarity_score = classification["polarity_score"]

        explicit_symbols = {
            str(s).strip().upper() for s in (getattr(item, "symbols", None) or [])
        } & universe_set

        for symbol in explicit_symbols:
            key = (headline, symbol)
            if key in seen:
                continue
            seen.add(key)
            events.append(
                NewsEvent(
                    headline=headline, source=source, published_at=published_at,
                    symbol=symbol, match_type="explicit", matched_themes=themes,
                    polarity=polarity, polarity_score=polarity_score,
                )
            )

        for theme in themes:
            for asset in assets_for_theme(theme):
                if asset.symbol not in universe_set:
                    continue
                key = (headline, asset.symbol)
                if key in seen:
                    continue
                seen.add(key)
                events.append(
                    NewsEvent(
                        headline=headline, source=source, published_at=published_at,
                        symbol=asset.symbol, match_type="theme", matched_themes=themes,
                        polarity=polarity, polarity_score=polarity_score,
                    )
                )

    return events


def persist_news_events(
    events: Sequence[NewsEvent], *, db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH  # type: ignore[name-defined]
) -> int:
    """Idempotent insert (event_key UNIQUE + INSERT OR IGNORE). Returns rows written."""
    if not events:
        return 0
    init_news_engine(db_path)
    with connect(db_path) as conn:
        before = conn.total_changes
        for e in events:
            conn.execute(
                """
                INSERT OR IGNORE INTO news_events
                    (event_key, published_at, headline, source, symbol, match_type,
                     matched_themes, polarity, polarity_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """.strip(),
                (
                    e.event_key, e.published_at.isoformat(), e.headline, e.source,
                    e.symbol, e.match_type, ",".join(e.matched_themes), e.polarity,
                    e.polarity_score,
                ),
            )
        return conn.total_changes - before


def apply_news_context(signals: Sequence[Any], events: Sequence[NewsEvent]) -> Sequence[Any]:
    """Shadow-annotate signals with matching news events. Never changes a trade.

    Stores `meta['news_events']`, `meta['news_themes']`, and
    `meta['news_polarity_score']` (mean polarity across matching events) on any
    signal whose symbol has a matching event. Nothing here touches
    `signal_type`, `confidence`, or `prob_profit` -- this is context for a human
    reading the log, and future training data, not an input to today's trade.
    """
    by_symbol: Dict[str, List[NewsEvent]] = {}
    for e in events:
        by_symbol.setdefault(e.symbol, []).append(e)

    for signal in signals:
        symbol = str(getattr(signal, "symbol", "") or "")
        matches = by_symbol.get(symbol)
        if not hasattr(signal, "meta") or signal.meta is None:
            signal.meta = {}
        if not matches:
            continue
        signal.meta["news_events"] = [
            {"headline": m.headline, "match_type": m.match_type, "polarity": m.polarity}
            for m in matches
        ]
        signal.meta["news_themes"] = sorted({t for m in matches for t in m.matched_themes})
        signal.meta["news_polarity_score"] = round(
            sum(m.polarity_score for m in matches) / len(matches), 3
        )

    return signals


def log_news_events(events: Sequence[NewsEvent]) -> None:
    if not events:
        logging.info("NEWS DIFFUSION: no matching news in the lookback window")
        return
    by_symbol: Dict[str, int] = {}
    for e in events:
        by_symbol[e.symbol] = by_symbol.get(e.symbol, 0) + 1
    logging.info(
        "NEWS DIFFUSION: %s event(s) across %s symbol(s)",
        len({e.headline for e in events}), len(by_symbol),
    )
    for e in events:
        logging.info(
            "NEWS EVENT | symbol=%s | match=%s | polarity=%s (%.2f) | themes=%s | %s",
            e.symbol, e.match_type, e.polarity, e.polarity_score,
            ",".join(e.matched_themes) or "-", e.headline[:120],
        )


# ---------------------------------------------------------------------------
# Scoring -- graded exactly like every other component, no exceptions.
# ---------------------------------------------------------------------------

def score_news_events(
    *, db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH, max_rows: int = 250  # type: ignore[name-defined]
) -> int:
    """Backfill next_day_return + correctness for matured news events.

    Reuses the exact (already split-corrected) return calculation used for the
    model components, so a corporate-action row can't corrupt this table the
    way it once corrupted that one.
    """
    from logic.model_performance_tracker import _compute_realized_return_from_row

    init_news_engine(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, symbol, published_at, polarity
            FROM news_events
            WHERE next_day_return IS NULL
            ORDER BY published_at ASC, id ASC
            LIMIT ?
            """.strip(),
            (int(max_rows),),
        ).fetchall()

    now_utc = datetime.now(timezone.utc)
    updates = []
    for row in rows:
        ts = datetime.fromisoformat(str(row["published_at"]).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if now_utc - ts < timedelta(hours=24):
            continue

        # _compute_realized_return_from_row derives both prices from its own
        # fetched series (see its docstring on the corporate-action bug this
        # fixed) -- price_at_signal is accepted but unused, so there is no
        # separate anchor price to capture here.
        ret = _compute_realized_return_from_row(
            symbol=str(row["symbol"]), timestamp_iso=str(row["published_at"]),
            price_at_signal=0.0,
        )
        if ret is None:
            continue

        polarity = str(row["polarity"])
        if polarity == "neutral":
            correct = None
        else:
            went_up = ret > 0
            correct = 1 if ((polarity == "bullish") == went_up) else 0

        updates.append((float(ret), correct, int(row["id"])))

    if not updates:
        return 0

    with connect(db_path) as conn:
        conn.executemany(
            "UPDATE news_events SET next_day_return = ?, correct = ? WHERE id = ?",
            updates,
        )
    return len(updates)


def summarize_news_accuracy(
    *, db_path: Union[str, "PathLike[str]"] = DEFAULT_DB_PATH, min_sample: int = 10  # type: ignore[name-defined]
) -> str:
    """Same honesty framing as the model-component report: accuracy vs base rate.

    Split by match_type, since an explicit tag and a theme-diffused inference
    have no reason to carry the same reliability.
    """
    import math

    lines: List[str] = []
    with connect(db_path) as conn:
        base = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN next_day_return > 0 THEN 1 ELSE 0 END) "
            "FROM news_events WHERE next_day_return IS NOT NULL"
        ).fetchone()
        base_n = int(base[0] or 0)
        if not base_n:
            return ""
        base_up = int(base[1] or 0) / base_n
        lines.append(
            f"  base rate: the market rose on {base_up * 100:.1f}% of {base_n} "
            f"scored news-linked observations"
        )
        lines += [f"  {'match_type':<10}{'n':>5}{'acc':>8}{'95% CI':>18}  verdict", "  " + "-" * 58]

        for match_type in ("explicit", "theme"):
            row = conn.execute(
                "SELECT COUNT(*), SUM(correct) FROM news_events "
                "WHERE correct IS NOT NULL AND match_type = ?",
                (match_type,),
            ).fetchone()
            n = int(row[0] or 0)
            if not n:
                lines.append(f"  {match_type:<10}{0:>5}{'--':>8}{'':>18}  not yet scored")
                continue
            p = int(row[1] or 0) / n
            se = math.sqrt(max(p * (1 - p), 0.0) / n)
            lo, hi = max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)
            verdict = (
                f"sample too small (n<{min_sample})" if n < min_sample
                else "better than chance" if lo > 0.5
                else "WORSE than chance" if hi < 0.5
                else "indistinguishable from chance"
            )
            lines.append(f"  {match_type:<10}{n:>5}{p * 100:>7.1f}%   [{lo*100:5.1f}%, {hi*100:5.1f}%]  {verdict}")

    return "\n".join(lines)
