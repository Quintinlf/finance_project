"""Thin HTTP client for the ThetaData v3 local terminal.

Gate 0 asks whether our volatility forecast carries information implied vol does
not. Answering it by accumulating daily snapshots reaches a conclusive 30-day
read in **April 2029**, because overlapping windows mean 126 trading days buy
only a handful of independent observations. Buying history collapses that to an
afternoon, and ThetaData sells it month-to-month, so the whole purchase is one
month's subscription and then cancel.

### The thing that surprises people

There is no hosted REST API. ThetaData ships a **Java terminal that you run
locally**, and it serves HTTP on 127.0.0.1:25503. So "subscribe and download"
is really: install Java 21+, get an API key from the portal, run the jar, then
talk to localhost. Everything in this module assumes that process is up, and
``probe()`` exists to say so in one call rather than letting every endpoint fail
with a connection error.

### v2 vs v3, which is a real trap

The older v2 terminal listened on **25510** and used ``root``/``exp`` with
strikes in tenths of a cent (``170000`` meant $170.00). v3 listens on **25503**,
renames those to ``symbol``/``expiration``, and takes strikes in **dollars**
(``170.000``). A v3 terminal answers v2 paths with HTTP 410. Getting this wrong
produces either a hard error or -- far worse -- a silently empty result set that
looks like "this ETF has no options". ``API_VERSION`` is pinned here and the
parameter names live in one place for exactly that reason.

Nothing in this module writes to the database or makes a trading decision. It
returns parsed rows.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Union

API_VERSION = "v3"

# The v3 terminal's default local HTTP port. Overridable because the terminal
# writes host/port into a config file next to the jar and the user may move it.
DEFAULT_BASE_URL = "http://127.0.0.1:25503"

# Retail tiers allow 2 (Value) to 8 (Pro) concurrent requests. This client is
# deliberately serial: the ingest is a one-time pull, wall-clock time is not
# precious, and a 429 storm mid-pull would leave a partial sample whose gaps
# correlate with nothing in particular but still have to be explained later.
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 2.0

# Tier -> first date of option history, from the published access table. Used
# only to explain an empty result, never to decide what to request: the probe
# measures the real boundary per symbol instead of trusting this.
PUBLISHED_HISTORY_START = {
    "free": "2023-06-01",
    "value": "2020-01-01",
    "standard": "2016-01-01",
    "pro": "2012-06-01",
}


class ThetaDataUnavailable(RuntimeError):
    """The terminal is not reachable, or answered in a way we cannot use."""


@dataclass
class ProbeResult:
    """Whether the terminal is up, and what it said if not."""

    reachable: bool
    base_url: str
    detail: str = ""
    version_mismatch: bool = False

    def explain(self) -> str:
        if self.reachable:
            return f"ThetaData terminal reachable at {self.base_url}"
        lines = [f"ThetaData terminal NOT reachable at {self.base_url}", f"  {self.detail}"]
        if self.version_mismatch:
            lines.append(
                "  The terminal answered but rejected a v3 path. That is the signature of a\n"
                "  v2 terminal (which listens on 25510). Download ThetaTerminalv3.jar."
            )
        else:
            lines.append(
                "  There is no hosted API -- the terminal runs locally. To start it:\n"
                "    1. Install Java 21+            (java -version)\n"
                "    2. Get an API key              https://www.thetadata.net/portal/api_key\n"
                "    3. java -jar ThetaTerminalv3.jar --api-key YOUR_KEY\n"
                "  Then leave it running and re-run this command."
            )
        return "\n".join(lines)


@dataclass
class ThetaDataClient:
    """Serial HTTP client against a locally running v3 terminal."""

    base_url: str = field(default_factory=lambda: os.environ.get(
        "THETADATA_BASE_URL", DEFAULT_BASE_URL).rstrip("/"))
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    retries: int = DEFAULT_RETRIES
    backoff: float = DEFAULT_BACKOFF_SECONDS

    # ---------------------------------------------------------------- transport

    def _url(self, path: str, params: Dict[str, Any]) -> str:
        clean = {k: v for k, v in params.items() if v is not None}
        return f"{self.base_url}/{API_VERSION}/{path.lstrip('/')}?" + urllib.parse.urlencode(clean)

    def _get_json(self, path: str, **params: Any) -> List[Dict[str, Any]]:
        """GET a v3 endpoint as JSON. Returns a list of row dicts.

        Retries only on transport errors and 429/5xx. A 4xx other than 429 is a
        request we got wrong, and retrying it three times just delays the error
        message that would tell us so.
        """
        params.setdefault("format", "json")
        url = self._url(path, params)
        last: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                return _parse_rows(body)
            except urllib.error.HTTPError as exc:
                if exc.code == 410:
                    raise ThetaDataUnavailable(
                        f"HTTP 410 for {path}: this terminal rejects {API_VERSION} paths. "
                        "A v2 terminal listens on 25510; v3 is required."
                    ) from exc
                if exc.code == 472:
                    # ThetaData's documented "no data for this request" code.
                    return []
                if exc.code != 429 and exc.code < 500:
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                    raise ThetaDataUnavailable(
                        f"HTTP {exc.code} for {path}: {detail}"
                    ) from exc
                last = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                if _is_connection_refused(exc):
                    # Nothing is listening on the port. Retrying cannot change
                    # that, and three backoffs only delay the message that
                    # tells the user to start the terminal.
                    raise ThetaDataUnavailable(
                        f"nothing listening at {self.base_url} ({exc})"
                    ) from exc
                last = exc

            if attempt < self.retries:
                sleep_for = self.backoff * (2 ** (attempt - 1))
                logging.warning(
                    "ThetaData %s attempt %d/%d failed (%s); retrying in %.1fs",
                    path, attempt, self.retries, last, sleep_for,
                )
                time.sleep(sleep_for)

        raise ThetaDataUnavailable(f"{path} failed after {self.retries} attempts: {last}")

    # ------------------------------------------------------------------- probe

    def probe(self) -> ProbeResult:
        """One cheap call that distinguishes 'not running' from 'wrong version'."""
        try:
            self._get_json("option/list/symbols")
            return ProbeResult(reachable=True, base_url=self.base_url)
        except ThetaDataUnavailable as exc:
            return ProbeResult(
                reachable=False,
                base_url=self.base_url,
                detail=str(exc),
                version_mismatch="410" in str(exc),
            )

    # --------------------------------------------------------------- listings

    def list_option_symbols(self) -> List[str]:
        """Every underlying with listed options, per the terminal's overnight index."""
        rows = self._get_json("option/list/symbols")
        return sorted({str(r["symbol"]).upper() for r in rows if r.get("symbol")})

    def list_option_dates(self, symbol: str) -> List[str]:
        """Dates with option data for one underlying, ascending ISO strings.

        This is the measurement that settles how much history the current
        subscription actually grants -- the published start dates are marketing
        copy and the free tier's own page contradicts itself (prose says one
        year, the access table says 2023-06-01).
        """
        rows = self._get_json("option/list/dates", symbol=symbol)
        out = []
        for r in rows:
            raw = r.get("date") if isinstance(r, dict) else r
            iso = _to_iso_date(raw)
            if iso:
                out.append(iso)
        return sorted(set(out))

    def list_expirations(self, symbol: str) -> List[str]:
        rows = self._get_json("option/list/expirations", symbol=symbol)
        out = []
        for r in rows:
            iso = _to_iso_date(r.get("expiration") if isinstance(r, dict) else r)
            if iso:
                out.append(iso)
        return sorted(set(out))

    # ---------------------------------------------------------------- history

    def option_history_eod(
        self,
        symbol: str,
        *,
        start: Union[str, date],
        end: Union[str, date],
        expiration: str = "*",
        strike: str = "*",
        right: str = "both",
        strike_range: Optional[int] = None,
        max_dte: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """EOD option bars with the closing NBBO quote.

        ``strike_range=n`` asks the server for n strikes either side of spot
        plus the at-the-money strike, and ``max_dte`` caps days to expiration.
        Pushing both filters server-side is what makes this affordable: without
        them, ``expiration=*`` with ``strike=*`` returns the entire chain for
        every listed expiry on every date, which is most of a gigabyte per
        symbol-year and almost all of it deep wings we would discard anyway.

        Note what is **not** here: implied volatility and greeks. The EOD
        endpoint carries only prices and quotes; IV is a Standard/Pro endpoint.
        We invert Black-Scholes instead -- see ``logic.black_scholes_iv``.
        """
        return self._get_json(
            "option/history/eod",
            symbol=symbol,
            expiration=expiration,
            strike=strike,
            right=right,
            start_date=_to_compact_date(start),
            end_date=_to_compact_date(end),
            strike_range=strike_range,
            max_dte=max_dte,
        )


# --------------------------------------------------------------------- helpers


def _parse_rows(body: str) -> List[Dict[str, Any]]:
    """Parse a v3 JSON body into row dicts.

    v3 returns a bare array of objects, but the older format wrapped rows in
    ``{"header": ..., "response": [[...]]}``. Both are accepted so a terminal
    update cannot silently yield zero rows.
    """
    text = body.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ThetaDataUnavailable(f"non-JSON response: {text[:200]}") from exc

    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("response"), list):
            rows = payload["response"]
            fmt = (payload.get("header") or {}).get("format")
            if fmt and rows and isinstance(rows[0], (list, tuple)):
                return [dict(zip(fmt, r)) for r in rows]
            return [r for r in rows if isinstance(r, dict)]
        if payload.get("error") or payload.get("message"):
            raise ThetaDataUnavailable(str(payload.get("error") or payload.get("message")))
    return []


def _is_connection_refused(exc: Exception) -> bool:
    """True when the OS says no process is listening on the port.

    Matched by errno rather than message text so it behaves the same on Windows
    (WSAECONNREFUSED, 10061) and POSIX (ECONNREFUSED, 111).
    """
    inner = getattr(exc, "reason", exc)
    if isinstance(inner, ConnectionRefusedError):
        return True
    return getattr(inner, "errno", None) in (61, 111, 10061)


def java_version() -> Optional[str]:
    """The installed Java version string, or None if java is not on PATH.

    The terminal needs Java 21+, and "java not installed" is a different fix
    from "terminal not started" -- worth distinguishing before the user goes
    hunting for a jar they cannot run.
    """
    import shutil
    import subprocess

    if shutil.which("java") is None:
        return None
    try:
        proc = subprocess.run(
            ["java", "-version"], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # `java -version` writes to stderr on most builds.
    out = (proc.stderr or "") + (proc.stdout or "")
    first = out.strip().splitlines()[0] if out.strip() else ""
    return first or None


def _to_iso_date(value: Any) -> Optional[str]:
    """Accept 20240131, '20240131' or '2024-01-31'; return '2024-01-31'."""
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    s = str(value).strip()
    if not s:
        return None
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    try:
        return date.fromisoformat(s[:10]).isoformat()
    except ValueError:
        return None


def _to_compact_date(value: Union[str, date]) -> str:
    """The API's YYYYMMDD form."""
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    s = str(value).strip()
    return s.replace("-", "") if "-" in s else s


def liquid_option_symbols(candidates: Optional[Sequence[str]] = None) -> List[str]:
    """Universe symbols worth asking the options market about.

    The full universe is 19 commodity ETFs, but most have no usable options
    chain, and the ones that do are exactly the ones measured in Gate 1/Gate 2.
    Restricting the pull to these keeps a month's subscription sufficient.
    """
    if candidates is not None:
        return [s.strip().upper() for s in candidates if s.strip()]
    return ["GLD", "SLV", "USO", "UNG", "DBC", "XLE", "CORN", "WEAT"]
