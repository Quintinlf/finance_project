# Tier 2 scope: trading volatility through options

Status: **scope only — nothing here is built, and nothing here is approved to trade.**
Written 2026-10-16.

## Why this is on the table

Direction is closed. Three independent measurements at n≈24,500 say so: 20 of 20
horizon tests below the base rate, the full calibration range worth 4.1 points of
real probability, and the move-size hypothesis failing with accuracy *declining*
to 46.6% on the largest predictions.

One thing did measure strongly. Forward realized volatility is predictable at
**r = 0.708 (R² ≈ 0.50)**, positive on every symbol in the universe, roughly 36×
stronger than anything directional. Direction is unpredictable; magnitude is not.

Shares cannot express that. A share position pays for direction and nothing else.
Options are the only instrument available here that pays for magnitude
independent of which way the move goes.

## The crux, stated before anything else

**r = 0.708 is almost certainly not edge, and the scope exists to find out whether
any part of it is.**

That number measures our forecast against *realized* volatility. But we would not
be trading against realized volatility — we would be trading against **implied**
volatility, which is the options market's own forecast of the same quantity. The
market has already priced its expectation in. The real question is not "can we
predict volatility," which we can. It is:

> Can we predict volatility **better than the options market already does**?

That is a completely different question and a much harder one. It is entirely
possible, and frankly likely, that regressing realized vol on implied vol leaves
our forecast with no significant residual information at all. If so this road is
closed too, and the whole programme should stop.

That test is cheap and it comes first. Everything below is contingent on it.

### Second crux: the variance risk premium is not a signal

Implied volatility sits systematically *above* subsequent realized volatility.
That gap is the variance risk premium, and it is one of the most documented and
most harvested effects in finance. It is compensation for bearing tail risk, not
a mispricing.

This matters because "our forecast is below IV" will be true most of the time,
and naively reading that as "sell volatility" is just harvesting a known premium
with no skill involved — while taking on exactly the tail risk the premium pays
for. Any test must measure our forecast against the *typical* premium, not
against zero.

## Data: what we have, and the binding constraint

Checked live against the account on 2026-10-16, against a UNG chain of 768 contracts:

| | Status |
|---|---|
| Option chains | Available |
| Implied volatility | **Available free** — 349/768 contracts (45%), on the INDICATIVE feed |
| Greeks (delta) | **Available free** — same 349 contracts |
| OPRA real-time feed | Not enabled: "OPRA agreement is not signed" |
| **Historical IV / chains** | **Not available** |

Two corrections to prior assumptions are worth recording. The code comment in
`logic/options_engine.py` claiming the delayed feed returns no greeks is **wrong**
— it returns them for roughly the liquid half of the chain. And OPRA is not a
paid upgrade here; it is an unsigned agreement in the Alpaca dashboard, worth
signing regardless since it costs nothing on a paper account.

**The binding constraint is the last row.** Alpaca serves current snapshots, not
history. yfinance gives current chains only. Without historical IV *the core test
cannot be backtested at all.* That leaves two paths:

- **Accumulate forward.** Snapshot IV daily into the existing database, then run
  the test once enough has built up. Costs nothing, requires no new dependency,
  and is the obvious thing to start immediately — but needs roughly **4–6 months**
  before the sample supports a conclusion. (Note what this project has already
  learned the hard way: every dramatic finding at small n evaporated at large n.
  Running this test early would produce a number, and that number would be
  worthless.)
- **Buy history.** ORATS, IVolatility and CBOE DataShop sell it. Hundreds to low
  thousands of dollars. This converts a 6-month wait into about a week of work.

There is no third option. This is the single most important thing to decide,
because it sets the timeline for everything else.

## What would have to be true

Hard gates, in order. Each one stops the programme if it fails.

**Gate 0 — incremental information over IV.**
Regress realized vol on both implied vol and our forecast. Our forecast's
coefficient must be significantly non-zero *with IV already in the model*, out of
sample, corrected for overlapping windows. A high R² on our forecast alone proves
nothing; IV will have a high R² too.
→ *Fails: stop. The entire thesis is dead and the honest conclusion is that this
system has no tradeable edge.*

**Gate 1 — survives option transaction costs.**
Share costs on this universe run 12–45bp. Option spreads on these same commodity
ETFs routinely run **5–15% of premium** — between ten and a hundred times worse.
Break-even must be recomputed from scratch against measured option spreads, not
share spreads, measured during regular market hours.
→ *Fails: stop. This is the most likely place to die after Gate 0.*

**Gate 2 — expressible at this account size.**
At $494 the position is one contract of one thing, and many of these chains are
thin. Defined-risk structures need the width to fit inside the position cap.
→ *Fails: the strategy may be real but unreachable until the account is funded.
That is a capital decision, not a code decision.*

**Gate 3 — survives paper trading.**
Minimum 100 paper trades scored through the existing `model_performance_tracker`,
judged against the base rate, with Bonferroni correction across every variant
tried. No real money before this.
→ *Fails: stop.*

## Risks, stated plainly

This is categorically more dangerous than the current book, and the difference is
not a matter of degree.

**Short volatility can lose more than the account holds.** The current system is
long-only shares: the worst case is the position going to zero. Selling premium
on something like UNG — 58% annualised vol, gaps on weather and inventory
reports — can lose a multiple of the premium collected overnight, with no
opportunity to react. A $494 account can be wiped out, or taken negative, on a
single gap. The variance risk premium pays steadily and then takes it all back at
once; that is precisely why it pays.

**Practically, naked selling is not available anyway.** Alpaca gates short
options behind approval levels and margin a $494 account does not have. Only
defined-risk structures — long straddles/strangles, vertical and calendar spreads
— are realistically on the table. *This constraint should be treated as a
feature and kept permanently, not as an obstacle to work around.*

**Long volatility bleeds.** Buying straddles has bounded loss but pays theta every
single day and needs the realized move to exceed the premium paid. Hit rates are
low by design, so the losing streaks are long and psychologically brutal, and the
sample needed to distinguish skill from luck is correspondingly large.

**Mechanics the share book never had to handle:** early assignment on American
options, pin risk at expiry, expiry-day liquidity collapse, and a position whose
value depends on three things moving at once rather than one.

**Honest prior:** selling volatility is a crowded, well-understood trade run by
people with better data, better execution and vastly more capital. The odds that
a retail vol forecast carries information the options market has missed are low.
Low is not zero, which is why Gate 0 is worth running — but the expected outcome
is that it fails, and the plan should be built to find that out cheaply rather
than to avoid finding it out.

## Phases

| Phase | Work | Duration | Money at risk |
|---|---|---|---|
| **0a** | Start snapshotting IV + greeks daily into the DB. Sign the OPRA agreement. | ~1 day to build | none |
| **0b** | Gate 0: does our forecast beat IV? | ~1 week *if* history is bought; otherwise 4–6 months of accumulation | none |
| **1** | Measure real option spreads; rebuild the cost model; recompute break-even. | ~1 week | none |
| **2** | Defined-risk paper trading, ≥100 scored trades. | 3–6 months | none |
| **3** | Real money, defined-risk only, sized to survive being wrong. | — | yes — a separate decision, not part of this scope |

Phase 0a is worth starting now regardless of what is decided about buying history,
because the accumulation clock only runs once it has been started, and if the
answer turns out to be "buy the data" the snapshots become the out-of-sample set.

## What this does not do

It does not make the current directional strategy work. If Gate 0 passes, this is
a *different strategy* that happens to share the data plumbing, the scoring
apparatus and the execution layer. The existing signal components do not feed it.

## Recommendation

Build Phase 0a — the daily IV snapshot — because it is a day's work, it risks
nothing, and the clock does not start until it exists. Then decide on buying
historical data, which is the real fork: a few hundred dollars converts a
six-month wait into a week.

Do not fund the account for this yet. Gate 0 is the thing standing between here
and knowing whether any of it is real, and no amount of capital moves that gate.
