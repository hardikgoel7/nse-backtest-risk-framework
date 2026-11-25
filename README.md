# NSE Equity Strategy Backtesting & Risk Framework

A research-grade backtesting framework, event-driven portfolio simulator and risk toolkit for
rule-based equity strategies on Indian markets, built to test whether a set of RSI and breakout
trading rules has a measurable edge, and to measure their risk properly.

**Headline result: no edge was detected.** Rule sets were pre-registered and tested against
random-entry controls with identical exits and costs; the three RSI rules were then tested once on
locked out-of-sample data, and a breakout proxy and a sector filter on development data. None beat
its control. The apparent profits came from market drift, survivorship bias and
payoff asymmetry. That negative result, and the machinery that produced it, is the deliverable.

**This is research code. No live trading, no real money, no profit claims.**

---

## 1. What was tested

| | Entry | Stop | Target | Max hold |
|---|---|---|---|---|
| **S1** | Monthly RSI > 60, weekly RSI > 60, daily RSI bounce at 40 (prior day in 38–42, today higher) | Low of the signal candle | 3 × risk | 15 days |
| **S2** | Same, but daily RSI bounce at 60 (band 58–62) | Low of the signal candle | 3 × risk | 15 days |
| **S3** | Monthly RSI in 38–42 ("value buy") | Low of the signal month | 3 × risk | 63 days (~3 months) |
| **S4** | Multi-year breakout: monthly close ≥ 1.02 × highest monthly close of the prior 60 months | Low of the breakout month | 3 × risk | 63 days |

Entry is a **buy-stop above the signal candle's high**. Risk is the fill price minus the stop.
The target is three times that distance above the fill.

**S4 is a proxy, not a cup-and-handle detector.** The visual pattern has no unambiguous numeric
definition, so S4 tests the part that can be coded: a breakout above a five-year base. It has no cup
shape, handle or volume test. Because the data starts in 2010, S4 signals begin around 2015.

Earlier variants (A: daily RSI in 38–42 without the bounce requirement; A2 = S1 with a fixed +30%
or +20% price target, or an exit at daily RSI ≥ 60) were also tested. All are reported below.

---

## 2. Method

**Universe.** Nifty 500 current constituents, daily OHLCV from Yahoo Finance (`.NS`), split- and
dividend-adjusted. Of 500 names, 454 were tradable: 13 were excluded for unadjusted corporate
actions and 32 for having no history in the test window. A liquidity floor of ₹5 crore 20-day
average traded value is applied to every strategy and every control alike.

**Windows.** Development 2010-01-01 to 2023-12-31. Holdout 2024-01-01 to 2026-11-20, locked by the
data loader and released exactly once per rule set. A lock file in `reports/` refuses a second run.

**No look-ahead.** A signal is known at the close of day T; the order goes live from T+1 and is
never filled on day T. Weekly and monthly RSI use only the last *completed* bar, so a partial
month can never leak into a signal. A unit test removes all future data and asserts the features
at T are unchanged.

**Costs.** Zero brokerage on delivery, STT 0.1% each side, stamp duty 0.015% on buy, NSE exchange
charge ₹306.99 per crore per side, SEBI ₹10 per crore, GST 18%, DP charge ₹15.34 per scrip sold,
plus 5 bps slippage per side. About 34 bps round trip on a ₹1 lakh position.
Source: [Zerodha charges](https://zerodha.com/charges) and NSE circular NSE/FA/73061.

**Controls.** Every strategy is compared against the same trade structure applied to days the
strategy did not select:

- **Random days** — 400 random days per stock.
- **All days** — every eligible day, and for the monthly rule every month. This is the primary
  yardstick; it has no sampling error.
- **Stock-matched** — each signal trade minus the all-day average of the *same stock*, so a stock
  that rose all decade cannot flatter the signal. Declared primary.
- **Month-matched** — each signal trade minus the all-day average across all stocks in the *same
  calendar month*, so market-wide moves cancel.
- **Bounce-only** — the daily RSI bounce without the monthly and weekly trend filter, which
  isolates what the trend filter adds.

**Statistics.** Confidence intervals come from a month-clustered bootstrap (10,000 draws), because
signals cluster in time across stocks and treating trades as independent would understate the
error. Intervals are Bonferroni-adjusted within each family of comparisons.

---

## 3. Results

### 3.1 Signal versus baseline, mean net return per trade

Development 2010–2023, Nifty 500, 3R exit. Excess = strategy minus baseline.

| Strategy | Trades | Own mean | vs all-day pooled | vs stock-matched | vs month-matched |
|---|---|---|---|---|---|
| S1 | 831 | +0.58% | +0.20% | **+0.08%** | +0.78% |
| S2 | 3,382 | +0.25% | −0.13% | **−0.21%** | −0.06% |
| S3 | 363 | +4.69% | −0.31% | **+0.58%** | −0.97% |

Holdout 2024–2026:

| Strategy | Trades | Own mean | vs all-day pooled | vs stock-matched | vs month-matched |
|---|---|---|---|---|---|
| S1 | 330 | +0.04% | +0.20% | **−0.06%** | +0.29% |
| S2 | 1,322 | −0.18% | −0.02% | **−0.21%** | −0.24% |
| S3 | 92 | +2.12% | +0.05% | **+3.32%** | −2.58% |

**Every confidence interval includes zero**, before and after the multiple-comparison adjustment.

### 3.2 The result that matters most

S3 earned **+4.69% per trade** on development data, and its interval excluded zero. Taken alone,
that looks like a working strategy.

Applying the identical stop, target and holding rule to **every month** of the same stocks earned
**+5.01%**. The signal did slightly *worse* than picking a month at random. The profit was the
market, not the rule.

The all-day baseline shows this directly. For 63-day holds it was +5.01% on development data and
+2.07% on holdout; for 15-day holds, +0.38% and −0.16%. The structure's return tracks the market
regime, and the typical trade loses money in both (medians −2.4% and −2.6%).

### 3.3 A hypothesis, pre-registered and rejected

S1 looked better against its own month (+0.78%) than against the pooled average (+0.20%), with a
plausible mechanism: the rule requires a daily RSI dip to 38–42, so its signals cluster in weaker
months. Out of sample this fell to **+0.29%** with an interval of −0.43% to +1.10%. Not replicated.

### 3.4 Exit rules

Four exits were tested on the earlier A2 rule (Nifty 500, development): a +30% target, a +20%
target, a 3R target and an exit at daily RSI ≥ 60. Excess over random entries was +0.17%, +0.21%,
+0.18% and +0.18%. All insignificant. These four reuse the same entries, so they are one piece of
evidence, not four. The +30% exit was carried to the holdout and gave +0.28% over random,
also insignificant.

### 3.5 Robustness

Trimming the best and worst 1% of trades leaves the signal-minus-control gap near zero everywhere.
Outliers do not hide an edge, but they are what makes the raw averages positive: in the holdout,
the top 1% of A2's trades produced 72% of its profit, and removing the top 1% of S1 and S2 trades
turns both negative.

### 3.6 Portfolio engine

Event-driven simulation of S1 over 2010–2026: ₹10 lakh, 1% of equity risked per trade, a 10-position
cap, 20% of equity per position, no leverage, costs on every fill, idle cash earning the 6.5%
risk-free rate.

| | Strategy | Nifty 50 |
|---|---|---|
| CAGR (full period) | 7.63% | 10.12% |
| Volatility | 9.80% | 16.38% |
| Max drawdown | −17.67% | −38.44% |
| Calmar | 0.43 | 0.26 |
| Sharpe | 0.14 | 0.27 |

801 trades: 14.9% hit the 3R target, 49.9% were stopped out, 35.2% timed out. Win rate 39.0%.

**The low volatility and shallow drawdown are not risk management.** The portfolio held an average
of **29.3% of equity in the market** and was entirely in cash on 36.2% of days, with a beta of
**0.19**. It is mostly a T-bill fund with a small equity sleeve. Over the full period it earned
less than the index and had a worse Sharpe ratio.

On the holdout the strategy returned 8.90% against the index's 6.25%, with alpha of +2.63% a year —
but a t-statistic of 0.39, so this is noise, not outperformance.

**An engine finding:** with an average stop 5% below entry, risking 1% of equity makes each
position 20% of equity, which is exactly the position cap. The 10-position limit can therefore
never bind; the real limit is about five positions. 318 orders were skipped for want of cash, while
only 7 were skipped for want of a slot.

---

### 3.7 Top-down sector filter

Rule: take a stock signal only when its sector index has daily, weekly and monthly RSI above 60.
Sector indices are equal-weighted, built from the universe's own members of each sector (11 sectors).

The filter raised raw returns sharply: S4 from **+3.54%** to **+6.31%** per trade (+78%), S3 from
+4.46% to +6.24%. The all-day baseline in the same sector state rose just as much, from **+3.72%** to
**+6.86%**. Excess over that baseline did not improve (S4 −0.55%, S3 −0.62% with the filter on;
none of 8 comparisons significant). **The filter selects good market conditions, not good stocks.**
Development data only, so this is exploratory.

### 3.8 Volatility: EGARCH(1,1)

EGARCH(1,1) with Student-t errors on the Nifty 50, refit every January on data through the previous
December, so no forecast uses a parameter estimated on later data.

- Asymmetry γ ≈ **−0.11**: a fall raises next-day volatility more than an equal rise (leverage effect).
- Persistence β ≈ **0.97**: volatility regimes last.
- Degrees of freedom fall from ~59 to ~**7.7** in recent fits: fat tails, so Student-t over normal.

Splitting trades by volatility regime gave no stable effect: the signal-minus-control gap changed
sign between development and holdout for S1 and S2.

**Stop distance.** Measured in units of each stock's 20-day realised volatility, the median stop sat
at **2.0σ**. Stops inside 1.5σ were hit **62–71%** of the time in both periods. Wider stops did not
reliably improve returns, because with a fixed 3R target a wider stop also moves the target further
away: target hits fell from ~23% to ~9% as stops widened. Stop-outs were traded for time-outs.

### 3.9 Value at Risk and Expected Shortfall

1-day VaR and ES at 99% and 97.5%, forecast three ways using only past data (rolling 250-day
historical, rolling normal, EGARCH-t), backtested over 3,413 trading days from 2013 with the
Kupiec test (right number of breaches?) and the Christoffersen test (breaches bunched together?).

On the Nifty 50 at 99%, 34 breaches were expected:

- **Historical: 58. Normal: 65.** Too many, and at 97.5% clustered: after a breach, the chance of
  another the next day was 8.4% against the 2.5% promised. Static methods react too slowly.
- **EGARCH-t: 12.** No clustering at all, but too conservative. Its average VaR (2.26%) was *lower*
  than historical (2.52%), so it was not more fearful overall, only at the right time. Its remaining
  breaches were genuine jump days, averaging a 5.2% loss.
- **ES:** the normal distribution understated losses on breach days by about 15–18% (99%: 2.52%
  promised, 3.06% realised). Historical ES was close.

No method passed the Kupiec test. On the S1 portfolio the EGARCH fit was degenerate, because the
portfolio is fully in cash on about a third of days; historical and normal VaR again failed.

## 4. Limitations

1. **Survivorship bias.** The universe is today's Nifty 500. Companies that failed or were delisted
   are missing, so absolute returns are biased upward. Comparisons against controls drawn from the
   same universe are far more reliable than any absolute figure here.
2. **Thirteen stocks excluded.** Any stock with a one-day move beyond ±50% surviving the cleaning
   step was dropped whole, rather than guessing a corporate-action adjustment. One of them, Yes Bank,
   was a real event, so dropping it removes a genuine loser and biases results slightly upward.
3. **Costs are 2026 rates applied to 2010–2026.** STT, stamp duty and exchange fees all differed
   historically.
4. **Slippage of 5 bps per side is an assumption**, not a measurement, and is optimistic for
   less liquid names even above the ₹5 crore liquidity floor.
5. **The risk-free rate is one constant (6.5%)**, not a daily series. Every Sharpe and Sortino is
   reported with a ±1% sensitivity. Source: RBI Handbook of Statistics, Table 217/221, auctions of
   91-day Treasury bills — roughly 8.3–8.6% in 2014-15, 6.4–6.8% in 2024, 5.4–5.9% in 2025.
6. **Many comparisons were run.** Across the project: two universes, four exit rules, five baselines
   and four rule sets. Bonferroni corrections were applied within each family, not across the whole
   project. Since nothing was significant anywhere, multiplicity cannot flatter the conclusion —
   but a positive finding here would have needed a far higher bar.
7. **The holdouts are spent.** Each rule set was released once. No re-runs, no tuning afterwards.
8. **Discretionary elements are untested.** The real process this was modelled on also uses chart
   reading, visual pattern judgement and stock selection. None of that is in the backtest.
9. **Benchmark mismatch.** The strategies trade mid and small caps but are compared with the Nifty 50;
   a Nifty 500 total-return index would be fairer.
10. **Sector labels are today's**, and sector indices are equal-weighted, not free-float weighted.
11. **No point-in-time index membership**, no corporate-action database, and trades still open at the
   end of the data are dropped as censored.

---

## 5. Repository

```
config.yaml              all assumptions: universe, costs, strategy thresholds, portfolio rules
src/bt/
  data.py                loading, cleaning, holdout lock, quarantine
  indicators.py          Wilder RSI; weekly/monthly RSI on completed bars only
  signals.py             the rules as code
  trade_sim.py           single-trade simulator (close entry and breakout entry)
  event_study.py         forward returns, clustered bootstrap, multiple-comparison adjustment
  costs.py               Indian transaction costs and slippage
  metrics.py             CAGR, vol, Sharpe, Sortino, drawdown, Calmar, alpha/beta
  risk.py                VaR/ES (historical, normal, EGARCH-t), Kupiec and Christoffersen tests
scripts/
  fetch_data.py          download, quality report, cleaning log, quarantine list
  run_benchmark.py       buy-and-hold benchmark
  run_event_study.py     forward-return study with bootstrap CIs
  run_exit_rules.py      exit-rule comparison against controls
  run_strategies.py      S1/S2/S3 with breakout entry
  run_baseline_all.py    all-day, stock-matched and month-matched baselines
  run_robustness.py      outlier-trimmed re-check
  run_portfolio.py       event-driven portfolio engine
  fetch_sectors.py       sector labels for the universe
  run_topdown.py         sector filter and the S4 breakout proxy
  run_volatility.py      EGARCH regimes and stop-distance analysis
  run_var.py             VaR/ES forecasts and backtests
tests/                   60 unit tests, including no-look-ahead causality tests
```

Price data and generated reports are not committed: prices are re-downloaded from Yahoo Finance by
`fetch_data.py`, and every result above is regenerated by the scripts.

### Reproducing

Download the Nifty 500 constituent list from niftyindices.com and save it as
`data/nifty500_constituents.csv` (it needs a `Symbol` column). Then:

```bash
pip install -r requirements.txt
pytest -q
python scripts/fetch_data.py --universe data/nifty500_constituents.csv --quiet
python scripts/run_strategies.py --universe data/nifty500_constituents.csv --tag s_dev
python scripts/run_baseline_all.py --universe data/nifty500_constituents.csv --tag base_dev
python scripts/run_baseline_report.py --tag base_dev
python scripts/run_robustness.py
python scripts/run_portfolio.py --universe data/nifty500_constituents.csv
python scripts/fetch_sectors.py --universe data/nifty500_constituents.csv
python scripts/run_topdown.py --universe data/nifty500_constituents.csv
python scripts/run_volatility.py --universe data/nifty500_constituents.csv
python scripts/run_var.py
```

The holdout runs are deliberately not listed: they are already spent and the lock files refuse them.

---

## 6. What this project demonstrates

Building a backtest that produces a large number is easy. Building one that can tell the difference
between an edge and a rising market is the hard part. This framework does the second thing:
pre-registered rules, matched controls, clustered confidence intervals, multiple-comparison
corrections, explicit data cleaning with every exclusion logged, a single locked out-of-sample
test per rule set, and risk forecasts that are themselves backtested.

The answer it returned was no. That is a result, and it is reported as one.
