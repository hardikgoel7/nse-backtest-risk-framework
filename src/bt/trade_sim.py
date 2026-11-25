"""Trade simulator (daily OHLC, no look-ahead).

TWO ENTRY STYLES
1) simulate_trade: entry at the CLOSE of day T+1 (+ slippage). Stop = low of the signal candle
   (A2: day T-1, A: day T) or `stop_override`. Target = entry*(1+target_pct), or entry + target_r*(entry-stop),
   or (p['exit_rsi']) hold until a daily close with RSI >= exit_rsi and sell at the NEXT close.
2) simulate_breakout_trade (the strategies' entry rule, clarified 2026-11-22): the signal candle's HIGH is the entry
   TRIGGER (buy-stop), its LOW is the STOP, risk = fill - stop, target = fill + target_r x risk.
   The order is live for `window` days starting the day AFTER the signal is known (day T+1).
   - first day with open >= trigger          -> filled at the OPEN (gap), worse than the trigger
   - first day with high >= trigger, low > stop -> filled at the trigger
   - low <= stop before any fill (or same day as an intraday trigger, order unknown) -> setup INVALIDATED, no trade
   - window ends without a fill -> 'unfilled'
   Slippage is added to the fill. Risk and the 3R target use the actual fill.
Exits: checks start the day AFTER entry; a gap through the stop exits at the open; stop is checked before target
on the same day; time cap exits at the close; trades running past the end of data are 'censored'.
"""
import numpy as np
from .costs import CostModel


def _exit(O, H, L, C, e, entry, stop, p, cost, notional, rsi=None):
    """Walk forward from entry day e. Returns the result dict."""
    n = len(C)
    if e + p["max_hold_days"] >= n:
        return {"status": "censored"}
    if not stop < entry:
        return {"status": "skipped"}
    use_rsi = p.get("exit_rsi") is not None and rsi is not None
    target = np.inf if use_rsi else entry + p["target_r"] * (entry - stop) if p.get("target_r") else entry * (1 + p["target_pct"])
    reason, px, j_exit = "timeout", C[e + p["max_hold_days"]], e + p["max_hold_days"]
    pending = False
    for j in range(e + 1, e + p["max_hold_days"] + 1):
        if O[j] <= stop:
            reason, px, j_exit = "stop", O[j], j; break
        if L[j] <= stop:                       # stop first, even if target also touched
            reason, px, j_exit = "stop", stop, j; break
        if O[j] >= target:
            reason, px, j_exit = "target", O[j], j; break
        if H[j] >= target:
            reason, px, j_exit = "target", target, j; break
        if pending:
            reason, px, j_exit = "rsi", C[j], j; break
        if use_rsi and rsi[j] >= p["exit_rsi"]:
            pending = True
    return _result(entry, stop, px, reason, j_exit - e, cost, notional)


def _result(entry, stop, px, reason, days, cost, notional):
    exit_fill = cost.fill_price(px, "sell")
    risk = (entry - stop) / entry
    gross = exit_fill / entry - 1
    sell_notional = notional / entry * exit_fill
    net = (sell_notional - cost.fees("sell", sell_notional) - notional - cost.fees("buy", notional)) / notional
    return {"status": "ok", "reason": reason, "days": days, "entry": entry, "stop": stop,
            "risk_pct": risk, "gross": gross, "net": net, "R": gross / risk}


def simulate_trade(O, H, L, C, t, stop_lag, p, cost: CostModel, notional=1e5, rsi=None, stop_override=None):
    n = len(C)
    e = t + 1
    if t - stop_lag < 0 or e >= n:
        return {"status": "censored"}
    if e + p["max_hold_days"] >= n:
        return {"status": "censored"}
    entry = cost.fill_price(C[e], "buy")
    stop = L[t - stop_lag] if stop_override is None else stop_override
    return _exit(O, H, L, C, e, entry, stop, p, cost, notional, rsi)


def simulate_breakout_trade(O, H, L, C, t, trigger, stop, window, p, cost: CostModel, notional=1e5):
    """Breakout entry: buy-stop at `trigger` (high of the signal candle), stop at `stop` (its low).
    Order live on days t+1 .. t+window. See module docstring for fill / invalidation rules."""
    n = len(C)
    if not (np.isfinite(trigger) and np.isfinite(stop)) or not stop < trigger:
        return {"status": "skipped"}
    last = min(t + window, n - 1)
    for j in range(t + 1, last + 1):
        if O[j] >= trigger:                                    # gap above the trigger: filled at the open
            if j + p["max_hold_days"] >= n:
                return {"status": "censored"}
            entry = cost.fill_price(O[j], "buy")
            if L[j] <= stop:                                   # gapped up, then fell through the stop the same day
                return _result(entry, stop, stop, "stop", 0, cost, notional) if stop < entry else {"status": "skipped"}
            return _exit(O, H, L, C, j, entry, stop, p, cost, notional)
        if L[j] <= stop:                                       # stop touched before any fill, or ambiguous same day
            return {"status": "invalidated"}
        if H[j] >= trigger:                                    # intraday fill at the trigger
            if j + p["max_hold_days"] >= n:
                return {"status": "censored"}
            entry = cost.fill_price(trigger, "buy")
            return _exit(O, H, L, C, j, entry, stop, p, cost, notional)
    return {"status": "unfilled"} if t + window <= n - 1 else {"status": "censored"}
