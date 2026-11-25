"""Data layer: download, cache, quality-check, and holdout-guarded loading.

Corporate actions: yfinance auto_adjust=True returns split- and dividend-adjusted OHLC.
We store adjusted prices and verify them with quality_report() (large one-day jumps
are flagged for manual review: unadjusted splits/bonuses show up as ~-50% days).

Survivorship bias: the universe file is TODAY'S constituents. Any backtest on it
over-represents winners. Week 2 replaces this with point-in-time membership.
"""
from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str = "config.yaml") -> dict:
    """Paths inside the config are relative to the config file's folder (the repo root),
    so code works from scripts AND from notebooks/."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = str(Path(path).resolve().parent)
    return cfg


def _p(cfg: dict, rel: str) -> Path:
    return Path(cfg.get("_root", ".")) / rel


def read_universe(cfg: dict) -> list[str]:
    df = pd.read_csv(_p(cfg, cfg["data"]["universe_file"]))
    col = next(c for c in df.columns if c.strip().lower() == "symbol")
    syms = [s.strip() for s in df[col].dropna()]
    # NSE lists "DUMMY..." placeholder rows (corporate-action placeholders, not real stocks)
    return [x + cfg["data"]["suffix"] for x in syms if not x.upper().startswith("DUMMY")]


def download(tickers: list[str], cfg: dict) -> None:
    """Run on a machine with internet access. Saves one parquet per ticker."""
    import yfinance as yf
    out = _p(cfg, cfg["data"]["raw_dir"]); out.mkdir(parents=True, exist_ok=True)
    for t in tickers:
        df = yf.download(t, start=cfg["data"]["start"], auto_adjust=True, progress=False)
        if df.empty:
            print(f"EMPTY  {t}"); continue
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.to_parquet(out / f"{t.replace('^', '_')}.parquet")
        print(f"ok     {t}  {df.index[0].date()} -> {df.index[-1].date()}  n={len(df)}")


def clean_prices(df: pd.DataFrame, jump: float = 0.5, window: int = 2,
                 max_lead: int = 20, tol: float = 0.10):
    """Data hygiene for known Yahoo errors. Returns (clean_df, log).

    Rule 1, reciprocal spike: a one-day move (up OR down) > `jump` that is reversed within `window`
    days (price back within `tol` of the pre-move level) is a bad print. Drop those rows.
    window=2 on purpose: bad prints reverse at once; real events (e.g. Yes Bank, Mar 2020) take
    days to recover, and a wider window wrongly deleted that real crash.
    Rule 2, early unadjusted split: a fall > `jump` within the first `max_lead` rows is
    treated as an unadjusted split. Drop all rows before it.
    Every action is logged. Real crashes (e.g. -20% days that do not reverse) are untouched.
    """
    close = df["Close"]
    r = close.pct_change()
    keep = pd.Series(True, index=df.index)
    log = []
    n = len(df)
    for i in range(1, n):
        if abs(r.iloc[i]) > jump:
            base = close.iloc[i - 1]
            for j in range(i + 1, min(i + 1 + window, n)):
                if abs(close.iloc[j] / base - 1) < tol:
                    keep.iloc[i:j] = False
                    log.append(("spike_dropped", str(df.index[i].date()), str(df.index[j - 1].date())))
                    break
    d1 = df[keep]
    r2 = d1["Close"].pct_change()          # recomputed AFTER rule 1 (a dropped spike's reversal leg is not a split)
    for i in range(1, min(max_lead, len(d1))):
        if r2.iloc[i] < -jump:
            log.append(("early_split_trim", str(d1.index[0].date()), str(d1.index[i - 1].date())))
            d1 = d1.iloc[i:]
            break
    return d1, log


def quarantine_reason(df: pd.DataFrame, thresh: float = 0.5):
    """After cleaning, any remaining one-day move beyond +/-thresh is almost surely an unadjusted
    corporate action (demerger, split, capital reduction) or a bad print. We do NOT guess a fix:
    the whole ticker is excluded and listed. Returns (date, move) or None."""
    r = df["Close"].pct_change().dropna()
    big = r[r.abs() > thresh]
    return None if big.empty else (str(big.abs().idxmax().date()), round(float(r[big.abs().idxmax()]), 3))


def load_prices(ticker: str, cfg: dict, allow_holdout: bool = False,
                clean: bool = True) -> pd.DataFrame:
    path = _p(cfg, cfg["data"]["raw_dir"]) / f"{ticker.replace('^', '_')}.parquet"
    df = pd.read_parquet(path).sort_index()
    if clean:
        df, _ = clean_prices(df)
    if cfg["holdout"]["locked"] and not allow_holdout:
        df = df.loc[: cfg["data"]["dev_end"]]
    return df


def quality_report(df: pd.DataFrame, jump: float = 0.20) -> dict:
    """Flags: duplicate dates, non-positive prices, NaNs, |1-day return| > jump."""
    if len(df) == 0:   # e.g. listed after dev_end: nothing in the dev window
        return {"rows": 0, "first": "-", "last": "-", "dup_dates": 0, "nan_close": 0, "nonpositive": 0,
                "jumps": [], "empty": True}
    r = df["Close"].pct_change()
    return {
        "rows": len(df),
        "first": str(df.index[0].date()), "last": str(df.index[-1].date()),
        "dup_dates": int(df.index.duplicated().sum()),
        "nan_close": int(df["Close"].isna().sum()),
        "nonpositive": int((df["Close"] <= 0).sum()),
        "jumps": [(str(d.date()), round(float(x), 3)) for d, x in r[r.abs() > jump].items()],
    }
