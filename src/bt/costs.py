"""Indian equity-delivery cost model. Rates come from config.yaml (sourced in PROJECT_LOG.md)."""
from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    brokerage_pct: float = 0.0
    stt_buy: float = 0.001
    stt_sell: float = 0.001
    stamp_buy: float = 0.00015
    exch_txn: float = 0.000030699
    sebi: float = 0.000001
    gst: float = 0.18
    dp_per_sell_scrip: float = 15.34
    slippage_bps: float = 5.0

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        return cls(**cfg["costs"])

    def fees(self, side: str, notional: float) -> float:
        """Rupee fees (excluding slippage) for one fill. side in {'buy','sell'}."""
        if side not in ("buy", "sell"):
            raise ValueError(side)
        if notional <= 0:
            return 0.0
        brokerage = self.brokerage_pct * notional
        exch = self.exch_txn * notional
        sebi = self.sebi * notional
        gst = self.gst * (brokerage + exch + sebi)
        if side == "buy":
            return brokerage + exch + sebi + gst + self.stt_buy * notional + self.stamp_buy * notional
        # DP charge already includes its GST (Zerodha quote)
        return (brokerage + exch + sebi + gst + self.stt_sell * notional
                + self.dp_per_sell_scrip)

    def fill_price(self, price: float, side: str) -> float:
        """Adverse slippage: you pay more on buys, receive less on sells."""
        s = self.slippage_bps / 1e4
        return price * (1 + s) if side == "buy" else price * (1 - s)

    def round_trip_bps(self, notional: float = 1e5) -> float:
        """Total round-trip cost in bps of notional, incl. slippage, at a given trade size."""
        fees = self.fees("buy", notional) + self.fees("sell", notional)
        return fees / notional * 1e4 + 2 * self.slippage_bps
