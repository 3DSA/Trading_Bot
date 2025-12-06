#!/usr/bin/env python3
"""
Backtest Hybrid - ADX Regime Strategy Validator

Tests the Hybrid Regime strategy that switches between:
- TREND MODE (ADX > 25): Momentum scalping with full position
- CHOP MODE (ADX < 20): Mean reversion with half position
- BUFFER ZONE (20-25): No trading

Features:
- ADX-based regime detection
- RSI + Bollinger Bands for mean reversion
- Regime-specific position sizing and stops

Usage:
    python backtest_hybrid.py --symbol TQQQ
    python backtest_hybrid.py --symbol TQQQ --days 10
    python backtest_hybrid.py --compare  # Compare Hybrid vs Trend-only

Author: Bi-Cameral System
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Configuration
BASE_DIR = Path(__file__).parent
CONFIG_PATH = BASE_DIR / "config.json"


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl_pct: float = 0.0
    pnl_dollars: float = 0.0
    exit_reason: str = ""
    regime: str = ""  # TREND or CHOP
    position_size: str = ""  # FULL or HALF
    entry_adx: float = 0.0
    entry_rsi: float = 0.0


class BacktestHybrid:
    """
    Backtester for the Hybrid Regime strategy.
    Simulates trades using historical 1-minute data.
    """

    # Position sizing
    FULL_POSITION = 10000  # $10,000 for TREND mode
    HALF_POSITION = 5000   # $5,000 for CHOP mode (50%)

    # TREND MODE risk (ADX > 25)
    TREND_STOP_LOSS = 0.01      # 1% stop loss
    TREND_TAKE_PROFIT = 0.02    # 2% take profit

    # CHOP MODE risk (ADX < 20) - tighter for mean reversion
    CHOP_STOP_LOSS = 0.005      # 0.5% tight stop
    CHOP_TAKE_PROFIT = 0.01     # 1% take profit (smaller target)

    # ADX thresholds
    ADX_PERIOD = 14
    ADX_TREND_THRESHOLD = 25   # Above = TREND mode
    ADX_CHOP_THRESHOLD = 20    # Below = CHOP mode

    # RSI for mean reversion
    RSI_PERIOD = 14
    RSI_OVERSOLD = 30          # Buy when RSI < 30

    # Bollinger Bands
    BB_PERIOD = 20
    BB_STD_DEV = 2

    def __init__(self, symbol: str, days: int = 7, mode: str = "hybrid"):
        """
        Initialize the backtester.

        Args:
            symbol: The ticker symbol to backtest (TQQQ, SQQQ, etc.)
            days: Number of days of historical data to use
            mode: "hybrid" (both regimes), "trend_only", or "chop_only"
        """
        self.symbol = symbol
        self.days = days
        self.mode = mode

        # Load strategy config
        self.config = self._load_config()
        self.strategy = self.config.get("strategy", {})

        # Strategy parameters
        self.ema_fast = self.strategy.get("ema_fast", 9)
        self.ema_slow = self.strategy.get("ema_slow", 21)
        self.vol_spike_threshold = self.strategy.get("vol_spike_threshold", 1.5)
        self.min_volume = self.strategy.get("min_volume", 1000)

        # Time filters
        time_filters = self.strategy.get("time_filters", {})
        self.morning_start = time_filters.get("morning_start", "09:30")
        self.lunch_start = time_filters.get("lunch_start", "11:45")
        self.lunch_end = time_filters.get("lunch_end", "13:15")
        self.day_end = time_filters.get("day_end", "15:55")

        # Trade tracking
        self.trades: list[Trade] = []
        self.position: Optional[dict] = None

        # Stats
        self.trend_signals = 0
        self.chop_signals = 0
        self.buffer_blocked = 0

        print(f"[INIT] BacktestHybrid for {symbol} ({mode} mode)")
        print(f"[INIT] TREND: ADX>{self.ADX_TREND_THRESHOLD}, SL={self.TREND_STOP_LOSS*100:.1f}%, TP={self.TREND_TAKE_PROFIT*100:.1f}%")
        print(f"[INIT] CHOP:  ADX<{self.ADX_CHOP_THRESHOLD}, SL={self.CHOP_STOP_LOSS*100:.1f}%, TP={self.CHOP_TAKE_PROFIT*100:.1f}%")

    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def download_data(self) -> pd.DataFrame:
        """Download historical 1-minute data from Yahoo Finance."""
        print(f"[DATA] Downloading {self.days} days of 1-minute data for {self.symbol}...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)

        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1m",
                prepost=False
            )

            if df.empty:
                raise ValueError(f"No data returned for {self.symbol}")

            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")

            print(f"[DATA] Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to download data: {e}")
            raise

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)."""
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        print("[CALC] Calculating indicators...")

        # EMAs
        df["ema_fast"] = df["Close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=self.ema_slow, adjust=False).mean()

        # VWAP (resets daily)
        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["tp_volume"] = df["typical_price"] * df["Volume"]
        df["date"] = df.index.date
        df["cumulative_tp_vol"] = df.groupby("date")["tp_volume"].cumsum()
        df["cumulative_vol"] = df.groupby("date")["Volume"].cumsum()
        df["vwap"] = df["cumulative_tp_vol"] / df["cumulative_vol"]

        # Volume
        df["vol_sma"] = df["Volume"].rolling(window=20).mean()
        df["vol_spike"] = df["Volume"] / df["vol_sma"]

        # ADX
        df["adx"] = self.calculate_adx(df, period=self.ADX_PERIOD)

        # RSI
        df["rsi"] = self.calculate_rsi(df["Close"], period=self.RSI_PERIOD)

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self.calculate_bollinger_bands(
            df["Close"], period=self.BB_PERIOD, std_dev=self.BB_STD_DEV
        )

        # Regime detection
        df["regime"] = df["adx"].apply(self._determine_regime)

        # Clean up
        df.drop(columns=["typical_price", "tp_volume", "cumulative_tp_vol",
                         "cumulative_vol", "date"], inplace=True)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # Report stats
        regime_counts = df["regime"].value_counts()
        print(f"[CALC] Regime Distribution:")
        for regime, count in regime_counts.items():
            pct = count / len(df) * 100
            print(f"       {regime}: {count} candles ({pct:.1f}%)")

        print(f"[CALC] Indicators calculated. {len(df)} valid candles.")
        return df

    def _determine_regime(self, adx: float) -> str:
        """Determine market regime based on ADX."""
        if adx >= self.ADX_TREND_THRESHOLD:
            return "TREND"
        elif adx < self.ADX_CHOP_THRESHOLD:
            return "CHOP"
        else:
            return "BUFFER"

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """Check if the timestamp is within allowed trading hours."""
        time_str = timestamp.strftime("%H:%M")

        if time_str < self.morning_start:
            return False
        if self.lunch_start <= time_str <= self.lunch_end:
            return False
        if time_str >= self.day_end:
            return False

        return True

    def check_trend_entry(self, row: pd.Series) -> bool:
        """Check TREND mode entry: Price > VWAP AND EMA9 > EMA21 AND Volume Spike."""
        price_above_vwap = row["Close"] > row["vwap"]
        ema_bullish = row["ema_fast"] > row["ema_slow"]
        volume_spike = row["vol_spike"] >= self.vol_spike_threshold
        min_volume = row["Volume"] >= self.min_volume

        return price_above_vwap and ema_bullish and volume_spike and min_volume

    def check_chop_entry(self, row: pd.Series) -> bool:
        """Check CHOP mode entry: RSI < 30 AND Price < Lower Bollinger Band."""
        rsi_oversold = row["rsi"] < self.RSI_OVERSOLD
        below_lower_bb = row["Close"] < row["bb_lower"]

        return rsi_oversold and below_lower_bb

    def check_entry_signal(self, row: pd.Series) -> tuple[bool, str]:
        """
        Check if entry conditions are met based on current regime.

        Returns:
            Tuple of (should_enter, regime)
        """
        if self.position is not None:
            return False, ""

        if not self.is_trading_time(row.name):
            return False, ""

        regime = row["regime"]

        # BUFFER zone - no trading
        if regime == "BUFFER":
            self.buffer_blocked += 1
            return False, ""

        # TREND mode (only if enabled)
        if regime == "TREND" and self.mode in ["hybrid", "trend_only"]:
            if self.check_trend_entry(row):
                self.trend_signals += 1
                return True, "TREND"

        # CHOP mode (only if enabled)
        if regime == "CHOP" and self.mode in ["hybrid", "chop_only"]:
            if self.check_chop_entry(row):
                self.chop_signals += 1
                return True, "CHOP"

        return False, ""

    def check_exit_signal(self, row: pd.Series) -> tuple[bool, str]:
        """Check if exit conditions are met based on entry regime."""
        if self.position is None:
            return False, ""

        entry_price = self.position["entry_price"]
        entry_regime = self.position["regime"]
        current_price = row["Close"]

        pnl_pct = (current_price - entry_price) / entry_price

        # Get regime-specific stops
        if entry_regime == "TREND":
            stop_loss = self.TREND_STOP_LOSS
            take_profit = self.TREND_TAKE_PROFIT
        else:  # CHOP
            stop_loss = self.CHOP_STOP_LOSS
            take_profit = self.CHOP_TAKE_PROFIT

        # Stop Loss
        if pnl_pct <= -stop_loss:
            return True, "STOP_LOSS"

        # Take Profit
        if pnl_pct >= take_profit:
            return True, "TAKE_PROFIT"

        # Regime-specific exits
        if entry_regime == "TREND":
            # Trend broken: Price < EMA9
            if current_price < row["ema_fast"]:
                return True, "TREND_BROKEN"
        else:  # CHOP
            # Mean reversion target: RSI > 50 and in profit
            if row["rsi"] > 50 and pnl_pct > 0:
                return True, "MEAN_REVERSION_TARGET"

        # End of day
        if row.name.strftime("%H:%M") >= self.day_end:
            return True, "EOD_EXIT"

        return False, ""

    def simulate(self, df: pd.DataFrame) -> list[Trade]:
        """Run the backtest simulation."""
        print("[SIM] Running backtest simulation...")

        for idx, row in df.iterrows():
            # Check for exit first
            if self.position is not None:
                should_exit, reason = self.check_exit_signal(row)
                if should_exit:
                    self._close_trade(row, reason)

            # Check for entry
            if self.position is None:
                should_enter, regime = self.check_entry_signal(row)
                if should_enter:
                    self._open_trade(row, regime)

        # Force close any open position
        if self.position is not None:
            self._close_trade(df.iloc[-1], "BACKTEST_END")

        print(f"[SIM] Simulation complete. {len(self.trades)} trades executed.")
        print(f"[SIM] TREND signals: {self.trend_signals}, CHOP signals: {self.chop_signals}, BUFFER blocked: {self.buffer_blocked}")
        return self.trades

    def _open_trade(self, row: pd.Series, regime: str):
        """Open a new position."""
        position_size = self.FULL_POSITION if regime == "TREND" else self.HALF_POSITION

        self.position = {
            "entry_price": row["Close"],
            "entry_time": row.name,
            "regime": regime,
            "position_size": position_size,
            "entry_adx": row["adx"],
            "entry_rsi": row["rsi"]
        }

    def _close_trade(self, row: pd.Series, reason: str):
        """Close the current position and record the trade."""
        if self.position is None:
            return

        exit_price = row["Close"]
        entry_price = self.position["entry_price"]
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_dollars = pnl_pct * self.position["position_size"]

        trade = Trade(
            entry_time=self.position["entry_time"],
            entry_price=entry_price,
            exit_time=row.name,
            exit_price=exit_price,
            pnl_pct=pnl_pct,
            pnl_dollars=pnl_dollars,
            exit_reason=reason,
            regime=self.position["regime"],
            position_size="FULL" if self.position["position_size"] == self.FULL_POSITION else "HALF",
            entry_adx=self.position["entry_adx"],
            entry_rsi=self.position["entry_rsi"]
        )

        self.trades.append(trade)
        self.position = None

    def generate_report(self) -> dict:
        """Generate a summary report of the backtest results."""
        if not self.trades:
            return {
                "symbol": self.symbol,
                "mode": self.mode,
                "total_trades": 0,
                "message": "No trades executed"
            }

        # Convert to DataFrame for analysis
        trades_data = [{
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "pnl_pct": t.pnl_pct,
            "pnl_dollars": t.pnl_dollars,
            "exit_reason": t.exit_reason,
            "regime": t.regime,
            "position_size": t.position_size,
            "entry_adx": t.entry_adx,
            "entry_rsi": t.entry_rsi
        } for t in self.trades]
        df_trades = pd.DataFrame(trades_data)

        # Overall stats
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades["pnl_pct"] > 0])
        losing_trades = len(df_trades[df_trades["pnl_pct"] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        total_pnl_dollars = df_trades["pnl_dollars"].sum()
        total_pnl_pct = (total_pnl_dollars / self.FULL_POSITION) * 100

        # Regime breakdown
        trend_trades = df_trades[df_trades["regime"] == "TREND"]
        chop_trades = df_trades[df_trades["regime"] == "CHOP"]

        trend_pnl = trend_trades["pnl_dollars"].sum() if len(trend_trades) > 0 else 0
        chop_pnl = chop_trades["pnl_dollars"].sum() if len(chop_trades) > 0 else 0

        trend_win_rate = (len(trend_trades[trend_trades["pnl_pct"] > 0]) / len(trend_trades) * 100) if len(trend_trades) > 0 else 0
        chop_win_rate = (len(chop_trades[chop_trades["pnl_pct"] > 0]) / len(chop_trades) * 100) if len(chop_trades) > 0 else 0

        # Drawdown
        cumulative_pnl = df_trades["pnl_dollars"].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min()

        # Exit reason breakdown
        exit_reasons = df_trades["exit_reason"].value_counts().to_dict()

        report = {
            "symbol": self.symbol,
            "mode": self.mode,
            "period_days": self.days,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_dollars": round(total_pnl_dollars, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "max_drawdown_dollars": round(max_drawdown, 2),
            "trend_trades": len(trend_trades),
            "trend_pnl": round(trend_pnl, 2),
            "trend_win_rate": round(trend_win_rate, 1),
            "chop_trades": len(chop_trades),
            "chop_pnl": round(chop_pnl, 2),
            "chop_win_rate": round(chop_win_rate, 1),
            "buffer_blocked": self.buffer_blocked,
            "exit_reasons": exit_reasons
        }

        return report

    def print_report(self, report: dict):
        """Print a formatted report to the console."""
        print("\n" + "=" * 70)
        print(f"  HYBRID REGIME BACKTEST - {report['symbol']} ({report['mode'].upper()})")
        print("=" * 70)

        if report.get("total_trades", 0) == 0:
            print("  No trades executed during backtest period.")
            print("=" * 70)
            return

        print(f"  Period:         {report['period_days']} days")
        print(f"  Total Trades:   {report['total_trades']}")
        print(f"  Win Rate:       {report['win_rate_pct']}%")
        print("-" * 70)
        print(f"  Total PnL:      ${report['total_pnl_dollars']:+,.2f} ({report['total_pnl_pct']:+.2f}%)")
        print(f"  Max Drawdown:   ${report['max_drawdown_dollars']:.2f}")
        print("-" * 70)
        print("  REGIME BREAKDOWN:")
        print(f"    TREND:  {report['trend_trades']} trades | PnL: ${report['trend_pnl']:+,.2f} | WR: {report['trend_win_rate']:.1f}%")
        print(f"    CHOP:   {report['chop_trades']} trades | PnL: ${report['chop_pnl']:+,.2f} | WR: {report['chop_win_rate']:.1f}%")
        print(f"    BUFFER: {report['buffer_blocked']} signals blocked (ADX 20-25)")
        print("-" * 70)
        print("  EXIT REASONS:")
        for reason, count in report.get("exit_reasons", {}).items():
            print(f"    {reason}: {count}")
        print("=" * 70)

    def print_trade_log(self):
        """Print detailed trade log."""
        if not self.trades:
            print("\n[LOG] No trades to display.")
            return

        print("\n" + "-" * 100)
        print("  TRADE LOG")
        print("-" * 100)
        print(f"{'#':<4} {'Entry Time':<18} {'Regime':<8} {'Size':<6} {'Entry $':<10} {'Exit $':<10} {'PnL $':<12} {'Reason':<20}")
        print("-" * 100)

        for i, trade in enumerate(self.trades, 1):
            entry_time = trade.entry_time.strftime("%m/%d %H:%M")
            print(f"{i:<4} {entry_time:<18} {trade.regime:<8} {trade.position_size:<6} "
                  f"${trade.entry_price:<9.2f} ${trade.exit_price:<9.2f} "
                  f"${trade.pnl_dollars:+10.2f} {trade.exit_reason:<20}")

        print("-" * 100)

    def run(self, verbose: bool = False) -> dict:
        """Execute the full backtest pipeline."""
        df = self.download_data()
        df = self.calculate_indicators(df)
        self.simulate(df)
        report = self.generate_report()
        self.print_report(report)

        if verbose:
            self.print_trade_log()

        return report


def run_comparison(symbol: str, days: int, verbose: bool = False):
    """Run comparison between Hybrid, Trend-only, and Chop-only."""
    print("\n" + "=" * 70)
    print("  HYBRID vs TREND-ONLY vs CHOP-ONLY COMPARISON")
    print("=" * 70)

    # Test 1: HYBRID (both regimes)
    print("\n" + "-" * 70)
    print("  TEST 1: HYBRID (TREND + CHOP)")
    print("-" * 70)
    bt_hybrid = BacktestHybrid(symbol=symbol, days=days, mode="hybrid")
    report_hybrid = bt_hybrid.run(verbose=verbose)

    # Test 2: TREND only
    print("\n" + "-" * 70)
    print("  TEST 2: TREND ONLY (ADX > 25)")
    print("-" * 70)
    bt_trend = BacktestHybrid(symbol=symbol, days=days, mode="trend_only")
    report_trend = bt_trend.run(verbose=verbose)

    # Test 3: CHOP only
    print("\n" + "-" * 70)
    print("  TEST 3: CHOP ONLY (ADX < 20)")
    print("-" * 70)
    bt_chop = BacktestHybrid(symbol=symbol, days=days, mode="chop_only")
    report_chop = bt_chop.run(verbose=verbose)

    # Comparison Summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<20} {'HYBRID':<18} {'TREND':<18} {'CHOP':<18}")
    print("-" * 70)

    metrics = [
        ("Total Trades", "total_trades"),
        ("Win Rate (%)", "win_rate_pct"),
        ("Total PnL ($)", "total_pnl_dollars"),
        ("Max Drawdown ($)", "max_drawdown_dollars"),
    ]

    for label, key in metrics:
        val_h = report_hybrid.get(key, 0)
        val_t = report_trend.get(key, 0)
        val_c = report_chop.get(key, 0)

        if "dollar" in key.lower() or "drawdown" in key.lower():
            print(f"  {label:<20} ${val_h:>15,.2f} ${val_t:>15,.2f} ${val_c:>15,.2f}")
        else:
            print(f"  {label:<20} {val_h:>16} {val_t:>16} {val_c:>16}")

    print("=" * 70)

    # Verdict
    hybrid_pnl = report_hybrid.get("total_pnl_dollars", 0)
    trend_pnl = report_trend.get("total_pnl_dollars", 0)
    chop_pnl = report_chop.get("total_pnl_dollars", 0)

    print("\n  VERDICT:")
    best_pnl = max(hybrid_pnl, trend_pnl, chop_pnl)
    if hybrid_pnl == best_pnl:
        print(f"  [WIN] HYBRID is best! PnL: ${hybrid_pnl:+,.2f}")
        print(f"        Trend contributed: ${report_hybrid.get('trend_pnl', 0):+,.2f}")
        print(f"        Chop contributed:  ${report_hybrid.get('chop_pnl', 0):+,.2f}")
    elif trend_pnl == best_pnl:
        print(f"  [TREND] Trend-only is best! PnL: ${trend_pnl:+,.2f}")
    else:
        print(f"  [CHOP] Chop-only is best! PnL: ${chop_pnl:+,.2f}")

    print("=" * 70)

    return report_hybrid, report_trend, report_chop


def main():
    """Entry point for the backtester."""
    parser = argparse.ArgumentParser(
        description="Backtest the Hybrid Regime strategy"
    )
    parser.add_argument(
        "--symbol", "-s",
        type=str,
        default="TQQQ",
        help="Ticker symbol to backtest (default: TQQQ)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days of historical data (default: 7)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed trade log"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        choices=["hybrid", "trend_only", "chop_only"],
        default="hybrid",
        help="Strategy mode (default: hybrid)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison: HYBRID vs TREND vs CHOP"
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  HYBRID REGIME BACKTESTER v1")
    print("  Bi-Cameral Trading Bot")
    print("  TREND (ADX>25) + CHOP (ADX<20) + BUFFER (20-25)")
    print("=" * 70 + "\n")

    if args.compare:
        run_comparison(args.symbol, args.days, args.verbose)
    else:
        backtester = BacktestHybrid(
            symbol=args.symbol,
            days=args.days,
            mode=args.mode
        )
        report = backtester.run(verbose=args.verbose)
        print(f"\n>>> Total PnL for {args.symbol}: ${report.get('total_pnl_dollars', 0):+,.2f}")


if __name__ == "__main__":
    main()
