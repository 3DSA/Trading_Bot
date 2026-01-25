#!/usr/bin/env python3
"""
Backtest Scalper - Momentum Scalping Strategy Validator

Tests the EMA 9/21 + VWAP + Volume Spike + ADX strategy on historical data.
Used to validate the "Switch Hitter" approach before going live.

Features:
- ADX Filter: Only trade when trend strength > 25 (avoid chop)
- Optimized Risk: Wider stops for 3x leveraged ETFs

Usage:
    python backtest_scalper.py --symbol TQQQ
    python backtest_scalper.py --symbol SQQQ --days 10
    python backtest_scalper.py --symbol TQQQ --no-adx  # Compare without ADX

Author: Bi-Cameral System
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Configuration
BASE_DIR = Path(__file__).parent.parent  # Go up from tests/ to project root
CONFIG_PATH = BASE_DIR / "config.json"


class BacktestScalper:
    """
    Backtester for the momentum scalping strategy.
    Simulates trades using historical 1-minute data.

    v3 Update: Double-Lock filtering to combat leveraged noise inflation.
    - TQQQ ADX is artificially inflated by 3x leverage
    - QQQ ADX provides true underlying trend confirmation
    """

    # Optimized risk parameters for 3x leveraged ETFs
    OPTIMIZED_STOP_LOSS = 0.01      # 1% (was 0.5% - too tight for TQQQ volatility)
    OPTIMIZED_TAKE_PROFIT = 0.02   # 2% (was 1.5% - need bigger wins to offset losses)
    OPTIMIZED_TRAILING = 0.005     # 0.5% trailing (was 0.3%)

    # ADX settings - Original (single lock)
    ADX_PERIOD = 14
    ADX_THRESHOLD = 25  # Only trade when ADX > 25 (strong trend)

    # Double-Lock ADX settings (combat leveraged noise)
    DOUBLE_LOCK_LEVERAGED_ADX = 30  # TQQQ/SQQQ must show ADX > 30 (stricter)
    DOUBLE_LOCK_UNDERLYING_ADX = 25  # QQQ must confirm with ADX > 25
    UNDERLYING_SYMBOL = "QQQ"  # The non-leveraged ETF for confirmation

    def __init__(self, symbol: str, days: int = 7, use_adx: bool = True,
                 use_optimized_risk: bool = True, use_double_lock: bool = False):
        """
        Initialize the backtester.

        Args:
            symbol: The ticker symbol to backtest (TQQQ, SQQQ, etc.)
            days: Number of days of historical data to use
            use_adx: If True, require ADX > 25 for entries
            use_optimized_risk: If True, use wider SL/TP for leveraged ETFs
            use_double_lock: If True, require BOTH leveraged ADX > 30 AND QQQ ADX > 25
        """
        self.symbol = symbol
        self.days = days
        self.use_adx = use_adx
        self.use_optimized_risk = use_optimized_risk
        self.use_double_lock = use_double_lock

        # Load strategy config
        self.config = self._load_config()
        self.strategy = self.config.get("strategy", {})
        self.risk = self.config.get("risk_management", {})

        # Strategy parameters
        self.ema_fast = self.strategy.get("ema_fast", 9)
        self.ema_slow = self.strategy.get("ema_slow", 21)
        self.vol_spike_threshold = self.strategy.get("vol_spike_threshold", 2.0)
        self.min_volume = self.strategy.get("min_volume", 1000)

        # Risk parameters - use optimized or config values
        if use_optimized_risk:
            self.stop_loss_pct = self.OPTIMIZED_STOP_LOSS
            self.take_profit_pct = self.OPTIMIZED_TAKE_PROFIT
            self.trailing_stop_pct = self.OPTIMIZED_TRAILING
        else:
            self.stop_loss_pct = self.risk.get("stop_loss_pct", 0.005)
            self.take_profit_pct = self.risk.get("take_profit_pct", 0.015)
            self.trailing_stop_pct = self.risk.get("trailing_stop_pct", 0.003)

        self.use_trailing_stop = self.risk.get("use_trailing_stop", True)

        # Time filters
        time_filters = self.strategy.get("time_filters", {})
        self.morning_start = time_filters.get("morning_start", "09:30")
        self.lunch_start = time_filters.get("lunch_start", "11:45")
        self.lunch_end = time_filters.get("lunch_end", "13:15")
        self.day_end = time_filters.get("day_end", "15:55")

        # Trade tracking
        self.trades = []
        self.position = None  # {"entry_price", "entry_time", "high_water_mark"}

        # Stats for ADX filtering
        self.signals_blocked_by_adx = 0
        self.signals_blocked_by_double_lock = 0

        # QQQ data (populated if using double-lock)
        self.qqq_df = None

        print(f"[INIT] BacktestScalper for {symbol}")
        print(f"[INIT] Strategy: EMA {self.ema_fast}/{self.ema_slow}, "
              f"TP {self.take_profit_pct*100:.1f}%, SL {self.stop_loss_pct*100:.1f}%")
        if use_double_lock:
            print(f"[INIT] DOUBLE-LOCK: {symbol} ADX > {self.DOUBLE_LOCK_LEVERAGED_ADX} AND QQQ ADX > {self.DOUBLE_LOCK_UNDERLYING_ADX}")
        elif use_adx:
            print(f"[INIT] ADX Filter: ENABLED (>{self.ADX_THRESHOLD})")
        else:
            print(f"[INIT] ADX Filter: DISABLED")

    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "strategy": {
                    "ema_fast": 9,
                    "ema_slow": 21,
                    "vol_spike_threshold": 2.0,
                    "min_volume": 1000,
                    "time_filters": {
                        "morning_start": "09:30",
                        "lunch_start": "11:45",
                        "lunch_end": "13:15",
                        "day_end": "15:55"
                    }
                },
                "risk_management": {
                    "stop_loss_pct": 0.005,
                    "take_profit_pct": 0.015,
                    "use_trailing_stop": True,
                    "trailing_stop_pct": 0.003
                }
            }

    def download_data(self) -> pd.DataFrame:
        """
        Download historical 1-minute data from Yahoo Finance.
        If using double-lock, also downloads QQQ data for confirmation.

        Returns:
            DataFrame with OHLCV data
        """
        print(f"[DATA] Downloading {self.days} days of 1-minute data for {self.symbol}...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)

        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval="1m",
                prepost=False  # Regular trading hours only
            )

            if df.empty:
                raise ValueError(f"No data returned for {self.symbol}")

            # Ensure timezone-aware index is converted to local time
            if df.index.tz is not None:
                df.index = df.index.tz_convert("America/New_York")

            print(f"[DATA] Downloaded {len(df)} candles from {df.index[0]} to {df.index[-1]}")

            # If using double-lock, also download QQQ for underlying ADX confirmation
            if self.use_double_lock:
                print(f"[DATA] Double-Lock enabled - downloading {self.UNDERLYING_SYMBOL} for confirmation...")
                qqq_ticker = yf.Ticker(self.UNDERLYING_SYMBOL)
                qqq_df = qqq_ticker.history(
                    start=start_date,
                    end=end_date,
                    interval="1m",
                    prepost=False
                )

                if qqq_df.empty:
                    raise ValueError(f"No data returned for {self.UNDERLYING_SYMBOL}")

                if qqq_df.index.tz is not None:
                    qqq_df.index = qqq_df.index.tz_convert("America/New_York")

                # Calculate ADX for QQQ
                qqq_df["qqq_adx"] = self.calculate_adx(qqq_df, period=self.ADX_PERIOD)
                self.qqq_df = qqq_df
                print(f"[DATA] QQQ: {len(qqq_df)} candles, Avg ADX: {qqq_df['qqq_adx'].mean():.1f}")

            return df

        except Exception as e:
            print(f"[ERROR] Failed to download data: {e}")
            raise

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction:
        - ADX < 20: Weak trend / Chop (avoid trading)
        - ADX 20-25: Developing trend
        - ADX 25-50: Strong trend (ideal for momentum trading)
        - ADX > 50: Very strong trend

        Args:
            df: DataFrame with High, Low, Close columns
            period: ADX period (default 14)

        Returns:
            Series with ADX values
        """
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate +DM and -DM (Directional Movement)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)

        # Smoothed TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)

        # Calculate DX (Directional Index)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

        # Calculate ADX (smoothed DX)
        adx = dx.ewm(alpha=1/period, adjust=False).mean()

        return adx

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators needed for the strategy.

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with added indicator columns
        """
        print("[CALC] Calculating indicators...")

        # EMA Fast and Slow
        df["ema_fast"] = df["Close"].ewm(span=self.ema_fast, adjust=False).mean()
        df["ema_slow"] = df["Close"].ewm(span=self.ema_slow, adjust=False).mean()

        # VWAP (resets daily)
        df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3
        df["tp_volume"] = df["typical_price"] * df["Volume"]

        # Group by date for daily VWAP reset
        df["date"] = df.index.date
        df["cumulative_tp_vol"] = df.groupby("date")["tp_volume"].cumsum()
        df["cumulative_vol"] = df.groupby("date")["Volume"].cumsum()
        df["vwap"] = df["cumulative_tp_vol"] / df["cumulative_vol"]

        # Volume SMA (20-period)
        df["vol_sma"] = df["Volume"].rolling(window=20).mean()
        df["vol_spike"] = df["Volume"] / df["vol_sma"]

        # ADX (Average Directional Index) - Trend Strength Filter
        df["adx"] = self.calculate_adx(df, period=self.ADX_PERIOD)

        # Time filters
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        df["time_str"] = df.index.strftime("%H:%M")

        # Clean up
        df.drop(columns=["typical_price", "tp_volume", "cumulative_tp_vol",
                         "cumulative_vol", "date"], inplace=True)

        # Forward fill any NaN values from warmup period
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # Report ADX stats
        avg_adx = df["adx"].mean()
        high_adx_pct = (df["adx"] > self.ADX_THRESHOLD).mean() * 100
        print(f"[CALC] ADX Stats: Avg={avg_adx:.1f}, "
              f">{self.ADX_THRESHOLD}={high_adx_pct:.1f}% of candles")
        print(f"[CALC] Indicators calculated. {len(df)} valid candles.")
        return df

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if the timestamp is within allowed trading hours.
        Excludes lunch hour and end-of-day.

        Args:
            timestamp: The candle timestamp

        Returns:
            True if trading is allowed
        """
        time_str = timestamp.strftime("%H:%M")

        # Before market open
        if time_str < self.morning_start:
            return False

        # Lunch hour (avoid chop)
        if self.lunch_start <= time_str <= self.lunch_end:
            return False

        # End of day (avoid closing auction volatility)
        if time_str >= self.day_end:
            return False

        return True

    def check_entry_signal(self, row: pd.Series) -> bool:
        """
        Check if entry conditions are met.

        Entry Criteria:
        1. Price > VWAP (bullish bias)
        2. EMA Fast > EMA Slow (uptrend)
        3. Volume > 2x SMA (momentum confirmation)
        4. ADX > 25 (strong trend - avoid chop)
        5. Within trading hours
        6. [DOUBLE-LOCK] TQQQ ADX > 30 AND QQQ ADX > 25

        Args:
            row: A single candle row from the DataFrame

        Returns:
            True if all entry conditions are met
        """
        # Already in position
        if self.position is not None:
            return False

        # Time filter
        if not self.is_trading_time(row.name):
            return False

        # Volume filter
        if row["Volume"] < self.min_volume:
            return False

        # Entry conditions (original)
        price_above_vwap = row["Close"] > row["vwap"]
        ema_bullish = row["ema_fast"] > row["ema_slow"]
        volume_spike = row["vol_spike"] >= self.vol_spike_threshold

        # Original signal without ADX
        base_signal = price_above_vwap and ema_bullish and volume_spike

        # DOUBLE-LOCK: Stricter filtering for leveraged ETFs
        if self.use_double_lock:
            # Leveraged ETF must show ADX > 30 (stricter threshold)
            leveraged_adx_strong = row["adx"] > self.DOUBLE_LOCK_LEVERAGED_ADX

            # Get QQQ ADX for this timestamp (underlying confirmation)
            qqq_adx = self._get_qqq_adx(row.name)
            qqq_adx_confirms = qqq_adx > self.DOUBLE_LOCK_UNDERLYING_ADX

            # Both must confirm for entry
            double_lock_confirmed = leveraged_adx_strong and qqq_adx_confirms

            if base_signal and not double_lock_confirmed:
                self.signals_blocked_by_double_lock += 1
                if leveraged_adx_strong and not qqq_adx_confirms:
                    pass  # QQQ didn't confirm - false signal blocked!
            return base_signal and double_lock_confirmed

        # Single ADX Filter - Only trade in strong trends
        elif self.use_adx:
            adx_strong = row["adx"] > self.ADX_THRESHOLD
            if base_signal and not adx_strong:
                self.signals_blocked_by_adx += 1
            return base_signal and adx_strong
        else:
            return base_signal

    def _get_qqq_adx(self, timestamp: pd.Timestamp) -> float:
        """
        Get QQQ ADX value for a given timestamp.
        Uses exact match or nearest previous timestamp.

        Args:
            timestamp: The timestamp to look up

        Returns:
            QQQ ADX value, or 0 if not found
        """
        if self.qqq_df is None:
            return 0

        try:
            # Try exact match first
            if timestamp in self.qqq_df.index:
                return self.qqq_df.loc[timestamp, "qqq_adx"]

            # Find nearest previous timestamp
            mask = self.qqq_df.index <= timestamp
            if mask.any():
                nearest_idx = self.qqq_df.index[mask][-1]
                return self.qqq_df.loc[nearest_idx, "qqq_adx"]

            return 0
        except Exception:
            return 0

    def check_exit_signal(self, row: pd.Series) -> tuple[bool, str]:
        """
        Check if exit conditions are met.

        Exit Criteria:
        1. Take Profit: Price >= Entry + TP%
        2. Stop Loss: Price <= Entry - SL%
        3. Trailing Stop: Price <= High Water Mark - Trail%
        4. Trend Break: EMA Fast < EMA Slow
        5. ADX Collapse: ADX < 20 (trend dying) [NEW]

        Args:
            row: A single candle row from the DataFrame

        Returns:
            Tuple of (should_exit, reason)
        """
        if self.position is None:
            return False, ""

        entry_price = self.position["entry_price"]
        high_water_mark = self.position["high_water_mark"]
        current_price = row["Close"]

        # Update high water mark
        if current_price > high_water_mark:
            self.position["high_water_mark"] = current_price
            high_water_mark = current_price

        # Calculate returns
        pnl_pct = (current_price - entry_price) / entry_price

        # Take Profit
        if pnl_pct >= self.take_profit_pct:
            return True, "TAKE_PROFIT"

        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return True, "STOP_LOSS"

        # Trailing Stop (only if in profit)
        if self.use_trailing_stop and pnl_pct > 0:
            trail_pct = (high_water_mark - current_price) / high_water_mark
            if trail_pct >= self.trailing_stop_pct:
                return True, "TRAILING_STOP"

        # ADX Collapse - Exit if trend is dying (only if using ADX)
        if self.use_adx and row["adx"] < 20:
            return True, "ADX_COLLAPSE"

        # Trend Break (EMA crossover to bearish)
        if row["ema_fast"] < row["ema_slow"]:
            return True, "TREND_BREAK"

        # End of day forced exit
        if row.name.strftime("%H:%M") >= self.day_end:
            return True, "EOD_EXIT"

        return False, ""

    def simulate(self, df: pd.DataFrame) -> list[dict]:
        """
        Run the backtest simulation.

        Args:
            df: DataFrame with OHLCV and indicators

        Returns:
            List of trade dictionaries
        """
        print("[SIM] Running backtest simulation...")

        for idx, row in df.iterrows():
            # Check for exit first (if in position)
            if self.position is not None:
                should_exit, reason = self.check_exit_signal(row)
                if should_exit:
                    self._close_trade(row, reason)

            # Check for entry (if not in position)
            if self.position is None:
                if self.check_entry_signal(row):
                    self._open_trade(row)

        # Force close any open position at end
        if self.position is not None:
            self._close_trade(df.iloc[-1], "BACKTEST_END")

        print(f"[SIM] Simulation complete. {len(self.trades)} trades executed.")
        if self.use_adx:
            print(f"[SIM] Signals blocked by ADX filter: {self.signals_blocked_by_adx}")
        return self.trades

    def _open_trade(self, row: pd.Series):
        """Open a new position."""
        self.position = {
            "entry_price": row["Close"],
            "entry_time": row.name,
            "high_water_mark": row["Close"],
            "entry_adx": row["adx"]
        }

    def _close_trade(self, row: pd.Series, reason: str):
        """Close the current position and record the trade."""
        if self.position is None:
            return

        exit_price = row["Close"]
        entry_price = self.position["entry_price"]
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_dollars = pnl_pct * 10000  # Assuming $10,000 position size

        trade = {
            "entry_time": self.position["entry_time"],
            "entry_price": entry_price,
            "exit_time": row.name,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "pnl_dollars": pnl_dollars,
            "exit_reason": reason,
            "duration_minutes": (row.name - self.position["entry_time"]).total_seconds() / 60,
            "entry_adx": self.position.get("entry_adx", 0)
        }

        self.trades.append(trade)
        self.position = None

    def generate_report(self) -> dict:
        """
        Generate a summary report of the backtest results.

        Returns:
            Dictionary with performance metrics
        """
        if not self.trades:
            return {
                "symbol": self.symbol,
                "total_trades": 0,
                "message": "No trades executed",
                "use_adx": self.use_adx,
                "signals_blocked": self.signals_blocked_by_adx
            }

        df_trades = pd.DataFrame(self.trades)

        # Basic stats
        total_trades = len(df_trades)
        winning_trades = len(df_trades[df_trades["pnl_pct"] > 0])
        losing_trades = len(df_trades[df_trades["pnl_pct"] <= 0])
        win_rate = winning_trades / total_trades * 100

        # PnL stats
        total_pnl_pct = df_trades["pnl_pct"].sum() * 100
        total_pnl_dollars = df_trades["pnl_dollars"].sum()
        avg_pnl_pct = df_trades["pnl_pct"].mean() * 100
        avg_win = df_trades[df_trades["pnl_pct"] > 0]["pnl_pct"].mean() * 100 if winning_trades > 0 else 0
        avg_loss = df_trades[df_trades["pnl_pct"] <= 0]["pnl_pct"].mean() * 100 if losing_trades > 0 else 0

        # Profit factor
        gross_profit = df_trades[df_trades["pnl_dollars"] > 0]["pnl_dollars"].sum()
        gross_loss = abs(df_trades[df_trades["pnl_dollars"] < 0]["pnl_dollars"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Drawdown calculation
        cumulative_pnl = df_trades["pnl_dollars"].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - rolling_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / 10000) * 100 if max_drawdown < 0 else 0

        # Exit reason breakdown
        exit_reasons = df_trades["exit_reason"].value_counts().to_dict()

        # Duration stats
        avg_duration = df_trades["duration_minutes"].mean()

        # ADX stats
        avg_entry_adx = df_trades["entry_adx"].mean() if "entry_adx" in df_trades.columns else 0

        report = {
            "symbol": self.symbol,
            "period_days": self.days,
            "use_adx": self.use_adx,
            "use_double_lock": self.use_double_lock,
            "signals_blocked": self.signals_blocked_by_adx,
            "signals_blocked_double_lock": self.signals_blocked_by_double_lock,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate_pct": round(win_rate, 1),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "total_pnl_dollars": round(total_pnl_dollars, 2),
            "avg_pnl_pct": round(avg_pnl_pct, 3),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_dollars": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "avg_duration_minutes": round(avg_duration, 1),
            "avg_entry_adx": round(avg_entry_adx, 1),
            "exit_reasons": exit_reasons,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct
        }

        return report

    def print_report(self, report: dict, label: str = ""):
        """Print a formatted report to the console."""
        header = f"  BACKTEST REPORT - {report['symbol']}"
        if label:
            header += f" ({label})"

        print("\n" + "=" * 60)
        print(header)
        print("=" * 60)

        if report.get("total_trades", 0) == 0:
            print("  No trades executed during backtest period.")
            if report.get("signals_blocked", 0) > 0:
                print(f"  Signals blocked by ADX: {report['signals_blocked']}")
            print("=" * 60)
            return

        if report.get("use_double_lock"):
            adx_status = f"DOUBLE-LOCK (TQQQ>30 + QQQ>25)"
        elif report.get("use_adx"):
            adx_status = "ON (>25)"
        else:
            adx_status = "OFF"
        print(f"  ADX Filter:     {adx_status}")
        print(f"  Risk:           SL={report['stop_loss_pct']*100:.1f}%, TP={report['take_profit_pct']*100:.1f}%")
        print(f"  Period:         {report['period_days']} days")
        print(f"  Total Trades:   {report['total_trades']}")
        print(f"  Win Rate:       {report['win_rate_pct']}%")
        print("-" * 60)
        print(f"  Total PnL:      ${report['total_pnl_dollars']:+.2f} ({report['total_pnl_pct']:+.2f}%)")
        print(f"  Avg PnL/Trade:  {report['avg_pnl_pct']:+.3f}%")
        print(f"  Avg Win:        {report['avg_win_pct']:+.3f}%")
        print(f"  Avg Loss:       {report['avg_loss_pct']:+.3f}%")
        print(f"  Profit Factor:  {report['profit_factor']:.2f}")
        print("-" * 60)
        print(f"  Max Drawdown:   ${report['max_drawdown_dollars']:.2f} ({report['max_drawdown_pct']:.2f}%)")
        print(f"  Avg Duration:   {report['avg_duration_minutes']:.1f} minutes")
        if report.get("use_double_lock"):
            print(f"  Avg Entry ADX:  {report['avg_entry_adx']:.1f}")
            print(f"  Signals Blocked by Double-Lock: {report.get('signals_blocked_double_lock', 0)}")
        elif report.get("use_adx"):
            print(f"  Avg Entry ADX:  {report['avg_entry_adx']:.1f}")
            print(f"  Signals Blocked:{report['signals_blocked']}")
        print("-" * 60)
        print("  Exit Reasons:")
        for reason, count in report.get("exit_reasons", {}).items():
            print(f"    {reason}: {count}")
        print("=" * 60)

    def print_trade_log(self):
        """Print detailed trade log."""
        if not self.trades:
            print("\n[LOG] No trades to display.")
            return

        print("\n" + "-" * 90)
        print("  TRADE LOG")
        print("-" * 90)
        print(f"{'#':<4} {'Entry Time':<20} {'Entry $':<10} {'Exit $':<10} {'PnL %':<10} {'ADX':<6} {'Reason':<15}")
        print("-" * 90)

        for i, trade in enumerate(self.trades, 1):
            entry_time = trade["entry_time"].strftime("%m/%d %H:%M")
            adx = trade.get("entry_adx", 0)
            print(f"{i:<4} {entry_time:<20} ${trade['entry_price']:<9.2f} "
                  f"${trade['exit_price']:<9.2f} {trade['pnl_pct']*100:+.2f}%     "
                  f"{adx:<6.1f} {trade['exit_reason']:<15}")

        print("-" * 90)

    def run(self, verbose: bool = False) -> dict:
        """
        Execute the full backtest pipeline.

        Args:
            verbose: If True, print detailed trade log

        Returns:
            Report dictionary
        """
        # Download data
        df = self.download_data()

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Run simulation
        self.simulate(df)

        # Generate and print report
        report = self.generate_report()
        label = "ADX+OptRisk" if self.use_adx else "No ADX"
        self.print_report(report, label)

        if verbose:
            self.print_trade_log()

        return report


def run_comparison(symbol: str, days: int, verbose: bool = False):
    """
    Run comparison between strategy WITH and WITHOUT ADX filter.

    Args:
        symbol: Ticker symbol to test
        days: Number of days of data
        verbose: Print trade logs
    """
    print("\n" + "=" * 70)
    print("  ADX FILTER COMPARISON TEST")
    print("  Hypothesis: ADX > 25 filters out 'chop' and improves win rate")
    print("=" * 70)

    # Test 1: WITHOUT ADX (old strategy)
    print("\n" + "-" * 70)
    print("  TEST 1: Original Strategy (No ADX, Tight Stops)")
    print("-" * 70)
    bt_no_adx = BacktestScalper(symbol=symbol, days=days,
                                 use_adx=False, use_optimized_risk=False)
    report_no_adx = bt_no_adx.run(verbose=verbose)

    # Test 2: WITH ADX + Optimized Risk
    print("\n" + "-" * 70)
    print("  TEST 2: Enhanced Strategy (ADX Filter + Optimized Risk)")
    print("-" * 70)
    bt_with_adx = BacktestScalper(symbol=symbol, days=days,
                                   use_adx=True, use_optimized_risk=True)
    report_with_adx = bt_with_adx.run(verbose=verbose)

    # Print Comparison Summary
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'No ADX':<20} {'ADX+OptRisk':<20} {'Delta':<15}")
    print("-" * 70)

    metrics = [
        ("Total Trades", "total_trades", "", 0),
        ("Win Rate", "win_rate_pct", "%", 1),
        ("Total PnL ($)", "total_pnl_dollars", "$", 2),
        ("Total PnL (%)", "total_pnl_pct", "%", 2),
        ("Avg PnL/Trade", "avg_pnl_pct", "%", 3),
        ("Profit Factor", "profit_factor", "", 2),
        ("Max Drawdown ($)", "max_drawdown_dollars", "$", 2),
    ]

    for label, key, suffix, decimals in metrics:
        val1 = report_no_adx.get(key, 0)
        val2 = report_with_adx.get(key, 0)
        delta = val2 - val1

        if suffix == "$":
            col1 = f"${val1:,.{decimals}f}"
            col2 = f"${val2:,.{decimals}f}"
            col3 = f"${delta:+,.{decimals}f}"
        elif suffix == "%":
            col1 = f"{val1:.{decimals}f}%"
            col2 = f"{val2:.{decimals}f}%"
            col3 = f"{delta:+.{decimals}f}%"
        else:
            col1 = f"{val1:.{decimals}f}"
            col2 = f"{val2:.{decimals}f}"
            col3 = f"{delta:+.{decimals}f}"

        print(f"  {label:<25} {col1:<20} {col2:<20} {col3:<15}")

    print("-" * 70)

    # Verdict
    pnl_improvement = report_with_adx.get("total_pnl_dollars", 0) - report_no_adx.get("total_pnl_dollars", 0)
    wr_improvement = report_with_adx.get("win_rate_pct", 0) - report_no_adx.get("win_rate_pct", 0)

    print("\n  VERDICT:")
    if pnl_improvement > 0 and wr_improvement > 0:
        print(f"  [WIN] ADX Filter IMPROVED results! PnL +${pnl_improvement:.2f}, WR +{wr_improvement:.1f}%")
    elif pnl_improvement > 0:
        print(f"  [MIXED] Better PnL (+${pnl_improvement:.2f}) but lower win rate ({wr_improvement:.1f}%)")
    elif wr_improvement > 0:
        print(f"  [MIXED] Better win rate (+{wr_improvement:.1f}%) but lower PnL (${pnl_improvement:.2f})")
    else:
        print(f"  [FAIL] ADX Filter did not improve results in this period")

    print("=" * 70)

    return report_no_adx, report_with_adx


def run_double_lock_comparison(symbol: str, days: int, verbose: bool = False):
    """
    Run comparison between Single ADX and Double-Lock filters.

    Double-Lock Hypothesis: TQQQ's 3x leverage inflates ADX readings,
    causing false trend signals. By requiring QQQ ADX > 25 as confirmation,
    we filter out noise and only trade when the UNDERLYING is trending.

    Args:
        symbol: Ticker symbol to test
        days: Number of days of data
        verbose: Print trade logs
    """
    print("\n" + "=" * 70)
    print("  DOUBLE-LOCK COMPARISON TEST")
    print("  Hypothesis: QQQ ADX confirmation eliminates leveraged noise")
    print("  Problem: TQQQ ADX is inflated 3x, causing false trend signals")
    print("  Solution: Require TQQQ ADX > 30 AND QQQ ADX > 25")
    print("=" * 70)

    # Test 1: Single ADX (current strategy)
    print("\n" + "-" * 70)
    print("  TEST 1: Single ADX Filter (TQQQ ADX > 25)")
    print("-" * 70)
    bt_single_adx = BacktestScalper(symbol=symbol, days=days,
                                     use_adx=True, use_optimized_risk=True,
                                     use_double_lock=False)
    report_single = bt_single_adx.run(verbose=verbose)

    # Test 2: Double-Lock ADX
    print("\n" + "-" * 70)
    print("  TEST 2: Double-Lock Filter (TQQQ ADX > 30 AND QQQ ADX > 25)")
    print("-" * 70)
    bt_double_lock = BacktestScalper(symbol=symbol, days=days,
                                      use_adx=True, use_optimized_risk=True,
                                      use_double_lock=True)
    report_double = bt_double_lock.run(verbose=verbose)

    # Print Comparison Summary
    print("\n" + "=" * 70)
    print("  DOUBLE-LOCK COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Metric':<25} {'Single ADX':<20} {'Double-Lock':<20} {'Delta':<15}")
    print("-" * 70)

    metrics = [
        ("Total Trades", "total_trades", "", 0),
        ("Win Rate", "win_rate_pct", "%", 1),
        ("Total PnL ($)", "total_pnl_dollars", "$", 2),
        ("Total PnL (%)", "total_pnl_pct", "%", 2),
        ("Avg PnL/Trade", "avg_pnl_pct", "%", 3),
        ("Profit Factor", "profit_factor", "", 2),
        ("Max Drawdown ($)", "max_drawdown_dollars", "$", 2),
        ("Signals Blocked", "signals_blocked_double_lock", "", 0),
    ]

    for label, key, suffix, decimals in metrics:
        val1 = report_single.get(key, 0)
        val2 = report_double.get(key, 0)
        delta = val2 - val1

        if suffix == "$":
            col1 = f"${val1:,.{decimals}f}"
            col2 = f"${val2:,.{decimals}f}"
            col3 = f"${delta:+,.{decimals}f}"
        elif suffix == "%":
            col1 = f"{val1:.{decimals}f}%"
            col2 = f"{val2:.{decimals}f}%"
            col3 = f"{delta:+.{decimals}f}%"
        else:
            col1 = f"{val1:.{decimals}f}"
            col2 = f"{val2:.{decimals}f}"
            col3 = f"{delta:+.{decimals}f}"

        print(f"  {label:<25} {col1:<20} {col2:<20} {col3:<15}")

    print("-" * 70)

    # Calculate key stats
    trades_avoided = report_single.get("total_trades", 0) - report_double.get("total_trades", 0)
    pnl_improvement = report_double.get("total_pnl_dollars", 0) - report_single.get("total_pnl_dollars", 0)
    wr_improvement = report_double.get("win_rate_pct", 0) - report_single.get("win_rate_pct", 0)

    print("\n  ANALYSIS:")
    print(f"  Trades Avoided by Double-Lock: {trades_avoided}")
    print(f"  Signals Blocked (QQQ didn't confirm): {report_double.get('signals_blocked_double_lock', 0)}")

    print("\n  VERDICT:")
    if pnl_improvement > 0:
        if wr_improvement >= 0:
            print(f"  [WIN] Double-Lock IMPROVED results!")
            print(f"        PnL: +${pnl_improvement:.2f}")
            print(f"        Win Rate: {wr_improvement:+.1f}%")
            print(f"        Bad trades avoided: {trades_avoided}")
        else:
            print(f"  [MIXED] Better PnL (+${pnl_improvement:.2f}) but lower win rate ({wr_improvement:.1f}%)")
    elif pnl_improvement == 0 and report_double.get("total_trades", 0) == 0:
        print(f"  [CAUTION] Double-Lock blocked ALL trades!")
        print(f"            This suggests the market was in CHOP mode.")
        print(f"            Single ADX took {report_single.get('total_trades', 0)} losing trades.")
        print(f"            Staying out was the right call!")
    else:
        if trades_avoided > 0 and report_single.get("total_pnl_dollars", 0) < 0:
            # Even if double-lock isn't better, if single ADX lost money and double-lock avoided trades, that's good
            single_loss = abs(report_single.get("total_pnl_dollars", 0))
            double_loss = abs(report_double.get("total_pnl_dollars", 0))
            if double_loss < single_loss:
                print(f"  [PARTIAL WIN] Double-Lock reduced losses!")
                print(f"        Single ADX loss: ${single_loss:.2f}")
                print(f"        Double-Lock loss: ${double_loss:.2f}")
                print(f"        Saved: ${single_loss - double_loss:.2f}")
            else:
                print(f"  [FAIL] Double-Lock did not improve results in this period")
        else:
            print(f"  [FAIL] Double-Lock did not improve results in this period")

    print("=" * 70)

    return report_single, report_double


def main():
    """Entry point for the backtester."""
    parser = argparse.ArgumentParser(
        description="Backtest the momentum scalping strategy with ADX filter"
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
        "--no-adx",
        action="store_true",
        help="Run without ADX filter (original strategy)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison: WITH vs WITHOUT ADX filter"
    )
    parser.add_argument(
        "--double-lock",
        action="store_true",
        help="Use Double-Lock filter: TQQQ ADX > 30 AND QQQ ADX > 25"
    )
    parser.add_argument(
        "--compare-double-lock",
        action="store_true",
        help="Run comparison: Single ADX vs Double-Lock"
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  MOMENTUM SCALPER BACKTESTER v3")
    print("  Bi-Cameral Trading Bot")
    print("  Features: ADX Filter + Double-Lock + Optimized Risk for 3x ETFs")
    print("=" * 60 + "\n")

    if args.compare_double_lock:
        # Run Double-Lock comparison mode
        run_double_lock_comparison(args.symbol, args.days, args.verbose)
    elif args.compare:
        # Run comparison mode
        run_comparison(args.symbol, args.days, args.verbose)
    else:
        # Single run mode
        use_adx = not args.no_adx
        use_double_lock = args.double_lock
        backtester = BacktestScalper(
            symbol=args.symbol,
            days=args.days,
            use_adx=use_adx,
            use_optimized_risk=use_adx or use_double_lock,
            use_double_lock=use_double_lock
        )
        report = backtester.run(verbose=args.verbose)

        # Return total PnL for easy comparison
        print(f"\n>>> Total PnL for {args.symbol}: ${report.get('total_pnl_dollars', 0):+.2f}")


if __name__ == "__main__":
    main()
