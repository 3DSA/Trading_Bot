#!/usr/bin/env python3
"""
Backtest Engine - Local Research Lab for the Bi-Cameral Trading Bot

Simulates trading strategies using historical data from yfinance.
Used for parameter tuning before deploying to live/paper trading.

Author: Bi-Cameral System
"""

import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yfinance as yf


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_duration_minutes: float
    trades: list[Trade] = field(default_factory=list)

    def __str__(self) -> str:
        return f"""
{'='*60}
  BACKTEST RESULTS - {self.symbol}
  Period: {self.start_date} to {self.end_date}
{'='*60}
  Total Trades:      {self.total_trades}
  Winning Trades:    {self.winning_trades}
  Losing Trades:     {self.losing_trades}
  Win Rate:          {self.win_rate:.1f}%

  Total PnL:         ${self.total_pnl:,.2f}
  Total PnL %:       {self.total_pnl_pct:.2f}%
  Max Drawdown:      {self.max_drawdown_pct:.2f}%
  Sharpe Ratio:      {self.sharpe_ratio:.2f}

  Avg Trade Duration: {self.avg_trade_duration_minutes:.1f} minutes
{'='*60}
"""


class BacktestEngine:
    """
    Simulates trading strategies on historical data.
    """

    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config.json"
    INITIAL_CAPITAL = 10000.0  # Starting capital for simulation

    def __init__(self, config_override: Optional[dict] = None):
        """
        Initialize the backtest engine.

        Args:
            config_override: Optional dict to override config.json values
        """
        self.config = self._load_config()
        if config_override:
            self._apply_override(config_override)

        # Extract strategy params
        strategy = self.config.get("strategy", {})
        self.buy_rsi_threshold = strategy.get("buy_rsi_threshold", 30)
        self.sell_rsi_threshold = strategy.get("sell_rsi_threshold", 70)
        self.bb_period = strategy.get("bb_period", 20)
        self.bb_std_dev = strategy.get("bb_std_dev", 2.0)
        self.sma_trend_period = strategy.get("sma_trend_period", 200)
        self.use_trailing_stop = strategy.get("use_trailing_stop", False)
        self.trailing_stop_pct = strategy.get("trailing_stop_pct", 0.015)
        self.force_exit_at_bb_upper = strategy.get("force_exit_at_bb_upper", True)

        # Extract risk params
        risk = self.config.get("risk_management", {})
        self.stop_loss_pct = risk.get("stop_loss_pct", 0.04)
        self.take_profit_pct = risk.get("take_profit_pct", 0.08)
        self.max_holding_minutes = risk.get("max_holding_minutes", 390)
        self.cooldown_minutes = risk.get("cooldown_minutes", 5)

        # VIX filter for simulating AI fear/greed
        self.use_vix_filter = True
        self.vix_data: Optional[np.ndarray] = None

        self.symbol = self.config.get("symbol", "TQQQ")

    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        try:
            with open(self.CONFIG_PATH, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _apply_override(self, override: dict):
        """Apply override values to config."""
        for key, value in override.items():
            if "." in key:
                # Handle nested keys like "strategy.buy_rsi_threshold"
                parts = key.split(".")
                target = self.config
                for part in parts[:-1]:
                    target = target.setdefault(part, {})
                target[parts[-1]] = value
            else:
                self.config[key] = value

    def fetch_data(self, days: int = 30) -> tuple[np.ndarray, np.ndarray, list[datetime]]:
        """
        Fetch historical minute-level data from yfinance.

        Args:
            days: Number of days of history to fetch

        Returns:
            Tuple of (close_prices, volumes, timestamps)
        """
        print(f"[DATA] Fetching {days} days of minute data for {self.symbol}...")

        # yfinance limits minute data to 7 days per request, max 60 days total
        ticker = yf.Ticker(self.symbol)

        # Fetch data in chunks if needed
        all_data = []
        end_date = datetime.now()

        # For minute data, yfinance allows max 7 days per request
        chunk_size = 7
        remaining_days = min(days, 60)  # yfinance max is 60 days for minute data

        while remaining_days > 0:
            fetch_days = min(chunk_size, remaining_days)
            start_date = end_date - timedelta(days=fetch_days)

            data = ticker.history(
                start=start_date,
                end=end_date,
                interval="1m"
            )

            if not data.empty:
                all_data.append(data)

            end_date = start_date
            remaining_days -= fetch_days

        if not all_data:
            raise ValueError(f"No data fetched for {self.symbol}")

        # Combine all chunks
        combined = all_data[-1]  # Start with oldest
        for df in reversed(all_data[:-1]):
            combined = df._append(combined)

        # Remove duplicates and sort
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()

        closes = combined["Close"].values
        volumes = combined["Volume"].values
        timestamps = [ts.to_pydatetime() for ts in combined.index]

        print(f"[DATA] Fetched {len(closes)} bars from {timestamps[0]} to {timestamps[-1]}")

        return closes, volumes, timestamps

    def fetch_vix_data(self, timestamps: list[datetime]) -> np.ndarray:
        """
        Fetch VIX data to simulate AI fear/greed filter.
        Maps daily VIX to minute-level timestamps.
        """
        print("[DATA] Fetching VIX data for fear/greed simulation...")

        try:
            vix = yf.Ticker("^VIX")
            start_date = timestamps[0].date()
            end_date = timestamps[-1].date() + timedelta(days=1)

            vix_hist = vix.history(start=start_date, end=end_date, interval="1d")

            if vix_hist.empty:
                print("[WARN] No VIX data available, disabling filter")
                return np.full(len(timestamps), 20.0)  # Neutral default

            # Create mapping from date to VIX close
            vix_by_date = {ts.date(): row["Close"] for ts, row in vix_hist.iterrows()}

            # Map VIX to each minute timestamp
            vix_values = []
            last_vix = 20.0  # Default
            for ts in timestamps:
                date = ts.date()
                if date in vix_by_date:
                    last_vix = vix_by_date[date]
                vix_values.append(last_vix)

            print(f"[DATA] VIX range: {min(vix_values):.1f} - {max(vix_values):.1f}")
            return np.array(vix_values)

        except Exception as e:
            print(f"[WARN] VIX fetch failed: {e}, using neutral values")
            return np.full(len(timestamps), 20.0)

    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        rsi = np.zeros(len(prices))
        rsi[:period] = 50  # Default for warmup period

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(prices) - 1):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi[i + 1] = 100
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100 - (100 / (1 + rs))

        return rsi

    def calculate_bollinger_bands(
        self, prices: np.ndarray, period: int, std_dev: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        middle = np.zeros(len(prices))
        upper = np.zeros(len(prices))
        lower = np.zeros(len(prices))

        for i in range(period - 1, len(prices)):
            window = prices[i - period + 1:i + 1]
            sma = np.mean(window)
            std = np.std(window)

            middle[i] = sma
            upper[i] = sma + (std_dev * std)
            lower[i] = sma - (std_dev * std)

        return upper, middle, lower

    def calculate_sma(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average."""
        sma = np.zeros(len(prices))

        for i in range(period - 1, len(prices)):
            sma[i] = np.mean(prices[i - period + 1:i + 1])

        return sma

    def run(self, days: int = 30) -> BacktestResult:
        """
        Run the backtest simulation.

        Args:
            days: Number of days of historical data to use

        Returns:
            BacktestResult with performance metrics
        """
        trailing_str = f"TS={self.trailing_stop_pct*100:.1f}%" if self.use_trailing_stop else "OFF"
        bb_exit_str = "BB_EXIT=ON" if self.force_exit_at_bb_upper else "BB_EXIT=OFF"

        print(f"\n{'='*60}")
        print(f"  BACKTEST ENGINE - {self.symbol}")
        print(f"  Strategy: RSI({self.buy_rsi_threshold}/{self.sell_rsi_threshold}), "
              f"BB({self.bb_period}, {self.bb_std_dev}), SMA({self.sma_trend_period})")
        print(f"  Risk: SL={self.stop_loss_pct*100:.1f}%, TP={self.take_profit_pct*100:.1f}%, {trailing_str}")
        print(f"  Flags: {bb_exit_str}, VIX_FILTER={'ON' if self.use_vix_filter else 'OFF'}")
        print(f"{'='*60}\n")

        # Fetch historical data
        closes, volumes, timestamps = self.fetch_data(days)

        # Fetch VIX for fear/greed simulation
        vix_data = self.fetch_vix_data(timestamps) if self.use_vix_filter else None

        # Calculate indicators
        print("[CALC] Computing indicators...")
        rsi = self.calculate_rsi(closes)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(
            closes, self.bb_period, self.bb_std_dev
        )
        sma = self.calculate_sma(closes, self.sma_trend_period)

        # Simulation state
        position: Optional[Trade] = None
        trades: list[Trade] = []
        capital = self.INITIAL_CAPITAL
        peak_capital = capital
        max_drawdown = 0.0
        last_exit_time: Optional[datetime] = None
        daily_returns: list[float] = []
        prev_capital = capital

        # Trailing stop state
        trailing_stop_price: float = 0.0
        highest_price_since_entry: float = 0.0

        # Warmup period (need enough data for SMA)
        warmup = max(self.sma_trend_period, self.bb_period, 14) + 1

        print(f"[SIM] Running simulation on {len(closes) - warmup} bars...")

        for i in range(warmup, len(closes)):
            price = closes[i]
            ts = timestamps[i]
            current_rsi = rsi[i]
            current_bb_lower = bb_lower[i]
            current_bb_upper = bb_upper[i]
            current_sma = sma[i]
            current_vix = vix_data[i] if vix_data is not None else 20.0

            # VIX-based dynamic RSI threshold (simulating AI manager)
            if self.use_vix_filter:
                if current_vix > 30:
                    # High fear - disable buying
                    effective_buy_enabled = False
                    effective_rsi_threshold = self.buy_rsi_threshold
                elif current_vix < 15:
                    # Low fear - aggressive
                    effective_buy_enabled = True
                    effective_rsi_threshold = min(40, self.buy_rsi_threshold + 10)
                else:
                    # Normal
                    effective_buy_enabled = True
                    effective_rsi_threshold = self.buy_rsi_threshold
            else:
                effective_buy_enabled = True
                effective_rsi_threshold = self.buy_rsi_threshold

            # Check for open position
            if position:
                entry_price = position.entry_price
                pnl_pct = (price - entry_price) / entry_price
                holding_time = (ts - position.entry_time).total_seconds() / 60

                # Update trailing stop
                if self.use_trailing_stop and price > highest_price_since_entry:
                    highest_price_since_entry = price
                    trailing_stop_price = price * (1 - self.trailing_stop_pct)

                exit_reason = None

                # Stop Loss (initial)
                if pnl_pct <= -self.stop_loss_pct:
                    exit_reason = "STOP_LOSS"

                # Trailing Stop (only if in profit and trailing is enabled)
                elif self.use_trailing_stop and trailing_stop_price > 0 and price <= trailing_stop_price:
                    exit_reason = "TRAILING_STOP"

                # Take Profit (hard target)
                elif pnl_pct >= self.take_profit_pct:
                    exit_reason = "TAKE_PROFIT"

                # RSI Overbought
                elif current_rsi >= self.sell_rsi_threshold:
                    exit_reason = "RSI_OVERBOUGHT"

                # Price above upper BB (only if enabled)
                elif self.force_exit_at_bb_upper and price >= current_bb_upper and current_bb_upper > 0:
                    exit_reason = "BB_UPPER"

                # Max holding time
                elif holding_time >= self.max_holding_minutes:
                    exit_reason = "TIME_LIMIT"

                if exit_reason:
                    # Close position
                    shares = capital / entry_price
                    pnl = shares * (price - entry_price)

                    position.exit_time = ts
                    position.exit_price = price
                    position.exit_reason = exit_reason
                    position.pnl = pnl
                    position.pnl_pct = pnl_pct * 100

                    capital += pnl
                    trades.append(position)
                    last_exit_time = ts
                    position = None
                    trailing_stop_price = 0.0
                    highest_price_since_entry = 0.0

            else:
                # Check for entry signal
                cooldown_ok = (
                    last_exit_time is None or
                    (ts - last_exit_time).total_seconds() / 60 >= self.cooldown_minutes
                )

                # Trend filter: only buy in uptrends
                is_uptrend = price > current_sma if current_sma > 0 else True

                if (cooldown_ok and
                    effective_buy_enabled and
                    is_uptrend and
                    current_rsi < effective_rsi_threshold and
                    price < current_bb_lower and
                    current_bb_lower > 0):

                    # Open position
                    position = Trade(
                        entry_time=ts,
                        entry_price=price
                    )
                    highest_price_since_entry = price
                    trailing_stop_price = 0.0  # Not active until we're in profit

            # Track drawdown
            if capital > peak_capital:
                peak_capital = capital
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, drawdown)

            # Track daily returns for Sharpe ratio
            if i > warmup and timestamps[i].date() != timestamps[i-1].date():
                daily_return = (capital - prev_capital) / prev_capital if prev_capital > 0 else 0
                daily_returns.append(daily_return)
                prev_capital = capital

        # Close any remaining position at end
        if position:
            price = closes[-1]
            ts = timestamps[-1]
            shares = capital / position.entry_price
            pnl = shares * (price - position.entry_price)
            pnl_pct = (price - position.entry_price) / position.entry_price

            position.exit_time = ts
            position.exit_price = price
            position.exit_reason = "END_OF_DATA"
            position.pnl = pnl
            position.pnl_pct = pnl_pct * 100

            capital += pnl
            trades.append(position)

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl <= 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = capital - self.INITIAL_CAPITAL
        total_pnl_pct = (total_pnl / self.INITIAL_CAPITAL) * 100

        # Sharpe ratio (annualized, assuming 252 trading days)
        if daily_returns:
            avg_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0

        # Average trade duration
        durations = [
            (t.exit_time - t.entry_time).total_seconds() / 60
            for t in trades if t.exit_time
        ]
        avg_duration = np.mean(durations) if durations else 0

        result = BacktestResult(
            symbol=self.symbol,
            start_date=timestamps[warmup].strftime("%Y-%m-%d"),
            end_date=timestamps[-1].strftime("%Y-%m-%d"),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl_pct,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe,
            avg_trade_duration_minutes=avg_duration,
            trades=trades
        )

        print(result)

        # Print recent trades
        print("\n  RECENT TRADES:")
        print("  " + "-"*56)
        for trade in trades[-5:]:
            exit_time_str = trade.exit_time.strftime('%H:%M') if trade.exit_time else 'OPEN'
            exit_price = trade.exit_price if trade.exit_price else 0
            sign = '+' if trade.pnl >= 0 else ''
            print(f"  {trade.entry_time.strftime('%m/%d %H:%M')} -> "
                  f"{exit_time_str} | "
                  f"${trade.entry_price:.2f} -> ${exit_price:.2f} | "
                  f"{trade.exit_reason or 'HOLDING'} | "
                  f"{sign}{trade.pnl_pct:.2f}%")
        print("  " + "-"*56)

        return result


def main():
    """Entry point for backtest engine."""
    parser = argparse.ArgumentParser(description="Backtest the Bi-Cameral Trading Bot")
    parser.add_argument("--days", type=int, default=30, help="Days of history to backtest")
    parser.add_argument("--symbol", type=str, help="Override symbol from config")
    parser.add_argument("--rsi-buy", type=int, help="Buy RSI threshold")
    parser.add_argument("--rsi-sell", type=int, help="Sell RSI threshold")
    parser.add_argument("--stop-loss", type=float, help="Stop loss percentage (e.g., 0.04)")
    parser.add_argument("--take-profit", type=float, help="Take profit percentage (e.g., 0.08)")

    args = parser.parse_args()

    # Build override dict
    override = {}
    if args.symbol:
        override["symbol"] = args.symbol
    if args.rsi_buy:
        override["strategy.buy_rsi_threshold"] = args.rsi_buy
    if args.rsi_sell:
        override["strategy.sell_rsi_threshold"] = args.rsi_sell
    if args.stop_loss:
        override["risk_management.stop_loss_pct"] = args.stop_loss
    if args.take_profit:
        override["risk_management.take_profit_pct"] = args.take_profit

    engine = BacktestEngine(config_override=override if override else None)
    result = engine.run(days=args.days)

    # Summary for easy copy-paste
    print(f"\n  SUMMARY: Win Rate = {result.win_rate:.1f}% | PnL = ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")


if __name__ == "__main__":
    main()
