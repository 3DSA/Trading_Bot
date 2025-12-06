#!/usr/bin/env python3
"""
Universal Backtester - Tests Strategy Library Classes Directly

This backtester imports the EXACT same strategy classes that reflex.py uses.
What you test here is EXACTLY what trades live.

Usage:
    python backtest_universal.py --strategy momentum_scalper --symbol TQQQ --days 30
    python backtest_universal.py --strategy mean_reversion --symbol TQQQ --days 30
    python backtest_universal.py --strategy crisis_alpha --symbol SQQQ --days 30
    python backtest_universal.py --strategy volatility_breakout --symbol TQQQ --days 5

Author: Bi-Cameral System
"""

import argparse
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import the ACTUAL Strategy Library (same code as live bot)
from strategies.factory import StrategyManager, get_strategy, list_strategies, REGIME_STRATEGY_MAP
from strategies.base import StrategySignal, SignalType, PositionSizing, StrategyConfig


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    qty: float = 100
    side: str = "BUY"
    entry_reason: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    position_size: str = "FULL"


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    strategy_name: str
    strategy_version: str
    symbol: str
    underlying_symbol: str
    start_date: datetime
    end_date: datetime
    total_bars: int
    trades: List[Trade] = field(default_factory=list)

    # Calculated metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0

    # Strategy-specific
    signals_generated: int = 0
    buy_signals: int = 0
    exit_signals: int = 0
    hold_signals: int = 0


class UniversalBacktester:
    """
    Backtester that uses the actual Strategy Library classes.

    This ensures that what you test is EXACTLY what trades live.
    """

    BASE_DIR = Path(__file__).parent

    # Position sizing (same as reflex.py)
    POSITION_SIZE_MAP = {
        PositionSizing.FULL: 1.0,
        PositionSizing.HALF: 0.5,
        PositionSizing.QUARTER: 0.25,
        PositionSizing.NONE: 0.0,
    }

    def __init__(self, strategy_name: str, symbol: str, underlying_symbol: str = "QQQ",
                 initial_capital: float = 100000, position_pct: float = 0.10):
        """
        Initialize the Universal Backtester.

        Args:
            strategy_name: Name of strategy from Strategy Library
            symbol: Target symbol to trade (TQQQ, SQQQ)
            underlying_symbol: Context symbol (QQQ)
            initial_capital: Starting capital
            position_pct: Percentage of capital per trade
        """
        load_dotenv(self.BASE_DIR / ".env", override=True)

        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET must be set in .env")

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        self.symbol = symbol
        self.underlying_symbol = underlying_symbol
        self.initial_capital = initial_capital
        self.position_pct = position_pct

        # Initialize Strategy Manager with config
        config = StrategyConfig(
            symbol=symbol,
            underlying_symbol=underlying_symbol,
            use_double_lock=True,
            double_lock_leveraged_adx=30,
            double_lock_underlying_adx=25,
        )

        self.strategy_manager = StrategyManager(default_config=config)

        # Load the requested strategy
        self._load_strategy(strategy_name)

        print(f"[INIT] Universal Backtester")
        print(f"[INIT] Strategy: {self.strategy.name} v{self.strategy.version}")
        print(f"[INIT] Description: {self.strategy.description}")
        print(f"[INIT] Symbol: {symbol} | Underlying: {underlying_symbol}")

    def _load_strategy(self, strategy_name: str):
        """Load strategy by name using the factory."""
        # Map strategy name to regime
        strategy_regime_map = {
            "momentum_scalper": "TREND",
            "sniper": "TREND",
            "mean_reversion": "CHOP",
            "rubber_band": "CHOP",
            "volatility_breakout": "VOLATILE",
            "news_trader": "VOLATILE",
            "crisis_alpha": "CRISIS",
            "bear": "CRISIS",
        }

        regime = strategy_regime_map.get(strategy_name.lower(), "TREND")
        self.strategy = self.strategy_manager.set_regime(regime)
        self.strategy_name = strategy_name

    def fetch_data(self, days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical data for backtesting.

        Args:
            days: Number of days of history

        Returns:
            Tuple of (symbol_df, underlying_df)
        """
        print(f"\n[DATA] Fetching {days} days of data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        symbols = [self.symbol, self.underlying_symbol]

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            feed=DataFeed.SIP  # Use SIP for historical (more complete than IEX)
        )

        bars = self.data_client.get_stock_bars(request)

        # Access the data dict inside BarSet
        bars_data = bars.data if hasattr(bars, 'data') else {}

        symbol_bars = bars_data.get(self.symbol, [])
        underlying_bars = bars_data.get(self.underlying_symbol, [])

        symbol_df = self._bars_to_df(symbol_bars)
        underlying_df = self._bars_to_df(underlying_bars)

        print(f"[DATA] {self.symbol}: {len(symbol_df)} bars")
        print(f"[DATA] {self.underlying_symbol}: {len(underlying_df)} bars")

        return symbol_df, underlying_df

    def _bars_to_df(self, bars) -> pd.DataFrame:
        """Convert Alpaca bars to DataFrame."""
        if not bars:
            return pd.DataFrame()

        # Handle both dict bars and object bars
        if isinstance(bars[0], dict):
            data = {
                "Open": [float(b['open']) for b in bars],
                "High": [float(b['high']) for b in bars],
                "Low": [float(b['low']) for b in bars],
                "Close": [float(b['close']) for b in bars],
                "Volume": [float(b['volume']) for b in bars],
            }
            index = pd.DatetimeIndex([b['timestamp'] for b in bars])
        else:
            data = {
                "Open": [float(b.open) for b in bars],
                "High": [float(b.high) for b in bars],
                "Low": [float(b.low) for b in bars],
                "Close": [float(b.close) for b in bars],
                "Volume": [float(b.volume) for b in bars],
            }
            index = pd.DatetimeIndex([b.timestamp for b in bars])

        return pd.DataFrame(data, index=index)

    def run(self, days: int = 30, warmup_bars: int = 50) -> BacktestResult:
        """
        Run the backtest.

        Args:
            days: Days of history to test
            warmup_bars: Bars needed for indicator warmup

        Returns:
            BacktestResult with all metrics
        """
        # Fetch data
        symbol_df, underlying_df = self.fetch_data(days)

        if symbol_df.empty or underlying_df.empty:
            raise ValueError("No data available for backtesting")

        # Align DataFrames by timestamp
        common_idx = symbol_df.index.intersection(underlying_df.index)
        symbol_df = symbol_df.loc[common_idx]
        underlying_df = underlying_df.loc[common_idx]

        print(f"\n[BACKTEST] Starting simulation...")
        print(f"[BACKTEST] Period: {symbol_df.index[0]} to {symbol_df.index[-1]}")
        print(f"[BACKTEST] Total bars: {len(symbol_df)}")
        print(f"[BACKTEST] Warmup: {warmup_bars} bars")

        # Initialize result
        result = BacktestResult(
            strategy_name=self.strategy.name,
            strategy_version=self.strategy.version,
            symbol=self.symbol,
            underlying_symbol=self.underlying_symbol,
            start_date=symbol_df.index[warmup_bars],
            end_date=symbol_df.index[-1],
            total_bars=len(symbol_df) - warmup_bars,
        )

        # Simulation state
        capital = self.initial_capital
        position: Optional[Trade] = None
        equity_curve = []
        peak_equity = capital
        max_drawdown = 0.0

        # Run simulation
        for i in range(warmup_bars, len(symbol_df)):
            # Slice data up to current bar (simulating live stream)
            current_symbol_df = symbol_df.iloc[:i+1].copy()
            current_underlying_df = underlying_df.iloc[:i+1].copy()

            current_bar = symbol_df.iloc[i]
            current_time = symbol_df.index[i]
            current_price = current_bar["Close"]

            # Prepare data with strategy-specific indicators
            prepared_df = self.strategy.prepare_data(current_symbol_df)

            # Build position dict for strategy
            current_position = None
            if position is not None:
                current_position = {
                    "entry_price": position.entry_price,
                    "qty": position.qty,
                    "entry_time": position.entry_time,
                    "highest_price": current_price,  # Simplified
                }

            # Generate signal from strategy
            signal = self.strategy.generate_signal(
                prepared_df,
                current_position=current_position,
                underlying_df=current_underlying_df
            )

            result.signals_generated += 1

            # Track signal types
            if signal.signal == SignalType.BUY:
                result.buy_signals += 1
            elif signal.signal == SignalType.EXIT:
                result.exit_signals += 1
            else:
                result.hold_signals += 1

            # Execute signals
            if signal.signal == SignalType.BUY and position is None:
                # Calculate position size
                size_mult = self.POSITION_SIZE_MAP.get(signal.position_size, 1.0)
                allocation = capital * self.position_pct * size_mult
                qty = int(allocation / current_price)

                if qty > 0:
                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        qty=qty,
                        entry_reason=signal.reason,
                        position_size=signal.position_size.name,
                    )

            elif signal.signal == SignalType.EXIT and position is not None:
                # Close position
                position.exit_time = current_time
                position.exit_price = current_price
                position.exit_reason = signal.reason

                # Calculate P&L
                position.pnl = (position.exit_price - position.entry_price) * position.qty
                position.pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100

                capital += position.pnl
                result.trades.append(position)
                position = None

            # Track equity curve
            current_equity = capital
            if position is not None:
                unrealized = (current_price - position.entry_price) * position.qty
                current_equity += unrealized

            equity_curve.append(current_equity)

            # Track drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity

            drawdown = peak_equity - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close any open position at end
        if position is not None:
            position.exit_time = symbol_df.index[-1]
            position.exit_price = symbol_df.iloc[-1]["Close"]
            position.exit_reason = "END_OF_BACKTEST"
            position.pnl = (position.exit_price - position.entry_price) * position.qty
            position.pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100
            capital += position.pnl
            result.trades.append(position)

        # Calculate metrics
        self._calculate_metrics(result, equity_curve, max_drawdown)

        return result

    def _calculate_metrics(self, result: BacktestResult, equity_curve: List[float],
                          max_drawdown: float):
        """Calculate performance metrics."""
        trades = result.trades

        if not trades:
            return

        # Basic metrics
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.win_rate = len(wins) / len(trades) * 100 if trades else 0

        result.total_pnl = sum(t.pnl for t in trades)
        result.total_pnl_pct = (result.total_pnl / self.initial_capital) * 100

        # Average win/loss
        result.avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        result.avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        result.max_drawdown = max_drawdown
        result.max_drawdown_pct = (max_drawdown / self.initial_capital) * 100

        # Sharpe ratio (simplified - annualized)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 390)  # Annualized

    def print_report(self, result: BacktestResult):
        """Print detailed backtest report."""
        print("\n" + "=" * 70)
        print("  UNIVERSAL BACKTEST REPORT")
        print("  Testing ACTUAL Strategy Library Code")
        print("=" * 70)

        print(f"\n  Strategy: {result.strategy_name} v{result.strategy_version}")
        print(f"  Symbol: {result.symbol} | Underlying: {result.underlying_symbol}")
        print(f"  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"  Total Bars: {result.total_bars:,}")

        print("\n" + "-" * 70)
        print("  SIGNAL ANALYSIS")
        print("-" * 70)
        print(f"  Total Signals: {result.signals_generated:,}")
        print(f"  BUY Signals: {result.buy_signals:,}")
        print(f"  EXIT Signals: {result.exit_signals:,}")
        print(f"  HOLD Signals: {result.hold_signals:,}")

        print("\n" + "-" * 70)
        print("  PERFORMANCE METRICS")
        print("-" * 70)
        print(f"  Total Trades: {len(result.trades)}")
        print(f"  Win Rate: {result.win_rate:.1f}% ({result.win_count}W / {result.loss_count}L)")
        print(f"  Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
        print(f"  Avg Win: ${result.avg_win:,.2f}")
        print(f"  Avg Loss: ${result.avg_loss:,.2f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown: ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

        if result.trades:
            print("\n" + "-" * 70)
            print("  TRADE LOG (Last 10)")
            print("-" * 70)
            for trade in result.trades[-10:]:
                pnl_color = "+" if trade.pnl > 0 else ""
                size_label = f"[{trade.position_size}]" if trade.position_size != "FULL" else ""
                exit_price = trade.exit_price if trade.exit_price else 0
                exit_time_str = trade.exit_time.strftime('%H:%M') if trade.exit_time else '???'
                print(f"  {trade.entry_time.strftime('%m/%d %H:%M')} -> {exit_time_str} | "
                      f"${trade.entry_price:.2f} -> ${exit_price:.2f} | "
                      f"{pnl_color}${trade.pnl:.2f} ({trade.pnl_pct:+.2f}%) {size_label}")
                print(f"    Entry: {trade.entry_reason[:50]}")
                print(f"    Exit: {trade.exit_reason[:50]}")

        print("\n" + "=" * 70)

        # Verdict
        if result.total_pnl > 0 and result.win_rate >= 50:
            print("  VERDICT: STRATEGY VALIDATED")
        elif result.total_pnl > 0:
            print("  VERDICT: PROFITABLE (but low win rate)")
        else:
            print("  VERDICT: NEEDS OPTIMIZATION")

        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Universal Backtester - Tests Strategy Library Classes Directly"
    )

    parser.add_argument(
        "--strategy",
        type=str,
        default="momentum_scalper",
        choices=["momentum_scalper", "mean_reversion", "volatility_breakout", "crisis_alpha",
                 "sniper", "rubber_band", "news_trader", "bear"],
        help="Strategy to test (default: momentum_scalper)"
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default="TQQQ",
        help="Target symbol (default: TQQQ)"
    )

    parser.add_argument(
        "--underlying",
        type=str,
        default="QQQ",
        help="Underlying/context symbol (default: QQQ)"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history (default: 30)"
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=100000,
        help="Initial capital (default: 100000)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup bars for indicators (default: 50)"
    )

    args = parser.parse_args()

    # Auto-set symbol for crisis_alpha
    if args.strategy in ["crisis_alpha", "bear"] and args.symbol == "TQQQ":
        print("[INFO] crisis_alpha strategy uses SQQQ, switching symbol...")
        args.symbol = "SQQQ"

    print("\n" + "=" * 70)
    print("  UNIVERSAL BACKTESTER")
    print("  'Test What You Trade'")
    print("=" * 70)
    print(f"\n  Available Strategies: {', '.join(list_strategies())}")
    print(f"  Selected: {args.strategy}")
    print("=" * 70)

    # Run backtest
    backtester = UniversalBacktester(
        strategy_name=args.strategy,
        symbol=args.symbol,
        underlying_symbol=args.underlying,
        initial_capital=args.capital,
    )

    result = backtester.run(days=args.days, warmup_bars=args.warmup)
    backtester.print_report(result)


if __name__ == "__main__":
    main()
