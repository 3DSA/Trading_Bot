#!/usr/bin/env python3
"""
Regime-Switching Backtester - Simulates Full Bi-Cameral System

This backtester imports the ACTUAL manager.py regime detection logic
and uses the ACTUAL Strategy Library. What you test here is what trades live.

Since VIX and Fear & Greed are scraped live, we simulate them with
configurable scenarios:
- NORMAL: VIX=18, F&G=55 (neutral market - ADX decides)
- FEARFUL: VIX=22, F&G=35 (elevated fear - volatility_breakout)
- CRISIS: VIX=30, F&G=18 (panic - crisis_alpha with SQQQ)
- GREEDY: VIX=14, F&G=70 (bullish - momentum_scalper if ADX > 25)

Usage:
    python tests/backtest_regime.py --days 90
    python tests/backtest_regime.py --days 90 --scenario fearful
    python tests/backtest_regime.py --days 90 --verbose

Author: Bi-Cameral System
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import the ACTUAL Strategy Library (same as live bot)
from strategies.factory import get_strategy, list_strategies, REGIME_STRATEGY_MAP
from strategies.base import StrategySignal, SignalType, PositionSizing, StrategyConfig
from strategies.shared_utils import calc_adx

# Import Options Adapter for shadow P&L tracking
from execution.options_adapter import OptionsAdapter, compare_equity_vs_options


@dataclass
class Trade:
    """Record of a single trade."""
    entry_time: datetime
    entry_price: float
    symbol: str = "TQQQ"
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    qty: float = 100
    entry_reason: str = ""
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    strategy_used: str = ""
    regime: str = ""


@dataclass
class RegimeBacktestResult:
    """Results of regime-switching backtest."""
    start_date: datetime
    end_date: datetime
    total_bars: int
    scenario: str = "normal"
    trades: List[Trade] = field(default_factory=list)

    # Performance metrics
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0

    # Regime stats
    regime_changes: int = 0
    regime_distribution: Dict[str, float] = field(default_factory=dict)

    # Per-strategy stats
    strategy_stats: Dict[str, Dict] = field(default_factory=dict)


class RegimeBacktester:
    """
    Backtester that uses the ACTUAL manager.py regime detection logic
    and the ACTUAL Strategy Library classes.
    """

    BASE_DIR = Path(__file__).parent.parent

    # Market scenarios (simulating VIX/F&G since we can't backtest scraped data)
    SCENARIOS = {
        "normal": {"vix": 18, "fear_greed": 55, "description": "Neutral market - ADX decides"},
        "fearful": {"vix": 22, "fear_greed": 35, "description": "Elevated fear - volatility focus"},
        "crisis": {"vix": 30, "fear_greed": 18, "description": "Market panic - SQQQ mode"},
        "greedy": {"vix": 14, "fear_greed": 70, "description": "Bullish greed - momentum if ADX > 25"},
        "euphoria": {"vix": 12, "fear_greed": 88, "description": "Extreme greed - caution mode"},
    }

    # Position sizing (from strategies.base)
    POSITION_SIZE_MAP = {
        PositionSizing.FULL: 1.0,
        PositionSizing.HALF: 0.5,
        PositionSizing.QUARTER: 0.25,
        PositionSizing.NONE: 0.0,
    }

    def __init__(self, scenario: str = "normal", initial_capital: float = 100000,
                 position_pct: float = 0.10, enable_shadow_options: bool = False):
        """Initialize the regime backtester."""
        load_dotenv(self.BASE_DIR / ".env", override=True)

        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET must be set in .env")

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        self.scenario = scenario
        self.scenario_params = self.SCENARIOS.get(scenario, self.SCENARIOS["normal"])
        self.initial_capital = initial_capital
        self.position_pct = position_pct

        # Shadow options tracking
        self.enable_shadow_options = enable_shadow_options
        self.options_adapter = OptionsAdapter(mode="simulation") if enable_shadow_options else None

        # Load ALL strategies from the Strategy Library
        self.strategies = {}
        for strategy_name in list_strategies():
            config = StrategyConfig(
                symbol="TQQQ" if strategy_name != "crisis_alpha" else "SQQQ",
                underlying_symbol="QQQ",
                use_double_lock=True,
            )
            self.strategies[strategy_name] = get_strategy(strategy_name, config)

        print(f"[INIT] Regime-Switching Backtester")
        print(f"[INIT] Scenario: {scenario} - {self.scenario_params['description']}")
        print(f"[INIT] Simulated VIX={self.scenario_params['vix']}, F&G={self.scenario_params['fear_greed']}")
        print(f"[INIT] Loaded strategies: {list(self.strategies.keys())}")
        if enable_shadow_options:
            print(f"[INIT] Shadow Options Tracking: ENABLED (30x leverage simulation)")

    def _deterministic_regime(self, intelligence: dict) -> dict:
        """
        EXACT copy of manager.py's _deterministic_regime logic.
        This is the 3-layer decision matrix.
        """
        vix = intelligence.get('vix', 15)
        fg = intelligence.get('fear_greed', 50)
        adx = intelligence.get('adx', 20)

        # LAYER 1: SAFETY
        if vix > 35:
            return {"active_strategy": "crisis_alpha", "symbol": "SQQQ", "buy_enabled": False,
                    "reasoning": f"SAFETY: VIX={vix} > 35. CASH."}

        if fg < 15:
            return {"active_strategy": "mean_reversion", "symbol": "TQQQ", "buy_enabled": False,
                    "reasoning": f"SAFETY: F&G={fg} < 15. Capitulation."}

        if fg > 85:
            return {"active_strategy": "momentum_scalper", "symbol": "TQQQ", "buy_enabled": False,
                    "reasoning": f"SAFETY: F&G={fg} > 85. Euphoria."}

        if vix > 28 or fg < 20:
            return {"active_strategy": "crisis_alpha", "symbol": "SQQQ", "buy_enabled": True,
                    "reasoning": f"CRISIS: VIX={vix}, F&G={fg}. SQQQ mode."}

        # LAYER 2: PHYSICS
        if adx < 25:
            return {"active_strategy": "mean_reversion", "symbol": "TQQQ", "buy_enabled": True,
                    "reasoning": f"PHYSICS: ADX={adx:.1f} < 25. No trend."}

        # LAYER 3: OPTIMIZATION
        if 20 <= vix <= 28 and fg < 40:
            return {"active_strategy": "volatility_breakout", "symbol": "TQQQ", "buy_enabled": True,
                    "reasoning": f"VOLATILE: VIX={vix}, F&G={fg}. ORB mode."}

        if adx > 25 and fg > 50:
            return {"active_strategy": "momentum_scalper", "symbol": "TQQQ", "buy_enabled": True,
                    "reasoning": f"TREND: ADX={adx:.1f}, F&G={fg}. Momentum."}

        # Default: Chop
        return {"active_strategy": "mean_reversion", "symbol": "TQQQ", "buy_enabled": True,
                "reasoning": f"CHOP: ADX={adx:.1f}, F&G={fg}. Sideways."}

    def fetch_data(self, days: int) -> Dict[str, pd.DataFrame]:
        """Fetch historical data for TQQQ, SQQQ, and QQQ."""
        end = datetime.now()
        start = end - timedelta(days=days)

        print(f"\n[DATA] Fetching {days} days of data...")

        symbols = ["TQQQ", "SQQQ", "QQQ"]
        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.SIP,  # Use SIP for complete historical data
        )

        bars = self.data_client.get_stock_bars(request)

        # Access the data dict inside BarSet
        bars_data = bars.data if hasattr(bars, 'data') else {}

        data = {}
        for symbol in symbols:
            symbol_bars = bars_data.get(symbol, [])
            data[symbol] = self._bars_to_df(symbol_bars)
            print(f"[DATA] {symbol}: {len(data[symbol])} bars")

        return data

    def _bars_to_df(self, bars) -> pd.DataFrame:
        """Convert Alpaca bars to DataFrame."""
        if not bars:
            return pd.DataFrame()

        # Handle both dict bars and object bars
        if isinstance(bars[0], dict):
            df = pd.DataFrame({
                "Open": [float(b['open']) for b in bars],
                "High": [float(b['high']) for b in bars],
                "Low": [float(b['low']) for b in bars],
                "Close": [float(b['close']) for b in bars],
                "Volume": [float(b['volume']) for b in bars],
            }, index=pd.DatetimeIndex([b['timestamp'] for b in bars]))
        else:
            df = pd.DataFrame({
                "Open": [float(b.open) for b in bars],
                "High": [float(b.high) for b in bars],
                "Low": [float(b.low) for b in bars],
                "Close": [float(b.close) for b in bars],
                "Volume": [float(b.volume) for b in bars],
            }, index=pd.DatetimeIndex([b.timestamp for b in bars]))

        return df

    def run(self, days: int = 90, warmup_bars: int = 50, verbose: bool = False) -> RegimeBacktestResult:
        """Run the regime-switching backtest."""
        # Fetch data for all symbols
        data = self.fetch_data(days)

        tqqq_df = data["TQQQ"]
        sqqq_df = data["SQQQ"]
        qqq_df = data["QQQ"]

        if tqqq_df.empty:
            raise ValueError("No TQQQ data available")

        # Align all DataFrames
        common_idx = tqqq_df.index.intersection(qqq_df.index).intersection(sqqq_df.index)
        tqqq_df = tqqq_df.loc[common_idx]
        sqqq_df = sqqq_df.loc[common_idx]
        qqq_df = qqq_df.loc[common_idx]

        print(f"\n[BACKTEST] Starting regime-switching simulation...")
        print(f"[BACKTEST] Period: {tqqq_df.index[0]} to {tqqq_df.index[-1]}")
        print(f"[BACKTEST] Total bars: {len(tqqq_df)}")

        # Pre-calculate indicators for each strategy
        print(f"[BACKTEST] Pre-calculating indicators for all strategies...")
        prepared_data = {}
        for name, strategy in self.strategies.items():
            symbol = "SQQQ" if name == "crisis_alpha" else "TQQQ"
            source_df = sqqq_df if symbol == "SQQQ" else tqqq_df
            prepared_data[name] = strategy.prepare_data(source_df.copy())

        # Prepare underlying (QQQ) data
        qqq_prepared = self.strategies["momentum_scalper"].prepare_data(qqq_df.copy())

        # Calculate ADX on TQQQ for regime detection
        tqqq_df["adx"] = calc_adx(tqqq_df, period=14)

        # Initialize result
        result = RegimeBacktestResult(
            start_date=tqqq_df.index[warmup_bars],
            end_date=tqqq_df.index[-1],
            total_bars=len(tqqq_df) - warmup_bars,
            scenario=self.scenario,
        )

        # Simulation state
        capital = self.initial_capital
        position: Optional[Trade] = None
        equity_curve = []
        peak_equity = capital
        max_drawdown = 0.0

        current_regime = None
        regime_counts = {s: 0 for s in self.strategies.keys()}
        strategy_stats = {s: {"trades": 0, "pnl": 0.0, "wins": 0} for s in self.strategies.keys()}

        print(f"[BACKTEST] Running simulation loop...")
        total_iterations = len(tqqq_df) - warmup_bars
        last_pct = 0

        for i in range(warmup_bars, len(tqqq_df)):
            # Progress
            pct = int((i - warmup_bars) / total_iterations * 100)
            if pct >= last_pct + 10:
                print(f"[BACKTEST] Progress: {pct}%")
                last_pct = pct

            current_time = tqqq_df.index[i]
            current_adx = tqqq_df.iloc[i]["adx"]
            if pd.isna(current_adx):
                current_adx = 20

            # Build intelligence dict (simulating scraped data)
            intelligence = {
                "vix": self.scenario_params["vix"],
                "fear_greed": self.scenario_params["fear_greed"],
                "adx": current_adx,
            }

            # Run the EXACT same regime detection as manager.py
            regime_decision = self._deterministic_regime(intelligence)
            active_strategy_name = regime_decision["active_strategy"]
            active_symbol = regime_decision["symbol"]
            buy_enabled = regime_decision["buy_enabled"]

            # Track regime
            regime_counts[active_strategy_name] += 1

            if active_strategy_name != current_regime:
                if current_regime is not None:
                    result.regime_changes += 1
                    if verbose:
                        print(f"  [{current_time.strftime('%m/%d %H:%M')}] {current_regime} -> {active_strategy_name} ({regime_decision['reasoning']})")
                current_regime = active_strategy_name

            # Get current price from the correct symbol
            if active_symbol == "SQQQ":
                current_price = sqqq_df.iloc[i]["Close"]
            else:
                current_price = tqqq_df.iloc[i]["Close"]

            # Build position dict and determine which strategy to use
            current_position = None
            if position is not None:
                # If we have a position in a different symbol, close it first
                if position.symbol != active_symbol:
                    # Force close position due to symbol change
                    if position.symbol == "SQQQ":
                        exit_price = sqqq_df.iloc[i]["Close"]
                    else:
                        exit_price = tqqq_df.iloc[i]["Close"]

                    position.exit_time = current_time
                    position.exit_price = exit_price
                    position.exit_reason = f"Symbol change: {position.symbol} -> {active_symbol}"
                    position.pnl = (position.exit_price - position.entry_price) * position.qty
                    position.pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100

                    capital += position.pnl
                    result.trades.append(position)

                    # Update strategy stats
                    strategy_stats[position.strategy_used]["trades"] += 1
                    strategy_stats[position.strategy_used]["pnl"] += position.pnl
                    if position.pnl > 0:
                        strategy_stats[position.strategy_used]["wins"] += 1

                    position = None
                else:
                    current_position = {
                        "entry_price": position.entry_price,
                        "qty": position.qty,
                        "entry_time": position.entry_time,
                        "highest_price": current_price,
                    }

            # IMPORTANT: Use the ORIGINAL strategy for exit checks, current strategy for entries
            # This ensures a position is managed by the strategy that opened it
            if position is not None:
                # Use the strategy that opened the position for exit checks
                strategy = self.strategies[position.strategy_used]
                prepared_df = prepared_data[position.strategy_used].iloc[:i+1]
            else:
                # Use the current regime's strategy for new entries
                strategy = self.strategies[active_strategy_name]
                prepared_df = prepared_data[active_strategy_name].iloc[:i+1]

            underlying_df = qqq_prepared.iloc[:i+1]

            # Generate signal from the appropriate strategy
            # Pass VIX value for crisis_alpha strategy
            if active_strategy_name == "crisis_alpha":
                signal = strategy.generate_signal(
                    prepared_df,
                    current_position=current_position,
                    underlying_df=underlying_df,
                    vix_value=self.scenario_params["vix"]
                )
            else:
                signal = strategy.generate_signal(
                    prepared_df,
                    current_position=current_position,
                    underlying_df=underlying_df
                )

            # Execute signals (only if buy_enabled for new entries)
            if signal.signal == SignalType.BUY and position is None and buy_enabled:
                size_mult = self.POSITION_SIZE_MAP.get(signal.position_size, 1.0)
                allocation = capital * self.position_pct * size_mult
                qty = int(allocation / current_price)

                if qty > 0:
                    position = Trade(
                        entry_time=current_time,
                        entry_price=current_price,
                        symbol=active_symbol,
                        qty=qty,
                        entry_reason=signal.reason,
                        strategy_used=active_strategy_name,
                        regime=regime_decision["reasoning"][:30],
                    )

                    # Shadow Options: Open parallel options position
                    if self.enable_shadow_options and self.options_adapter:
                        qqq_price = qqq_df.iloc[i]["Close"]
                        contract = self.options_adapter.translate_signal(
                            symbol=active_symbol,
                            underlying_price=qqq_price,
                            signal_confidence=signal.confidence,
                            current_time=current_time,
                            strategy_name=active_strategy_name,
                            signal_reason=signal.reason,
                        )
                        if contract:
                            self.options_adapter.open_shadow_position(
                                contract=contract,
                                underlying_price=qqq_price,
                                current_time=current_time,
                                original_symbol=active_symbol,
                                signal_reason=signal.reason,
                                strategy_name=active_strategy_name,
                                capital_allocation=1000.0,  # $1000 per options trade
                            )

            elif signal.signal == SignalType.EXIT and position is not None:
                position.exit_time = current_time
                position.exit_price = current_price
                position.exit_reason = signal.reason
                position.pnl = (position.exit_price - position.entry_price) * position.qty
                position.pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100

                capital += position.pnl
                result.trades.append(position)

                # Update strategy stats
                strategy_stats[position.strategy_used]["trades"] += 1
                strategy_stats[position.strategy_used]["pnl"] += position.pnl
                if position.pnl > 0:
                    strategy_stats[position.strategy_used]["wins"] += 1

                position = None

            # Track equity
            current_equity = capital
            if position is not None:
                unrealized = (current_price - position.entry_price) * position.qty
                current_equity += unrealized

            equity_curve.append(current_equity)

            if current_equity > peak_equity:
                peak_equity = current_equity

            drawdown = peak_equity - current_equity
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Close any open position
        if position is not None:
            if position.symbol == "SQQQ":
                exit_price = sqqq_df.iloc[-1]["Close"]
            else:
                exit_price = tqqq_df.iloc[-1]["Close"]

            position.exit_time = tqqq_df.index[-1]
            position.exit_price = exit_price
            position.exit_reason = "END_OF_BACKTEST"
            position.pnl = (position.exit_price - position.entry_price) * position.qty
            position.pnl_pct = ((position.exit_price - position.entry_price) / position.entry_price) * 100
            capital += position.pnl
            result.trades.append(position)

            strategy_stats[position.strategy_used]["trades"] += 1
            strategy_stats[position.strategy_used]["pnl"] += position.pnl
            if position.pnl > 0:
                strategy_stats[position.strategy_used]["wins"] += 1

        # Calculate metrics
        self._calculate_metrics(result, equity_curve, max_drawdown, regime_counts, strategy_stats)

        return result

    def _calculate_metrics(self, result: RegimeBacktestResult, equity_curve: List[float],
                          max_drawdown: float, regime_counts: dict, strategy_stats: dict):
        """Calculate performance metrics."""
        trades = result.trades
        total_bars = sum(regime_counts.values())

        # Regime distribution
        result.regime_distribution = {k: v / total_bars * 100 if total_bars > 0 else 0
                                      for k, v in regime_counts.items()}

        # Strategy stats
        result.strategy_stats = strategy_stats

        if not trades:
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.win_rate = len(wins) / len(trades) * 100 if trades else 0

        result.total_pnl = sum(t.pnl for t in trades)
        result.total_pnl_pct = (result.total_pnl / self.initial_capital) * 100

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        result.max_drawdown = max_drawdown

        # Sharpe ratio
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            if returns.std() > 0:
                result.sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252 * 390)

    def print_report(self, result: RegimeBacktestResult):
        """Print detailed backtest report."""
        print("\n" + "=" * 70)
        print("  REGIME-SWITCHING BACKTEST REPORT")
        print("  Testing ACTUAL Bi-Cameral System Logic")
        print("=" * 70)

        print(f"\n  Scenario: {result.scenario.upper()} - {self.scenario_params['description']}")
        print(f"  Simulated: VIX={self.scenario_params['vix']}, F&G={self.scenario_params['fear_greed']}")
        print(f"  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"  Total Bars: {result.total_bars:,}")

        print("\n" + "-" * 70)
        print("  REGIME DISTRIBUTION (based on ADX + scenario)")
        print("-" * 70)
        for strategy, pct in sorted(result.regime_distribution.items(), key=lambda x: -x[1]):
            if pct > 0:
                print(f"  {strategy:20} {pct:5.1f}%")
        print(f"  Regime Changes: {result.regime_changes}")

        print("\n" + "-" * 70)
        print("  PER-STRATEGY PERFORMANCE")
        print("-" * 70)
        for name, stats in result.strategy_stats.items():
            if stats["trades"] > 0:
                win_rate = stats["wins"] / stats["trades"] * 100
                print(f"  {name:20} {stats['trades']:3} trades | ${stats['pnl']:+8,.2f} | {win_rate:5.1f}% WR")

        print("\n" + "-" * 70)
        print("  OVERALL PERFORMANCE")
        print("-" * 70)
        print(f"  Total Trades: {len(result.trades)}")
        print(f"  Win Rate: {result.win_rate:.1f}% ({result.win_count}W / {result.loss_count}L)")
        print(f"  Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print(f"  Max Drawdown: ${result.max_drawdown:,.2f}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")

        # Trade log
        if result.trades:
            print("\n" + "-" * 70)
            print("  TRADE LOG (Last 10)")
            print("-" * 70)
            for trade in result.trades[-10:]:
                entry_str = trade.entry_time.strftime("%m/%d %H:%M")
                exit_str = trade.exit_time.strftime("%H:%M") if trade.exit_time else "OPEN"
                pnl_str = f"${trade.pnl:+,.2f}" if trade.pnl >= 0 else f"${trade.pnl:,.2f}"
                print(f"  {entry_str} -> {exit_str} | {trade.symbol} [{trade.strategy_used[:12]}] | {pnl_str}")
                print(f"    Entry: {trade.entry_reason[:55]}")
                print(f"    Exit: {trade.exit_reason[:55]}")

        # Verdict
        print("\n" + "=" * 70)
        if result.total_pnl > 0 and result.win_rate >= 50:
            print("  VERDICT: SYSTEM VALIDATED")
        elif result.total_pnl > 0:
            print("  VERDICT: PROFITABLE (but low win rate)")
        else:
            print("  VERDICT: NEEDS OPTIMIZATION")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Regime-Switching Backtester")
    parser.add_argument("--days", type=int, default=90, help="Days of history")
    parser.add_argument("--scenario", type=str, default="normal",
                        choices=["normal", "fearful", "crisis", "greedy", "euphoria"],
                        help="Market scenario to simulate")
    parser.add_argument("--verbose", action="store_true", help="Show regime changes")

    args = parser.parse_args()

    print("=" * 70)
    print("  REGIME-SWITCHING BACKTESTER")
    print("  'Test the ACTUAL Bi-Cameral System'")
    print("=" * 70)
    print(f"\n  Available scenarios: normal, fearful, crisis, greedy, euphoria")

    bt = RegimeBacktester(scenario=args.scenario)
    result = bt.run(days=args.days, verbose=args.verbose)
    bt.print_report(result)


if __name__ == "__main__":
    main()
