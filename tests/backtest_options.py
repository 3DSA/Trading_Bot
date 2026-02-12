#!/usr/bin/env python3
"""
Options-Native Strategy Backtester with Brain Switching

This is the ONLY backtester for the trading bot. We trade OPTIONS ONLY.
The "brain" switches between strategies based on market conditions:

    - VEGA SNAP: When panic conditions detected (VIX spike + price crash) [Priority 1]
    - GAMMA SCALPER: When explosive moves detected (high velocity) [Priority 2]
    - DELTA SURFER: When steady trends detected (ADX > 25, low velocity) [Priority 3]

Key Features:
1. TIME STOPS ARE ENFORCED - Every bar, we check time held
2. GREEKS-BASED P&L - Simulates Delta, Gamma, Theta effects
3. SHORT HOLD TIMES - Minutes, not hours (except Delta Surfer)
4. BRAIN SWITCHING - Automatically selects best strategy

Usage:
    python tests/backtest_options.py --days 30
    python tests/backtest_options.py --days 60 --vix 25 --verbose
    python tests/backtest_options.py --days 30 --strategy gamma_scalper  # Single strategy mode

Author: Bi-Cameral Quant Team
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
import yfinance as yf
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import Options Strategy Library
from strategies.options import (
    get_option_strategy,
    list_option_strategies,
    select_option_strategy,
    OptionStrategyManager,
    GammaScalperStrategy,
    VegaSnapStrategy,
    DeltaSurferStrategy,
    OptionSignalType,
    OptionPosition,
    OptionType,
)


@dataclass
class OptionTrade:
    """Record of a single options trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None

    # Contract details
    contract_type: str = "CALL"  # CALL or PUT
    strike: float = 0.0
    underlying_symbol: str = "QQQ"

    # Prices
    entry_premium: float = 0.0
    exit_premium: float = 0.0
    entry_underlying: float = 0.0
    exit_underlying: float = 0.0

    # Position
    contracts: int = 1

    # P&L
    pnl: float = 0.0
    pnl_pct: float = 0.0

    # Trade info
    strategy_name: str = ""
    entry_reason: str = ""
    exit_reason: str = ""

    # Timing
    hold_minutes: float = 0.0

    # Greeks at entry
    entry_delta: float = 0.50
    entry_gamma: float = 0.08

    # Market conditions at entry
    vix_at_entry: float = 0.0
    velocity_at_entry: float = 0.0
    zscore_at_entry: float = 0.0


@dataclass
class OptionsBacktestResult:
    """Results of options backtest."""
    start_date: datetime
    end_date: datetime
    total_bars: int
    mode: str  # "brain" or single strategy name
    vix_mode: str  # "real" or "simulated"
    vix_stats: Dict = field(default_factory=dict)  # min, max, mean, elevated_days

    trades: List[OptionTrade] = field(default_factory=list)

    # Performance
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # Options-specific metrics
    avg_hold_minutes: float = 0.0
    time_stop_exits: int = 0
    profit_target_exits: int = 0
    stop_loss_exits: int = 0

    # Strategy switching stats (brain mode)
    strategy_switches: int = 0
    strategy_distribution: Dict[str, int] = field(default_factory=dict)
    strategy_pnl: Dict[str, float] = field(default_factory=dict)

    # Leverage
    avg_leverage: float = 0.0


class OptionsBacktester:
    """
    Options-Native Backtester with Brain Switching.

    The brain evaluates each bar and decides:
    1. Which strategy to use (Vega Snap vs Gamma Scalper vs Delta Surfer)
    2. Whether to enter a trade
    3. Whether to exit current position

    Strategy Selection Priority:
    1. VIX >= 22 AND Z-Score < -2.5 -> VEGA SNAP (panic mode)
    2. Velocity >= 0.3% in 1 min -> GAMMA SCALPER (explosion mode)
    3. ADX >= 25 AND Velocity < 0.2% -> DELTA SURFER (trend mode)
    4. Default -> GAMMA SCALPER (wait for explosions)
    """

    BASE_DIR = Path(__file__).parent.parent

    # Theta decay rate per hour for 1-DTE options
    THETA_PER_HOUR = 0.008  # 0.8% per hour

    def __init__(
        self,
        strategy_name: Optional[str] = None,  # None = brain mode
        use_real_vix: bool = True,
        vix_fallback: float = 20.0,
        initial_capital: float = 10000,
        position_size: float = 1000,
    ):
        """
        Initialize options backtester.

        Args:
            strategy_name: Single strategy to test, or None for brain mode
            use_real_vix: If True, fetch real VIX data from Yahoo Finance
            vix_fallback: Fallback VIX level if real data unavailable
            initial_capital: Starting capital
            position_size: $ per trade
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

        self.strategy_name = strategy_name
        self.brain_mode = strategy_name is None
        self.use_real_vix = use_real_vix
        self.vix_fallback = vix_fallback
        self.vix_data = None  # Will be populated with real VIX data
        self.initial_capital = initial_capital
        self.position_size = position_size

        # Load strategies
        if self.brain_mode:
            self.strategies = {
                "gamma_scalper": get_option_strategy("gamma_scalper"),
                "vega_snap": get_option_strategy("vega_snap"),
                "delta_surfer": get_option_strategy("delta_surfer"),
            }
            self.current_strategy_name = "gamma_scalper"
            print(f"[INIT] Options Brain Backtester")
            print(f"[INIT] Mode: BRAIN (auto-switching)")
            print(f"[INIT] Strategies: {list(self.strategies.keys())}")
        else:
            self.strategy = get_option_strategy(strategy_name)
            print(f"[INIT] Options Single Strategy Backtester")
            print(f"[INIT] Strategy: {self.strategy.name} v{self.strategy.version}")

        print(f"[INIT] VIX Mode: {'REAL DATA' if use_real_vix else f'SIMULATED ({vix_fallback})'}")
        print(f"[INIT] Position Size: ${position_size}")

    def fetch_data(self, days: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch QQQ minute data.

        Args:
            days: Number of days of data (used if start_date/end_date not provided)
            start_date: Optional start date string (YYYY-MM-DD)
            end_date: Optional end date string (YYYY-MM-DD)
        """
        if start_date and end_date:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            print(f"\n[DATA] Fetching QQQ data from {start_date} to {end_date}...")
        else:
            end = datetime.now()
            start = end - timedelta(days=days)
            print(f"\n[DATA] Fetching {days} days of QQQ minute data...")

        request = StockBarsRequest(
            symbol_or_symbols=["QQQ"],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.SIP,
        )

        bars = self.data_client.get_stock_bars(request)
        bars_data = bars.data if hasattr(bars, 'data') else {}

        qqq_bars = bars_data.get("QQQ", [])
        df = self._bars_to_df(qqq_bars)

        print(f"[DATA] QQQ: {len(df)} bars")

        return df

    def _bars_to_df(self, bars) -> pd.DataFrame:
        """Convert Alpaca bars to DataFrame."""
        if not bars:
            return pd.DataFrame()

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

    def fetch_vix_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Fetch real VIX historical data from Yahoo Finance.

        Args:
            start_date: Start date for VIX data
            end_date: End date for VIX data

        Returns:
            DataFrame with VIX Close prices indexed by date
        """
        print(f"\n[VIX] Fetching real VIX data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        # Add buffer days for market holidays
        buffer_start = start_date - timedelta(days=5)
        buffer_end = end_date + timedelta(days=1)

        try:
            vix = yf.download(
                "^VIX",
                start=buffer_start.strftime('%Y-%m-%d'),
                end=buffer_end.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=True
            )

            if vix.empty:
                print(f"[VIX] WARNING: No VIX data available, using fallback value {self.vix_fallback}")
                return pd.DataFrame()

            # Handle multi-level columns from yfinance
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)

            # Keep only Close price
            vix_close = vix[['Close']].copy()
            vix_close.columns = ['VIX']

            # Make index timezone-naive for easier merging
            vix_close.index = vix_close.index.tz_localize(None)

            print(f"[VIX] Fetched {len(vix_close)} days of VIX data")
            print(f"[VIX] VIX Range: {vix_close['VIX'].min():.2f} - {vix_close['VIX'].max():.2f}")
            print(f"[VIX] VIX Mean: {vix_close['VIX'].mean():.2f}")

            # Count days with elevated VIX (>= 22)
            elevated_days = (vix_close['VIX'] >= 22).sum()
            print(f"[VIX] Days with VIX >= 22: {elevated_days} ({elevated_days/len(vix_close)*100:.1f}%)")

            return vix_close

        except Exception as e:
            print(f"[VIX] ERROR fetching VIX data: {e}")
            print(f"[VIX] Using fallback value {self.vix_fallback}")
            return pd.DataFrame()

    def get_vix_for_timestamp(self, timestamp: datetime) -> float:
        """
        Get VIX value for a specific timestamp.

        Uses the VIX close from the same trading day.
        Falls back to vix_fallback if no data available.

        Args:
            timestamp: The timestamp to get VIX for

        Returns:
            VIX value
        """
        if self.vix_data is None or self.vix_data.empty:
            return self.vix_fallback

        # Get date from timestamp (handling timezone)
        if timestamp.tzinfo is not None:
            date = timestamp.tz_localize(None).date()
        else:
            date = timestamp.date()

        # Find the most recent VIX data on or before this date
        matching_dates = self.vix_data.index[self.vix_data.index.date <= date]
        if len(matching_dates) == 0:
            return self.vix_fallback

        closest_date = matching_dates[-1]
        return float(self.vix_data.loc[closest_date, 'VIX'])

    def _select_strategy(self, velocity: float, zscore: float, vix: float, adx: float = 20.0) -> str:
        """
        Brain logic: Select the best strategy for current conditions.

        Priority Order:
        1. PANIC: Z-Score < -2.5 + VIX elevated -> Vega Snap (rare, high conviction)
        2. EXPLOSIVE: Velocity > 0.3% -> Gamma Scalper (catch the move)
        3. TRENDING: ADX > 25 + low velocity -> Delta Surfer (ride the trend)
        4. DEFAULT: Gamma Scalper (wait for opportunity)

        Args:
            velocity: Current 1-bar price velocity
            zscore: Current Z-Score
            vix: Current VIX level
            adx: Current ADX value (trend strength)

        Returns:
            Strategy name to use
        """
        # PRIORITY 1: PANIC CONDITIONS -> Vega Snap
        if vix >= 22 and zscore < -2.5:
            return "vega_snap"

        # PRIORITY 2: EXPLOSIVE CONDITIONS -> Gamma Scalper
        if abs(velocity) >= 0.003:  # 0.3% in 1 minute
            return "gamma_scalper"

        # PRIORITY 3: TRENDING CONDITIONS -> Delta Surfer
        # Tighter: ADX >= 28 to match strategy's entry threshold
        if adx >= 28 and abs(velocity) < 0.002:  # Strong trend but not explosive
            return "delta_surfer"

        # DEFAULT: Gamma Scalper (waiting for explosions)
        return "gamma_scalper"

    def simulate_option_pnl(
        self,
        entry_underlying: float,
        current_underlying: float,
        entry_premium: float,
        delta: float,
        gamma: float,
        time_held_minutes: float,
        option_type: str = "CALL",
    ) -> float:
        """Simulate option P&L using Greeks."""
        underlying_move = current_underlying - entry_underlying

        if option_type == "PUT":
            underlying_move = -underlying_move

        # Delta effect
        delta_pnl = delta * underlying_move

        # Gamma effect
        if underlying_move > 0:
            gamma_pnl = 0.5 * gamma * (underlying_move ** 2)
        else:
            gamma_pnl = -0.5 * gamma * (underlying_move ** 2)

        # Theta decay
        hours_held = time_held_minutes / 60
        theta_decay = entry_premium * self.THETA_PER_HOUR * hours_held

        new_premium = entry_premium + delta_pnl + gamma_pnl - theta_decay
        return max(0.01, new_premium)

    def run(self, days: int = 30, warmup_bars: int = 30, verbose: bool = False,
            start_date: Optional[str] = None, end_date: Optional[str] = None) -> OptionsBacktestResult:
        """Run the options backtest."""
        df = self.fetch_data(days, start_date=start_date, end_date=end_date)

        if df.empty or len(df) < warmup_bars:
            raise ValueError("Insufficient data for backtest")

        # Fetch real VIX data if enabled
        vix_stats = {}
        if self.use_real_vix:
            # Get date range from the data
            data_start = df.index[0]
            data_end = df.index[-1]
            if data_start.tzinfo is not None:
                data_start = data_start.tz_localize(None)
                data_end = data_end.tz_localize(None)

            self.vix_data = self.fetch_vix_data(data_start, data_end)
            if not self.vix_data.empty:
                vix_stats = {
                    "min": float(self.vix_data['VIX'].min()),
                    "max": float(self.vix_data['VIX'].max()),
                    "mean": float(self.vix_data['VIX'].mean()),
                    "elevated_days": int((self.vix_data['VIX'] >= 22).sum()),
                    "total_days": len(self.vix_data),
                }

        # Prepare data for all strategies
        print(f"\n[BACKTEST] Preparing indicators...")

        if self.brain_mode:
            prepared_data = {}
            for name, strat in self.strategies.items():
                prepared_data[name] = strat.prepare_data(df.copy())
            # Use gamma_scalper's prepared data for common indicators
            prepared_df = prepared_data["gamma_scalper"]
        else:
            prepared_df = self.strategy.prepare_data(df)
            prepared_data = {self.strategy_name: prepared_df}

        print(f"[BACKTEST] Starting simulation...")
        print(f"[BACKTEST] Period: {prepared_df.index[warmup_bars]} to {prepared_df.index[-1]}")
        print(f"[BACKTEST] Total bars: {len(prepared_df) - warmup_bars}")

        # Initialize result
        mode = "brain" if self.brain_mode else self.strategy_name
        vix_mode = "real" if (self.use_real_vix and self.vix_data is not None and not self.vix_data.empty) else "simulated"
        result = OptionsBacktestResult(
            start_date=prepared_df.index[warmup_bars],
            end_date=prepared_df.index[-1],
            total_bars=len(prepared_df) - warmup_bars,
            mode=mode,
            vix_mode=vix_mode,
            vix_stats=vix_stats,
        )

        if self.brain_mode:
            result.strategy_distribution = {"gamma_scalper": 0, "vega_snap": 0, "delta_surfer": 0}
            result.strategy_pnl = {"gamma_scalper": 0.0, "vega_snap": 0.0, "delta_surfer": 0.0}

        # Simulation state
        capital = self.initial_capital
        position: Optional[OptionTrade] = None
        current_option_position: Optional[OptionPosition] = None

        # Brain state
        current_strategy_name = "gamma_scalper" if self.brain_mode else self.strategy_name
        last_strategy_name = current_strategy_name

        # Progress
        total_iterations = len(prepared_df) - warmup_bars
        last_pct = 0

        for i in range(warmup_bars, len(prepared_df)):
            pct = int((i - warmup_bars) / total_iterations * 100)
            if pct >= last_pct + 10:
                print(f"[BACKTEST] Progress: {pct}%")
                last_pct = pct

            current_time = prepared_df.index[i]
            current_row = prepared_df.iloc[i]
            current_price = current_row["Close"]

            # Get market indicators for brain decisions
            velocity = current_row.get("velocity", 0)
            zscore = current_row.get("zscore", 0) if "zscore" in current_row else 0
            adx = current_row.get("adx", 20.0) if "adx" in current_row else 20.0

            # If no zscore in gamma_scalper, calculate from vega_snap data
            if self.brain_mode and "zscore" not in current_row:
                vega_df = prepared_data["vega_snap"]
                if i < len(vega_df):
                    vega_row = vega_df.iloc[i]
                    zscore = vega_row.get("zscore", 0)

            # If no adx in current data, calculate from delta_surfer data
            if self.brain_mode and "adx" not in current_row:
                surfer_df = prepared_data["delta_surfer"]
                if i < len(surfer_df):
                    surfer_row = surfer_df.iloc[i]
                    adx = surfer_row.get("adx", 20.0)

            # Get current VIX value (real or fallback)
            current_vix = self.get_vix_for_timestamp(current_time)

            # BRAIN: Select strategy for this bar
            if self.brain_mode:
                current_strategy_name = self._select_strategy(
                    velocity=velocity,
                    zscore=zscore,
                    vix=current_vix,
                    adx=adx,
                )

                # Track strategy switches
                if current_strategy_name != last_strategy_name:
                    result.strategy_switches += 1
                    if verbose:
                        print(f"  [{current_time.strftime('%m/%d %H:%M')}] BRAIN SWITCH: {last_strategy_name} -> {current_strategy_name}")
                    last_strategy_name = current_strategy_name

                result.strategy_distribution[current_strategy_name] += 1
                strategy = self.strategies[current_strategy_name]
                current_data = prepared_data[current_strategy_name].iloc[:i+1]
            else:
                strategy = self.strategy
                current_data = prepared_df.iloc[:i+1]

            # Generate signal
            signal = strategy.generate_signal(
                current_data,
                current_position=current_option_position,
                vix_value=current_vix,
            )

            # POSITION MANAGEMENT
            if position is not None:
                time_held = (current_time - position.entry_time).total_seconds() / 60

                # Simulate current option price
                current_premium = self.simulate_option_pnl(
                    entry_underlying=position.entry_underlying,
                    current_underlying=current_price,
                    entry_premium=position.entry_premium,
                    delta=position.entry_delta,
                    gamma=position.entry_gamma,
                    time_held_minutes=time_held,
                    option_type=position.contract_type,
                )

                pnl_pct = (current_premium - position.entry_premium) / position.entry_premium

                # Check exit
                should_exit = False
                exit_reason = ""

                if signal.signal == OptionSignalType.EXIT:
                    should_exit = True
                    exit_reason = signal.reason

                # Force time stop based on the strategy that opened the position
                original_strategy = self.strategies[position.strategy_name] if self.brain_mode else strategy
                max_hold = getattr(original_strategy, 'TIME_STOP_MINUTES',
                                   getattr(original_strategy, 'MAX_HOLD_MINUTES', 30))
                if time_held >= max_hold and not should_exit:
                    should_exit = True
                    exit_reason = f"FORCED TIME STOP: {time_held:.1f} min >= {max_hold}"

                if should_exit:
                    # Close position
                    position.exit_time = current_time
                    position.exit_premium = current_premium
                    position.exit_underlying = current_price
                    position.exit_reason = exit_reason
                    position.hold_minutes = time_held

                    pnl_per_contract = (current_premium - position.entry_premium) * 100
                    position.pnl = pnl_per_contract * position.contracts
                    position.pnl_pct = pnl_pct * 100

                    capital += position.pnl
                    result.trades.append(position)

                    # Track by strategy (brain mode)
                    if self.brain_mode:
                        result.strategy_pnl[position.strategy_name] += position.pnl

                    # Track exit reasons
                    exit_upper = exit_reason.upper()
                    if "TIME" in exit_upper:
                        result.time_stop_exits += 1
                    elif "PROFIT" in exit_upper or "SNAP" in exit_upper:
                        result.profit_target_exits += 1
                    elif "STOP" in exit_upper or "LOSS" in exit_upper:
                        result.stop_loss_exits += 1

                    if verbose:
                        pnl_str = f"+${position.pnl:.2f}" if position.pnl >= 0 else f"${position.pnl:.2f}"
                        print(f"  [{current_time.strftime('%m/%d %H:%M')}] EXIT {position.contract_type} [{position.strategy_name}]")
                        print(f"    Premium: ${position.entry_premium:.2f} -> ${current_premium:.2f}")
                        print(f"    P&L: {pnl_str} ({position.pnl_pct:+.1f}%)")
                        print(f"    Held: {time_held:.1f} min | {exit_reason[:50]}")

                    position = None
                    current_option_position = None

            else:
                # NO POSITION - check for entry
                if signal.signal in [OptionSignalType.BUY_CALL, OptionSignalType.BUY_PUT]:
                    contract = signal.contract

                    if contract and strategy.is_trading_time(current_time):
                        contracts = max(1, int(self.position_size / (contract.mid_price * 100)))
                        option_type = "CALL" if signal.signal == OptionSignalType.BUY_CALL else "PUT"

                        position = OptionTrade(
                            entry_time=current_time,
                            contract_type=option_type,
                            strike=contract.strike,
                            underlying_symbol="QQQ",
                            entry_premium=contract.mid_price,
                            entry_underlying=current_price,
                            contracts=contracts,
                            strategy_name=current_strategy_name,
                            entry_reason=signal.reason,
                            entry_delta=contract.delta,
                            entry_gamma=contract.gamma,
                            vix_at_entry=current_vix,
                            velocity_at_entry=velocity,
                            zscore_at_entry=zscore,
                        )

                        current_option_position = OptionPosition(
                            contract=contract,
                            entry_price=contract.mid_price,
                            entry_time=current_time,
                            quantity=contracts,
                            entry_underlying_price=current_price,
                            strategy_name=current_strategy_name,
                            signal_reason=signal.reason,
                        )

                        if verbose:
                            print(f"  [{current_time.strftime('%m/%d %H:%M')}] BUY {option_type} [{current_strategy_name}]")
                            print(f"    Strike: ${contract.strike} | Premium: ${contract.mid_price:.2f}")
                            print(f"    Delta: {contract.delta:.2f} | Gamma: {contract.gamma:.3f}")
                            print(f"    Reason: {signal.reason[:60]}")

        # Close any open position at end
        if position is not None:
            current_time = prepared_df.index[-1]
            current_price = prepared_df.iloc[-1]["Close"]
            time_held = (current_time - position.entry_time).total_seconds() / 60

            current_premium = self.simulate_option_pnl(
                entry_underlying=position.entry_underlying,
                current_underlying=current_price,
                entry_premium=position.entry_premium,
                delta=position.entry_delta,
                gamma=position.entry_gamma,
                time_held_minutes=time_held,
                option_type=position.contract_type,
            )

            position.exit_time = current_time
            position.exit_premium = current_premium
            position.exit_underlying = current_price
            position.exit_reason = "END_OF_BACKTEST"
            position.hold_minutes = time_held

            pnl_per_contract = (current_premium - position.entry_premium) * 100
            position.pnl = pnl_per_contract * position.contracts
            position.pnl_pct = ((current_premium - position.entry_premium) / position.entry_premium) * 100

            capital += position.pnl
            result.trades.append(position)

            if self.brain_mode:
                result.strategy_pnl[position.strategy_name] += position.pnl

        # Calculate metrics
        self._calculate_metrics(result)

        return result

    def _calculate_metrics(self, result: OptionsBacktestResult):
        """Calculate performance metrics."""
        trades = result.trades

        if not trades:
            return

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        result.win_count = len(wins)
        result.loss_count = len(losses)
        result.win_rate = len(wins) / len(trades) * 100 if trades else 0

        result.total_pnl = sum(t.pnl for t in trades)
        result.total_pnl_pct = (result.total_pnl / self.initial_capital) * 100

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit

        result.avg_hold_minutes = sum(t.hold_minutes for t in trades) / len(trades)

        # Average leverage
        leverages = []
        for t in trades:
            if t.entry_underlying > 0 and abs(t.exit_underlying - t.entry_underlying) > 0.01:
                underlying_pct = (t.exit_underlying - t.entry_underlying) / t.entry_underlying * 100
                if t.contract_type == "PUT":
                    underlying_pct = -underlying_pct
                if abs(underlying_pct) > 0.01:
                    leverage = t.pnl_pct / underlying_pct
                    leverages.append(abs(leverage))

        result.avg_leverage = sum(leverages) / len(leverages) if leverages else 0

    def print_report(self, result: OptionsBacktestResult):
        """Print detailed backtest report."""
        print("\n" + "=" * 70)
        print("  OPTIONS BRAIN BACKTEST REPORT")
        print("  Testing Options-Native Strategy Switching")
        print("=" * 70)

        print(f"\n  Mode: {result.mode.upper()}")
        print(f"  VIX Mode: {result.vix_mode.upper()}")
        if result.vix_stats:
            print(f"  VIX Range: {result.vix_stats['min']:.1f} - {result.vix_stats['max']:.1f} (Mean: {result.vix_stats['mean']:.1f})")
            print(f"  VIX >= 22 Days: {result.vix_stats['elevated_days']} / {result.vix_stats['total_days']}")
        print(f"  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        print(f"  Total Bars: {result.total_bars:,}")

        # Brain stats
        if result.mode == "brain":
            print("\n" + "-" * 70)
            print("  BRAIN STRATEGY SELECTION")
            print("-" * 70)
            total_bars = sum(result.strategy_distribution.values())
            for strat, count in result.strategy_distribution.items():
                pct = count / total_bars * 100 if total_bars > 0 else 0
                pnl = result.strategy_pnl.get(strat, 0)
                print(f"  {strat:20} {pct:5.1f}% of time | P&L: ${pnl:+,.2f}")
            print(f"  Strategy Switches: {result.strategy_switches}")

        print("\n" + "-" * 70)
        print("  PERFORMANCE SUMMARY")
        print("-" * 70)
        print(f"  Total Trades: {len(result.trades)}")
        print(f"  Win Rate: {result.win_rate:.1f}% ({result.win_count}W / {result.loss_count}L)")
        print(f"  Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        print("\n" + "-" * 70)
        print("  OPTIONS-SPECIFIC METRICS")
        print("-" * 70)
        print(f"  Avg Hold Time: {result.avg_hold_minutes:.1f} minutes")
        print(f"  Avg Leverage: {result.avg_leverage:.1f}x")
        print(f"  Exit Breakdown:")
        print(f"    - Profit Target: {result.profit_target_exits}")
        print(f"    - Stop Loss: {result.stop_loss_exits}")
        print(f"    - Time Stop: {result.time_stop_exits}")

        # Per-strategy breakdown in brain mode
        if result.mode == "brain" and result.trades:
            print("\n" + "-" * 70)
            print("  PER-STRATEGY PERFORMANCE")
            print("-" * 70)
            for strat_name in ["gamma_scalper", "vega_snap", "delta_surfer"]:
                strat_trades = [t for t in result.trades if t.strategy_name == strat_name]
                if strat_trades:
                    strat_wins = len([t for t in strat_trades if t.pnl > 0])
                    strat_pnl = sum(t.pnl for t in strat_trades)
                    strat_wr = strat_wins / len(strat_trades) * 100
                    avg_hold = sum(t.hold_minutes for t in strat_trades) / len(strat_trades)
                    print(f"  {strat_name:20} {len(strat_trades):3} trades | ${strat_pnl:+8,.2f} | {strat_wr:5.1f}% WR | {avg_hold:.0f}min avg")

        # Trade log
        if result.trades:
            print("\n" + "-" * 70)
            print("  TRADE LOG (Last 15)")
            print("-" * 70)
            for trade in result.trades[-15:]:
                entry_str = trade.entry_time.strftime("%m/%d %H:%M")
                exit_str = trade.exit_time.strftime("%H:%M") if trade.exit_time else "OPEN"
                pnl_str = f"${trade.pnl:+,.2f}" if trade.pnl >= 0 else f"${trade.pnl:,.2f}"
                pnl_pct_str = f"{trade.pnl_pct:+.1f}%"
                strat_short = trade.strategy_name[:6]

                print(f"  {entry_str} -> {exit_str} | {trade.contract_type} ${trade.strike:.0f} [{strat_short}]")
                print(f"    QQQ: ${trade.entry_underlying:.2f} -> ${trade.exit_underlying:.2f}")
                print(f"    Premium: ${trade.entry_premium:.2f} -> ${trade.exit_premium:.2f}")
                print(f"    P&L: {pnl_str} ({pnl_pct_str}) | {trade.hold_minutes:.1f}min")
                print(f"    Exit: {trade.exit_reason[:50]}")

        # Verdict
        print("\n" + "=" * 70)
        if result.total_pnl > 0 and result.win_rate >= 45:
            if result.avg_hold_minutes <= 15:
                print("  VERDICT: OPTIONS BRAIN VALIDATED")
                print("  (Profitable with proper time discipline)")
            else:
                print("  VERDICT: PROFITABLE (but holding too long)")
        elif result.total_pnl > 0:
            print("  VERDICT: PROFITABLE (but review win rate)")
        else:
            print("  VERDICT: NEEDS OPTIMIZATION")
            if result.time_stop_exits > result.profit_target_exits:
                print("  (Too many time stops - entries not explosive enough)")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Options Brain Backtester")
    parser.add_argument("--days", type=int, default=30, help="Days of history (ignored if --start/--end provided)")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["gamma_scalper", "vega_snap", "delta_surfer"],
                        help="Single strategy mode (omit for brain mode)")
    parser.add_argument("--no-real-vix", action="store_true",
                        help="Disable real VIX data (use simulated)")
    parser.add_argument("--vix-fallback", type=float, default=20.0,
                        help="Fallback VIX level if real data unavailable")
    parser.add_argument("--verbose", action="store_true", help="Show all trades")
    parser.add_argument("--capital", type=float, default=10000,
                        help="Starting capital")

    args = parser.parse_args()

    print("=" * 70)
    print("  OPTIONS BRAIN BACKTESTER")
    print("  'Respect the Greeks, Switch with the Market'")
    print("=" * 70)
    print(f"\n  Available strategies: {list_option_strategies()}")
    print(f"  Mode: {'BRAIN (auto-switch)' if args.strategy is None else args.strategy}")
    print(f"  VIX: {'REAL DATA' if not args.no_real_vix else f'SIMULATED ({args.vix_fallback})'}")

    bt = OptionsBacktester(
        strategy_name=args.strategy,
        use_real_vix=not args.no_real_vix,
        vix_fallback=args.vix_fallback,
        initial_capital=args.capital,
    )
    result = bt.run(days=args.days, verbose=args.verbose,
                    start_date=args.start, end_date=args.end)
    bt.print_report(result)


if __name__ == "__main__":
    main()
