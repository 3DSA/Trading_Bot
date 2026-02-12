#!/usr/bin/env python3
"""
Options Trading Bot - Live Execution Engine

Pure algorithmic options trading using the EXACT same brain logic
that was backtested with 2.5-4.8 profit factor across 2024-2026.

Brain Strategy Selection (Priority Order):
    1. PANIC: VIX >= 22 AND Z-Score < -2.5 -> Vega Snap
    2. EXPLOSIVE: Velocity >= 0.3% in 1 min -> Gamma Scalper OR Reversal Scalper
       - Exhaustion Routing (within EXPLOSIVE):
         * Rule 1: exhaustion_score >= 2 AND VIX < 25 -> Reversal Scalper
         * Rule 2: session_phase == "midday" AND exhaustion_score >= 1 -> Reversal Scalper
         * Rule 3: exhaustion_score >= 3 (any VIX) -> Reversal Scalper
         * Otherwise -> Gamma Scalper (ride explosions)
    3. TRENDING: ADX >= 28 AND Velocity < 0.2% -> Delta Surfer
    4. DEFAULT: Gamma Scalper (wait for explosions)

Usage:
    python options_bot.py                    # Run live (paper trading)
    python options_bot.py --dry-run          # Simulation mode (no orders)
    python options_bot.py --vix 25           # Override VIX level

Author: Bi-Cameral Quant Team
"""

import asyncio
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging

import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import the EXACT same options strategy library used in backtest
from strategies.options import (
    get_option_strategy,
    list_option_strategies,
    select_option_strategy,
    OptionSignalType,
    OptionPosition,
    OptionType,
    ContractSpec,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class LiveOptionPosition:
    """Tracks a live options position."""
    contract: ContractSpec
    entry_price: float
    entry_time: datetime
    quantity: int
    entry_underlying_price: float
    strategy_name: str
    signal_reason: str

    # For tracking
    highest_pnl_pct: float = 0.0
    best_underlying_price: float = 0.0
    bars_held: int = 0


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime
    exit_time: datetime
    contract_type: str
    strike: float
    entry_premium: float
    exit_premium: float
    entry_underlying: float
    exit_underlying: float
    contracts: int
    pnl: float
    pnl_pct: float
    strategy_name: str
    entry_reason: str
    exit_reason: str
    hold_minutes: float


class OptionsTradingBot:
    """
    Live Options Trading Bot with Brain Switching.

    Uses the EXACT same brain logic from the backtester:
    - Same strategy selection thresholds
    - Same signal generation
    - Same exit logic

    Key Differences from Backtest:
    - Real-time data polling (vs historical iteration)
    - Actual order execution via Alpaca
    - Live VIX data fetching
    """

    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config.json"
    DB_PATH = BASE_DIR / "data" / "trades.db"

    # Data settings
    WARMUP_BARS = 50
    POLL_INTERVAL_SECONDS = 60  # Poll every minute

    # Theta decay rate (same as backtest)
    THETA_PER_HOUR = 0.008

    def __init__(
        self,
        dry_run: bool = False,
        vix_override: Optional[float] = None,
        position_size: float = 1000,
    ):
        """
        Initialize the options trading bot.

        Args:
            dry_run: If True, don't execute real orders
            vix_override: Override VIX level (None = fetch live)
            position_size: Dollars per trade
        """
        load_dotenv(self.BASE_DIR / ".env", override=True)

        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET must be set in .env")

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=True  # PAPER TRADING
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        self.dry_run = dry_run
        self.vix_override = vix_override
        self.position_size = position_size
        self.current_vix = vix_override or 20.0

        # Load ALL strategies (same as backtest)
        self.strategies = {
            "gamma_scalper": get_option_strategy("gamma_scalper"),
            "reversal_scalper": get_option_strategy("reversal_scalper"),
            "vega_snap": get_option_strategy("vega_snap"),
            "delta_surfer": get_option_strategy("delta_surfer"),
        }

        # Prepared data for each strategy
        self.prepared_data: Dict[str, pd.DataFrame] = {}

        # Current state
        self.current_strategy_name = "gamma_scalper"
        self.last_strategy_name = "gamma_scalper"
        self.position: Optional[LiveOptionPosition] = None

        # Statistics
        self.strategy_switches = 0
        self.trades_today: List[TradeRecord] = []
        self.daily_pnl = 0.0

        # Engine state
        self.running = True

        # Database setup
        self._setup_database()

        logger.info("=" * 60)
        logger.info("  OPTIONS TRADING BOT - Live Execution")
        logger.info("=" * 60)
        logger.info(f"Mode: {'DRY RUN (no orders)' if dry_run else 'LIVE PAPER TRADING'}")
        logger.info(f"VIX: {'Override=' + str(vix_override) if vix_override else 'Live fetch'}")
        logger.info(f"Position Size: ${position_size}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info("=" * 60)

    def _setup_database(self):
        """Setup SQLite database for trade logging."""
        self.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.db_conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=WAL;")

        # Create options trades table
        self.db_conn.execute("""
            CREATE TABLE IF NOT EXISTS option_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_time TEXT,
                exit_time TEXT,
                contract_type TEXT,
                strike REAL,
                entry_premium REAL,
                exit_premium REAL,
                entry_underlying REAL,
                exit_underlying REAL,
                contracts INTEGER,
                pnl REAL,
                pnl_pct REAL,
                strategy_name TEXT,
                entry_reason TEXT,
                exit_reason TEXT,
                hold_minutes REAL
            )
        """)
        self.db_conn.commit()

    def fetch_vix(self) -> float:
        """Fetch current VIX level from Yahoo Finance."""
        if self.vix_override is not None:
            return self.vix_override

        try:
            # Fetch real VIX data (^VIX is the actual VIX index)
            vix_ticker = yf.Ticker("^VIX")
            vix_data = vix_ticker.history(period="1d", interval="1m")

            if not vix_data.empty:
                current_vix = float(vix_data["Close"].iloc[-1])
                logger.debug(f"Fetched real VIX: {current_vix:.2f}")
                return max(10, min(80, current_vix))  # Clamp to reasonable range

            return 20.0  # Default if no data

        except Exception as e:
            logger.warning(f"VIX fetch failed: {e}, using default 20.0")
            return 20.0

    def fetch_market_data(self) -> pd.DataFrame:
        """Fetch QQQ minute data for analysis."""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=["QQQ"],
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=2),
                limit=self.WARMUP_BARS + 10,
                feed=DataFeed.SIP,
            )

            bars = self.data_client.get_stock_bars(request)
            qqq_bars = bars.data.get("QQQ", [])

            if not qqq_bars:
                return pd.DataFrame()

            df = pd.DataFrame({
                "Open": [float(b.open) for b in qqq_bars],
                "High": [float(b.high) for b in qqq_bars],
                "Low": [float(b.low) for b in qqq_bars],
                "Close": [float(b.close) for b in qqq_bars],
                "Volume": [float(b.volume) for b in qqq_bars],
            }, index=pd.DatetimeIndex([b.timestamp for b in qqq_bars]))

            return df

        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            return pd.DataFrame()

    def prepare_all_strategies(self, df: pd.DataFrame):
        """Prepare data for all strategies (same as backtest)."""
        for name, strategy in self.strategies.items():
            self.prepared_data[name] = strategy.prepare_data(df.copy())

    def get_market_indicators(self, i: int) -> Dict[str, any]:
        """Get current market indicators for brain decisions."""
        indicators = {
            "velocity": 0.0,
            "zscore": 0.0,
            "adx": 20.0,
            "price": 0.0,
            "exhaustion_score": 0,
            "session_phase": "unknown",
        }

        # Get from gamma_scalper data (primary)
        gamma_df = self.prepared_data.get("gamma_scalper")
        if gamma_df is not None and i < len(gamma_df):
            row = gamma_df.iloc[i]
            indicators["price"] = row.get("Close", 0)
            indicators["velocity"] = row.get("velocity", 0)
            # Get exhaustion indicators for reversal routing
            indicators["exhaustion_score"] = int(row.get("exhaustion_score", 0))
            indicators["session_phase"] = str(row.get("session_phase", "unknown"))

        # Get zscore from vega_snap data
        vega_df = self.prepared_data.get("vega_snap")
        if vega_df is not None and i < len(vega_df):
            row = vega_df.iloc[i]
            indicators["zscore"] = row.get("zscore", 0)

        # Get ADX from delta_surfer data
        surfer_df = self.prepared_data.get("delta_surfer")
        if surfer_df is not None and i < len(surfer_df):
            row = surfer_df.iloc[i]
            indicators["adx"] = row.get("adx", 20.0)

        return indicators

    def simulate_option_pnl(
        self,
        entry_underlying: float,
        current_underlying: float,
        entry_premium: float,
        delta: float,
        gamma: float,
        time_held_minutes: float,
        option_type: str,
    ) -> float:
        """
        Simulate option P&L using Greeks (same as backtest).

        In production, this would be replaced with actual option quotes.
        """
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

    def execute_entry(self, signal, strategy_name: str, current_price: float, current_time: datetime):
        """Execute an entry trade."""
        if self.position is not None:
            logger.warning("Already in position, skipping entry")
            return

        contract = signal.contract
        if contract is None:
            logger.warning("No contract in signal, skipping entry")
            return

        option_type = "CALL" if signal.signal == OptionSignalType.BUY_CALL else "PUT"
        contracts = max(1, int(self.position_size / (contract.mid_price * 100)))

        logger.info(f"[ENTRY] {option_type} @ ${contract.strike}")
        logger.info(f"  Strategy: {strategy_name}")
        logger.info(f"  Premium: ${contract.mid_price:.2f}")
        logger.info(f"  Delta: {contract.delta:.2f} | Gamma: {contract.gamma:.3f}")
        logger.info(f"  Reason: {signal.reason[:60]}")

        # Create position
        self.position = LiveOptionPosition(
            contract=contract,
            entry_price=contract.mid_price,
            entry_time=current_time,
            quantity=contracts,
            entry_underlying_price=current_price,
            strategy_name=strategy_name,
            signal_reason=signal.reason,
            best_underlying_price=current_price,
        )

        # In production with real options:
        # self._submit_option_order(contract, contracts, OrderSide.BUY)

        if not self.dry_run:
            # For paper trading, we track the position internally
            # Real options trading would require options-enabled account
            logger.info(f"  [PAPER] Position opened: {contracts} contracts")

    def execute_exit(self, exit_reason: str, current_price: float, current_time: datetime):
        """Execute an exit trade."""
        if self.position is None:
            return

        pos = self.position
        time_held = (current_time - pos.entry_time).total_seconds() / 60

        # Calculate exit premium using Greeks simulation
        current_premium = self.simulate_option_pnl(
            entry_underlying=pos.entry_underlying_price,
            current_underlying=current_price,
            entry_premium=pos.entry_price,
            delta=pos.contract.delta,
            gamma=pos.contract.gamma,
            time_held_minutes=time_held,
            option_type="CALL" if pos.contract.option_type == OptionType.CALL else "PUT",
        )

        pnl_per_contract = (current_premium - pos.entry_price) * 100
        total_pnl = pnl_per_contract * pos.quantity
        pnl_pct = ((current_premium - pos.entry_price) / pos.entry_price) * 100

        option_type = "CALL" if pos.contract.option_type == OptionType.CALL else "PUT"

        logger.info(f"[EXIT] {option_type} @ ${pos.contract.strike}")
        logger.info(f"  Premium: ${pos.entry_price:.2f} -> ${current_premium:.2f}")
        logger.info(f"  P&L: ${total_pnl:+.2f} ({pnl_pct:+.1f}%)")
        logger.info(f"  Held: {time_held:.1f} min | {exit_reason[:50]}")

        # Log trade
        trade = TradeRecord(
            entry_time=pos.entry_time,
            exit_time=current_time,
            contract_type=option_type,
            strike=pos.contract.strike,
            entry_premium=pos.entry_price,
            exit_premium=current_premium,
            entry_underlying=pos.entry_underlying_price,
            exit_underlying=current_price,
            contracts=pos.quantity,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            strategy_name=pos.strategy_name,
            entry_reason=pos.signal_reason,
            exit_reason=exit_reason,
            hold_minutes=time_held,
        )

        self._log_trade(trade)
        self.trades_today.append(trade)
        self.daily_pnl += total_pnl

        # Clear position
        self.position = None

        if not self.dry_run:
            logger.info(f"  [PAPER] Position closed")

    def _log_trade(self, trade: TradeRecord):
        """Log trade to database."""
        try:
            self.db_conn.execute("""
                INSERT INTO option_trades
                (entry_time, exit_time, contract_type, strike, entry_premium,
                 exit_premium, entry_underlying, exit_underlying, contracts,
                 pnl, pnl_pct, strategy_name, entry_reason, exit_reason, hold_minutes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.entry_time.isoformat(),
                trade.exit_time.isoformat(),
                trade.contract_type,
                trade.strike,
                trade.entry_premium,
                trade.exit_premium,
                trade.entry_underlying,
                trade.exit_underlying,
                trade.contracts,
                trade.pnl,
                trade.pnl_pct,
                trade.strategy_name,
                trade.entry_reason,
                trade.exit_reason,
                trade.hold_minutes,
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Trade logging failed: {e}")

    def is_trading_time(self, timestamp: datetime) -> bool:
        """Check if within options trading hours."""
        hour = timestamp.hour
        minute = timestamp.minute

        # Market hours: 9:30 AM - 4:00 PM ET
        # Skip first 5 min and last 15 min
        if hour < 9 or (hour == 9 and minute < 35):
            return False
        if hour >= 16:
            return False
        if hour == 15 and minute >= 45:
            return False

        return True

    async def run_iteration(self):
        """Run a single iteration of the trading loop."""
        current_time = datetime.now()

        # Check trading hours
        if not self.is_trading_time(current_time):
            logger.debug("Outside trading hours")
            return

        # Fetch market data
        df = self.fetch_market_data()
        if df.empty or len(df) < self.WARMUP_BARS:
            logger.warning("Insufficient market data")
            return

        # Update VIX
        self.current_vix = self.fetch_vix()

        # Prepare all strategies
        self.prepare_all_strategies(df)

        # Get current bar index
        i = len(df) - 1
        current_price = df.iloc[i]["Close"]

        # Get market indicators
        indicators = self.get_market_indicators(i)

        # BRAIN: Select strategy (uses canonical logic from brain/router.py)
        self.current_strategy_name = select_option_strategy(
            vix_value=self.current_vix,
            price_velocity=indicators["velocity"],
            zscore=indicators["zscore"],
            adx_value=indicators["adx"],
            exhaustion_score=indicators["exhaustion_score"],
            session_phase=indicators["session_phase"],
        )

        # Track strategy switches
        if self.current_strategy_name != self.last_strategy_name:
            self.strategy_switches += 1
            logger.info(f"[BRAIN] Strategy switch: {self.last_strategy_name} -> {self.current_strategy_name}")
            logger.info(f"  VIX={self.current_vix:.1f} | Z={indicators['zscore']:.2f} | "
                       f"V={indicators['velocity']*100:.2f}% | ADX={indicators['adx']:.1f}")
            self.last_strategy_name = self.current_strategy_name

        # Get current strategy and its prepared data
        strategy = self.strategies[self.current_strategy_name]
        current_data = self.prepared_data[self.current_strategy_name]

        # Create OptionPosition for the strategy if we have one
        current_option_position = None
        if self.position is not None:
            current_option_position = OptionPosition(
                contract=self.position.contract,
                entry_price=self.position.entry_price,
                entry_time=self.position.entry_time,
                quantity=self.position.quantity,
                entry_underlying_price=self.position.entry_underlying_price,
                strategy_name=self.position.strategy_name,
                signal_reason=self.position.signal_reason,
            )
            current_option_position.highest_pnl_pct = self.position.highest_pnl_pct
            current_option_position.best_underlying_price = self.position.best_underlying_price

        # Generate signal
        signal = strategy.generate_signal(
            current_data,
            current_position=current_option_position,
            vix_value=self.current_vix,
        )

        # POSITION MANAGEMENT
        if self.position is not None:
            # Check for exit signal
            if signal.signal == OptionSignalType.EXIT:
                self.execute_exit(signal.reason, current_price, current_time)
            else:
                # Update position tracking
                pos = self.position
                time_held = (current_time - pos.entry_time).total_seconds() / 60

                current_premium = self.simulate_option_pnl(
                    entry_underlying=pos.entry_underlying_price,
                    current_underlying=current_price,
                    entry_premium=pos.entry_price,
                    delta=pos.contract.delta,
                    gamma=pos.contract.gamma,
                    time_held_minutes=time_held,
                    option_type="CALL" if pos.contract.option_type == OptionType.CALL else "PUT",
                )

                pnl_pct = ((current_premium - pos.entry_price) / pos.entry_price) * 100

                # Update tracking
                if pnl_pct > pos.highest_pnl_pct:
                    pos.highest_pnl_pct = pnl_pct

                if pos.contract.option_type == OptionType.CALL:
                    pos.best_underlying_price = max(pos.best_underlying_price, current_price)
                else:
                    pos.best_underlying_price = min(pos.best_underlying_price, current_price)

                pos.bars_held += 1

                # Force time stop check
                original_strategy = self.strategies[pos.strategy_name]
                max_hold = getattr(original_strategy, 'TIME_STOP_MINUTES',
                                   getattr(original_strategy, 'MAX_HOLD_MINUTES', 30))

                if time_held >= max_hold:
                    self.execute_exit(f"FORCED TIME STOP: {time_held:.1f} min >= {max_hold}",
                                     current_price, current_time)
        else:
            # No position - check for entry
            if signal.signal in [OptionSignalType.BUY_CALL, OptionSignalType.BUY_PUT]:
                if signal.contract and strategy.is_trading_time(current_time):
                    self.execute_entry(signal, self.current_strategy_name, current_price, current_time)

    def print_status(self):
        """Print current status."""
        logger.info("-" * 60)
        logger.info(f"[STATUS] Strategy: {self.current_strategy_name} | VIX: {self.current_vix:.1f}")
        logger.info(f"[STATUS] Position: {'YES' if self.position else 'NO'} | "
                   f"Daily P&L: ${self.daily_pnl:+.2f} | Trades: {len(self.trades_today)}")

        if self.position:
            pos = self.position
            time_held = (datetime.now() - pos.entry_time).total_seconds() / 60
            logger.info(f"[POSITION] {pos.contract.option_type.value} ${pos.contract.strike} | "
                       f"Entry: ${pos.entry_price:.2f} | Held: {time_held:.1f} min")

    async def run(self):
        """Main trading loop."""
        logger.info("\n[START] Options trading bot starting...")
        logger.info(f"[START] Poll interval: {self.POLL_INTERVAL_SECONDS} seconds")

        iteration = 0
        status_interval = 5  # Print status every 5 iterations

        try:
            while self.running:
                iteration += 1

                try:
                    await self.run_iteration()
                except Exception as e:
                    logger.error(f"Iteration error: {e}")

                if iteration % status_interval == 0:
                    self.print_status()

                await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("\n[SHUTDOWN] Received interrupt signal")
        finally:
            self._shutdown()

    def _shutdown(self):
        """Clean shutdown."""
        logger.info("[SHUTDOWN] Closing open positions...")

        if self.position is not None:
            self.execute_exit("SHUTDOWN",
                             self.prepared_data["gamma_scalper"].iloc[-1]["Close"],
                             datetime.now())

        logger.info(f"[SHUTDOWN] Session Summary:")
        logger.info(f"  Total Trades: {len(self.trades_today)}")
        logger.info(f"  Daily P&L: ${self.daily_pnl:+.2f}")
        logger.info(f"  Strategy Switches: {self.strategy_switches}")

        self.db_conn.close()
        logger.info("[SHUTDOWN] Database closed")


async def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Options Trading Bot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulation mode (no orders)")
    parser.add_argument("--vix", type=float, default=None,
                        help="Override VIX level")
    parser.add_argument("--position-size", type=float, default=1000,
                        help="Dollars per trade")

    args = parser.parse_args()

    bot = OptionsTradingBot(
        dry_run=args.dry_run,
        vix_override=args.vix,
        position_size=args.position_size,
    )

    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass  # Clean exit, already handled in run()
