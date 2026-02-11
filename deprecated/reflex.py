#!/usr/bin/env python3
"""
Reflex Engine v5 - Universal Strategy Router

A strategy-agnostic execution platform that loads trading strategies dynamically.
The engine itself contains NO trading logic - it only:
1. Listens to config.json for active_strategy
2. Loads the requested strategy from the Strategy Library
3. Executes signals from the strategy

=== SUPPORTED STRATEGIES ===
- momentum_scalper: "The Sniper" - EMA + VWAP for trending markets
- mean_reversion: "The Rubber Band" - Z-Score + BB for choppy markets
- volatility_breakout: "The News Trader" - ORB for morning momentum
- crisis_alpha: "The Bear" - VIX-weighted SQQQ for market turmoil

=== STRATEGY SELECTION ===
The AI Manager (manager.py) writes active_strategy to config.json.
Reflex loads the strategy and executes its signals.

Author: Bi-Cameral System
"""

import asyncio
import json
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestBarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# Import Strategy Library
from strategies.factory import StrategyManager, get_strategy, REGIME_STRATEGY_MAP
from strategies.base import StrategySignal, SignalType, PositionSizing, StrategyConfig


@dataclass
class BarData:
    """Represents OHLCV bar data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class ReflexEngine:
    """
    Universal Strategy Router - Executes signals from any loaded strategy.

    This engine is strategy-agnostic. It:
    1. Reads active_strategy from config.json
    2. Loads the strategy class from the Strategy Library
    3. Converts market data to DataFrames
    4. Calls strategy.generate_signal()
    5. Executes the returned StrategySignal
    """

    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config.json"
    DB_PATH = BASE_DIR / "data" / "trades.db"

    # Data warmup
    WARMUP_BARS = 50
    POLL_INTERVAL = 60

    def __init__(self):
        """Initialize the Reflex Engine v5."""
        load_dotenv(self.BASE_DIR / ".env", override=True)

        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET must be set in .env")

        # Initialize Alpaca clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=True
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        # Load config
        self.config = self._load_config()

        # Symbols from config
        self.symbol = self.config.get("symbol", "TQQQ")
        self.underlying_symbol = self.config.get("qqq_symbol", "QQQ")

        # Strategy Manager
        self._init_strategy_manager()

        # Price history (deques for efficiency)
        self.symbol_bars: deque[BarData] = deque(maxlen=self.WARMUP_BARS)
        self.underlying_bars: deque[BarData] = deque(maxlen=self.WARMUP_BARS)

        # Position state
        self.has_position = False
        self.entry_price = 0.0
        self.position_qty = 0.0
        self.highest_price_since_entry = 0.0
        self.entry_time: Optional[datetime] = None
        self.entry_strategy: Optional[str] = None

        # Risk management from config
        risk = self.config.get("risk_management", {})
        self.use_trailing_stop = risk.get("use_trailing_stop", True)
        self.trailing_stop_pct = risk.get("trailing_stop_pct", 0.003)
        self.max_daily_loss_pct = risk.get("max_daily_loss_pct", 0.02)
        self.cooldown_minutes = risk.get("cooldown_minutes", 10)

        # Position sizing percentages (of buying power)
        self.full_position_pct = 0.10   # 10%
        self.half_position_pct = 0.05   # 5%
        self.quarter_position_pct = 0.025  # 2.5%

        # Daily tracking
        self.daily_pnl = 0.0
        self.daily_pnl_date: Optional[datetime] = None
        self.last_exit_time: Optional[datetime] = None

        # Engine state
        self.running = True
        self.last_config_reload = datetime.now()

        # Database
        self.db_conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=WAL;")

        print(f"[INIT] ReflexEngine v5 - Universal Strategy Router")
        print(f"[INIT] Trading: {self.symbol} | Underlying: {self.underlying_symbol}")

    def _init_strategy_manager(self):
        """Initialize the Strategy Manager with config."""
        strategy_cfg = self.config.get("strategy", {})

        # Build StrategyConfig from config.json
        config = StrategyConfig(
            symbol=self.config.get("symbol", "TQQQ"),
            underlying_symbol=self.config.get("qqq_symbol", "QQQ"),
            use_double_lock=strategy_cfg.get("use_double_lock", True),
            double_lock_leveraged_adx=strategy_cfg.get("double_lock_leveraged_adx", 30),
            double_lock_underlying_adx=strategy_cfg.get("double_lock_underlying_adx", 25),
            adx_trend_threshold=strategy_cfg.get("adx_threshold", 25),
            adx_chop_threshold=strategy_cfg.get("adx_exit_threshold", 20),
        )

        self.strategy_manager = StrategyManager(default_config=config)

        # Load active strategy from config
        active_strategy = self.config.get("active_strategy", "momentum_scalper")
        self._load_strategy(active_strategy)

    def _load_strategy(self, strategy_name: str):
        """Load a strategy by name."""
        try:
            # Convert strategy name to regime for StrategyManager
            regime = self._strategy_to_regime(strategy_name)
            self.strategy_manager.set_regime(regime)
            self.active_strategy_name = strategy_name
            print(f"[STRATEGY] Loaded: {self.strategy_manager.current_strategy.name} "
                  f"v{self.strategy_manager.current_strategy.version}")
        except Exception as e:
            print(f"[ERROR] Failed to load strategy '{strategy_name}': {e}")
            # Fallback to momentum_scalper
            self.strategy_manager.set_regime("TREND")
            self.active_strategy_name = "momentum_scalper"

    def _strategy_to_regime(self, strategy_name: str) -> str:
        """Map strategy name to regime for StrategyManager."""
        strategy_regime_map = {
            "momentum_scalper": "TREND",
            "sniper": "TREND",
            "mean_reversion": "CHOP",
            "rubber_band": "CHOP",
            "volatility_breakout": "VOLATILE",
            "news_trader": "VOLATILE",
            "orb": "VOLATILE",
            "crisis_alpha": "CRISIS",
            "bear": "CRISIS",
            "vix": "CRISIS",
        }
        return strategy_regime_map.get(strategy_name.lower(), "TREND")

    def _load_config(self) -> dict:
        """Load configuration from config.json."""
        default_config = {
            "symbol": "TQQQ",
            "qqq_symbol": "QQQ",
            "buy_enabled": True,
            "active_strategy": "momentum_scalper",
            "strategy": {
                "use_double_lock": True,
                "double_lock_leveraged_adx": 30,
                "double_lock_underlying_adx": 25,
            },
            "risk_management": {
                "use_trailing_stop": True,
                "trailing_stop_pct": 0.003,
                "max_daily_loss_pct": 0.02,
                "cooldown_minutes": 10
            }
        }

        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
                return {**default_config, **config}
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            print(f"[WARN] Config read failed ({e}), using defaults")
            return default_config

    def _reconcile_state(self):
        """Reconcile internal state with Alpaca positions."""
        try:
            positions = self.trading_client.get_all_positions()

            for pos in positions:
                if pos.symbol == self.symbol:
                    self.has_position = True
                    self.entry_price = float(pos.avg_entry_price)
                    self.position_qty = float(pos.qty)
                    self.highest_price_since_entry = float(pos.current_price)
                    print(f"[RECONCILE] Found position: {self.symbol} "
                          f"@ ${self.entry_price:.2f} x {self.position_qty}")
                    return

            self.has_position = False
            self.entry_price = 0.0
            self.position_qty = 0.0
            print(f"[RECONCILE] No existing position for {self.symbol}")

        except Exception as e:
            print(f"[ERROR] Position reconciliation failed: {e}")

    def _warmup_data(self):
        """Fetch historical bars for warmup."""
        symbols = [self.symbol, self.underlying_symbol]
        print(f"[WARMUP] Fetching {self.WARMUP_BARS} bars for {symbols}...")

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=2),
                limit=self.WARMUP_BARS,
                feed=DataFeed.IEX
            )

            bars = self.data_client.get_stock_bars(request)

            # Process symbol bars
            if self.symbol in bars:
                for bar in bars[self.symbol]:
                    bar_data = BarData(
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume)
                    )
                    self.symbol_bars.append(bar_data)
                print(f"[WARMUP] Loaded {len(self.symbol_bars)} {self.symbol} bars")

            # Process underlying bars
            if self.underlying_symbol in bars:
                for bar in bars[self.underlying_symbol]:
                    bar_data = BarData(
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=float(bar.volume)
                    )
                    self.underlying_bars.append(bar_data)
                print(f"[WARMUP] Loaded {len(self.underlying_bars)} {self.underlying_symbol} bars")

        except Exception as e:
            print(f"[ERROR] Warmup failed: {e}")

    def _fetch_latest_bars(self) -> tuple[Optional[BarData], Optional[BarData]]:
        """Fetch latest bars for both symbols."""
        symbols = [self.symbol, self.underlying_symbol]

        try:
            request = StockLatestBarRequest(
                symbol_or_symbols=symbols,
                feed=DataFeed.IEX
            )
            bars = self.data_client.get_stock_latest_bar(request)

            symbol_bar = None
            underlying_bar = None

            if self.symbol in bars:
                b = bars[self.symbol]
                symbol_bar = BarData(
                    timestamp=b.timestamp,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume)
                )

            if self.underlying_symbol in bars:
                b = bars[self.underlying_symbol]
                underlying_bar = BarData(
                    timestamp=b.timestamp,
                    open=float(b.open),
                    high=float(b.high),
                    low=float(b.low),
                    close=float(b.close),
                    volume=float(b.volume)
                )

            return symbol_bar, underlying_bar

        except Exception as e:
            print(f"[ERROR] Failed to fetch bars: {e}")
            return None, None

    def _bars_to_dataframe(self, bars: deque[BarData]) -> pd.DataFrame:
        """Convert bar deque to pandas DataFrame for strategy library."""
        if not bars:
            return pd.DataFrame()

        data = {
            "Open": [b.open for b in bars],
            "High": [b.high for b in bars],
            "Low": [b.low for b in bars],
            "Close": [b.close for b in bars],
            "Volume": [b.volume for b in bars],
        }

        index = pd.DatetimeIndex([b.timestamp for b in bars])
        df = pd.DataFrame(data, index=index)

        return df

    def _build_position_dict(self) -> Optional[dict]:
        """Build position dict for strategy signal generation."""
        if not self.has_position:
            return None

        return {
            "entry_price": self.entry_price,
            "qty": self.position_qty,
            "entry_time": self.entry_time,
            "highest_price": self.highest_price_since_entry,
            "strategy": self.entry_strategy,
        }

    def _generate_signal(self, symbol_bar: BarData, underlying_bar: BarData) -> tuple[Optional[StrategySignal], dict]:
        """
        Generate signal using the active strategy.

        This is where the magic happens - we delegate ALL trading logic
        to the loaded strategy class.
        """
        # Convert bars to DataFrames
        df = self._bars_to_dataframe(self.symbol_bars)
        underlying_df = self._bars_to_dataframe(self.underlying_bars)

        if df.empty:
            return None, {"error": "No data"}

        # Prepare data with strategy-specific indicators
        strategy = self.strategy_manager.current_strategy
        prepared_df = strategy.prepare_data(df)

        # Build position dict
        position = self._build_position_dict()

        # Generate signal from strategy
        signal = strategy.generate_signal(
            prepared_df,
            current_position=position,
            underlying_df=underlying_df
        )

        # Build metrics for logging
        current = prepared_df.iloc[-1] if not prepared_df.empty else None
        metrics = {
            "price": symbol_bar.close,
            "strategy": strategy.name,
            "signal": signal.signal.value,
            "reason": signal.reason,
            "confidence": signal.confidence,
        }

        # Add strategy metadata
        if signal.metadata:
            metrics.update(signal.metadata)

        return signal, metrics

    def _is_cooldown_active(self) -> bool:
        """Check if we're in cooldown period after last trade."""
        if self.last_exit_time is None:
            return False

        cooldown_end = self.last_exit_time + timedelta(minutes=self.cooldown_minutes)
        return datetime.now() < cooldown_end

    def _reset_daily_pnl(self, current_date):
        """Reset daily P&L tracking on new day."""
        if self.daily_pnl_date is None or current_date != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = current_date

    def _execute_signal(self, signal: StrategySignal, metrics: dict):
        """
        Execute a StrategySignal from the active strategy.

        Handles BUY, SELL, EXIT signals with position sizing.
        """
        try:
            current_price = metrics.get("price", 0)

            # ========== BUY SIGNAL ==========
            if signal.signal == SignalType.BUY:
                if self.has_position:
                    # Check for pyramid (adding to position)
                    if "pyramid" in signal.reason.lower():
                        self._execute_pyramid(signal, metrics)
                    return

                # Check kill switch
                if not self.config.get("buy_enabled", True):
                    print(f"[BLOCKED] Kill switch active - {signal.reason}")
                    return

                # Check cooldown
                if self._is_cooldown_active():
                    print(f"[BLOCKED] Cooldown active - {signal.reason}")
                    return

                # Check daily loss limit
                if self.daily_pnl <= -self.max_daily_loss_pct:
                    print(f"[BLOCKED] Daily loss limit hit - {signal.reason}")
                    return

                # Calculate position size
                account = self.trading_client.get_account()
                buying_power = float(account.buying_power)

                # Map PositionSizing to percentage
                size_map = {
                    PositionSizing.FULL: self.full_position_pct,
                    PositionSizing.HALF: self.half_position_pct,
                    PositionSizing.QUARTER: self.quarter_position_pct,
                    PositionSizing.NONE: 0,
                }
                allocation_pct = size_map.get(signal.position_size, self.full_position_pct)
                allocation = buying_power * allocation_pct

                qty = int(allocation / current_price)
                if qty < 1:
                    print(f"[TRADE] Insufficient funds for BUY")
                    return

                # Submit order
                order = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order)

                # Update position state
                self.has_position = True
                self.entry_price = current_price
                self.position_qty = qty
                self.highest_price_since_entry = current_price
                self.entry_time = datetime.now()
                self.entry_strategy = self.active_strategy_name

                # Store stop/take profit from signal
                self.active_stop_loss = signal.stop_loss
                self.active_take_profit = signal.take_profit
                self.active_trailing_pct = signal.trailing_stop_pct

                size_label = signal.position_size.name
                strategy_name = self.strategy_manager.current_strategy.name
                print(f"[TRADE] BUY {qty} {self.symbol} @ ${current_price:.2f} | "
                      f"{strategy_name} | {size_label} | {signal.reason}")

                self._log_execution("BUY", current_price, qty,
                                   f"{strategy_name}|{signal.reason}", None)

            # ========== EXIT/SELL SIGNAL ==========
            elif signal.signal == SignalType.EXIT:
                if not self.has_position or self.position_qty <= 0:
                    return

                order = MarketOrderRequest(
                    symbol=self.symbol,
                    qty=self.position_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )
                self.trading_client.submit_order(order)

                pnl = (current_price - self.entry_price) * self.position_qty
                pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100

                # Track daily P&L
                self.daily_pnl += pnl_pct / 100

                strategy_name = self.entry_strategy or self.active_strategy_name
                print(f"[TRADE] SELL {self.position_qty} {self.symbol} @ ${current_price:.2f} | "
                      f"{strategy_name} | {signal.reason} | P/L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

                self._log_execution("SELL", current_price, self.position_qty,
                                   f"{strategy_name}|{signal.reason}", pnl)

                # Reset position state
                self._reset_position_state()
                self.last_exit_time = datetime.now()

        except Exception as e:
            print(f"[ERROR] Trade execution failed: {e}")

    def _execute_pyramid(self, signal: StrategySignal, metrics: dict):
        """Execute pyramid (add to winning position)."""
        try:
            current_price = metrics.get("price", 0)

            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)

            # Pyramid uses QUARTER sizing
            allocation = buying_power * self.quarter_position_pct
            qty = int(allocation / current_price)

            if qty < 1:
                return

            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order)

            # Update average entry
            total_cost = (self.entry_price * self.position_qty) + (current_price * qty)
            self.position_qty += qty
            self.entry_price = total_cost / self.position_qty

            pyramid_level = metrics.get("pyramid_level", 2)
            print(f"[PYRAMID] +{qty} {self.symbol} @ ${current_price:.2f} | "
                  f"Level {pyramid_level} | {signal.reason}")

            self._log_execution("PYRAMID", current_price, qty, signal.reason, None)

        except Exception as e:
            print(f"[ERROR] Pyramid execution failed: {e}")

    def _reset_position_state(self):
        """Reset all position-related state."""
        self.has_position = False
        self.entry_price = 0.0
        self.position_qty = 0.0
        self.highest_price_since_entry = 0.0
        self.entry_time = None
        self.entry_strategy = None
        self.active_stop_loss = None
        self.active_take_profit = None
        self.active_trailing_pct = None

    def _close_position(self, reason: str = "MANUAL") -> bool:
        """Close existing position for symbol switch or emergency."""
        if not self.has_position or self.position_qty <= 0:
            return False

        try:
            symbol_bar, _ = self._fetch_latest_bars()
            current_price = symbol_bar.close if symbol_bar else self.entry_price

            order = MarketOrderRequest(
                symbol=self.symbol,
                qty=self.position_qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order)

            pnl = (current_price - self.entry_price) * self.position_qty
            pnl_pct = ((current_price - self.entry_price) / self.entry_price) * 100

            print(f"[CLOSE] SELL {self.position_qty} {self.symbol} @ ${current_price:.2f} | "
                  f"{reason} | P/L: ${pnl:.2f} ({pnl_pct:+.2f}%)")

            self._log_execution("SELL", current_price, self.position_qty, reason, pnl)

            self._reset_position_state()
            self.last_exit_time = datetime.now()

            return True

        except Exception as e:
            print(f"[ERROR] Failed to close position: {e}")
            return False

    def _log_execution(self, side: str, price: float, qty: float,
                       trigger_reason: str, profit_loss: Optional[float]):
        """Log trade to database."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO executions (timestamp, symbol, side, price, qty, trigger_reason, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                self.symbol,
                side,
                price,
                qty,
                trigger_reason,
                profit_loss
            ))
            self.db_conn.commit()
        except Exception as e:
            print(f"[ERROR] DB logging failed: {e}")

    def _print_heartbeat(self, metrics: dict):
        """Print status heartbeat."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        price = metrics.get("price", 0)
        strategy_name = metrics.get("strategy", "?")
        signal_type = metrics.get("signal", "HOLD")
        confidence = metrics.get("confidence", 0)
        reason = metrics.get("reason", "")

        # Position info
        if self.has_position:
            pnl_pct = ((price - self.entry_price) / self.entry_price) * 100 if self.entry_price > 0 else 0
            pos_str = f"POS: {self.position_qty}@${self.entry_price:.2f} ({pnl_pct:+.1f}%)"
        else:
            pos_str = "NO_POS"

        # Kill switch indicator
        kill_switch = "" if self.config.get("buy_enabled", True) else " [KILL]"

        print(f"[{timestamp}] {self.symbol} ${price:.2f} | {strategy_name} | "
              f"{signal_type} ({confidence:.0%}) | {pos_str}{kill_switch}")

        # Show reason if not just holding
        if signal_type != "HOLD" and reason:
            print(f"         -> {reason[:60]}")

    def _handle_config_changes(self):
        """Handle dynamic config changes (strategy switch, symbol switch)."""
        new_config = self._load_config()

        # Check for strategy change
        new_strategy = new_config.get("active_strategy", "momentum_scalper")
        if new_strategy != self.active_strategy_name:
            print(f"\n{'='*60}")
            print(f"[SWITCH] Strategy Change: {self.active_strategy_name} -> {new_strategy}")
            print(f"{'='*60}")
            self._load_strategy(new_strategy)

        # Check for symbol change
        new_symbol = new_config.get("symbol", self.symbol)
        if new_symbol != self.symbol:
            print(f"\n{'='*60}")
            print(f"[SWITCH] Symbol Change: {self.symbol} -> {new_symbol}")
            print(f"{'='*60}")

            # Close existing position
            if self.has_position:
                print(f"[SWITCH] Liquidating {self.symbol} before switch...")
                self._close_position(reason="SYMBOL_SWITCH")

            # Update symbol
            old_symbol = self.symbol
            self.symbol = new_symbol

            # Clear data
            self.symbol_bars.clear()
            self.underlying_bars.clear()

            # Warmup new data
            print(f"[SWITCH] Warming up data for {self.symbol}...")
            self._warmup_data()
            self._reconcile_state()

            print(f"[SWITCH] Now trading {self.symbol} (was {old_symbol})")
            print(f"{'='*60}\n")

        self.config = new_config

        # Update risk params
        risk = self.config.get("risk_management", {})
        self.trailing_stop_pct = risk.get("trailing_stop_pct", self.trailing_stop_pct)
        self.max_daily_loss_pct = risk.get("max_daily_loss_pct", self.max_daily_loss_pct)

    async def _poll_loop(self):
        """Main polling loop."""
        poll_count = 0
        last_bar_time = None

        while self.running:
            try:
                # Reload config every 60 seconds
                if (datetime.now() - self.last_config_reload).seconds >= 60:
                    self._handle_config_changes()
                    self.last_config_reload = datetime.now()

                # Fetch latest data
                symbol_bar, underlying_bar = self._fetch_latest_bars()

                if symbol_bar and underlying_bar:
                    # Only process new data
                    if last_bar_time != symbol_bar.timestamp:
                        last_bar_time = symbol_bar.timestamp

                        # Add to history
                        self.symbol_bars.append(symbol_bar)
                        self.underlying_bars.append(underlying_bar)

                        # Reset daily P&L on new day
                        self._reset_daily_pnl(symbol_bar.timestamp.date())

                        # Generate signal from strategy
                        signal, metrics = self._generate_signal(symbol_bar, underlying_bar)

                        # Print heartbeat
                        self._print_heartbeat(metrics)

                        # Execute signal
                        if signal and signal.signal != SignalType.HOLD:
                            self._execute_signal(signal, metrics)

                elif poll_count % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No data - market may be closed")

                poll_count += 1

                # Align to :00 second mark (Tick-Tock with Manager)
                now = datetime.now()
                seconds_remaining = 60 - now.second
                sleep_seconds = seconds_remaining + 0.1

                await asyncio.sleep(sleep_seconds)

            except Exception as e:
                print(f"[ERROR] Poll loop error: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Main entry point."""
        strategy = self.strategy_manager.current_strategy

        print("=" * 70)
        print("  REFLEX ENGINE v5 - UNIVERSAL STRATEGY ROUTER")
        print("  'One Engine, Many Weapons'")
        print("=" * 70)
        print(f"  Trading: {self.symbol} | Underlying: {self.underlying_symbol}")
        print("-" * 70)
        print(f"  Active Strategy: {strategy.name} v{strategy.version}")
        print(f"  Description: {strategy.description}")
        print(f"  Preferred Regime: {strategy.preferred_regime}")
        print("-" * 70)
        print("  Available Strategies:")
        print("    - momentum_scalper (Sniper): Trend following")
        print("    - mean_reversion (Rubber Band): Dip buying")
        print("    - volatility_breakout (News Trader): ORB morning momentum")
        print("    - crisis_alpha (Bear): VIX-weighted SQQQ")
        print("=" * 70)

        self._reconcile_state()
        self._warmup_data()

        print(f"\n[START] Polling every {self.POLL_INTERVAL}s...")
        print(f"[KILL] Manager buy_enabled={self.config.get('buy_enabled', True)}")
        print("-" * 70)

        try:
            await self._poll_loop()
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Received interrupt...")
            self.running = False
        finally:
            self.db_conn.close()
            print("[SHUTDOWN] Complete")


async def main():
    engine = ReflexEngine()
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
