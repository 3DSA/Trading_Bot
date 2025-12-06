#!/usr/bin/env python3
"""
Reflex Engine - The Bi-Cameral Trading Bot Execution Heart

A high-frequency daemon that listens to market data via WebSocket,
calculates technical indicators (RSI, Bollinger Bands), and executes
trades based on configurable thresholds.

Author: Bi-Cameral System
"""

import asyncio
import json
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


class ReflexEngine:
    """
    The execution engine that processes real-time market data and
    executes trades based on technical signals.
    """

    # Configuration paths
    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config.json"
    DB_PATH = BASE_DIR / "data" / "trades.db"

    # Technical indicator periods
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD_DEV = 2
    WARMUP_BARS = 50

    # Config reload interval (seconds)
    CONFIG_RELOAD_INTERVAL = 60

    def __init__(self):
        """Initialize the Reflex Engine with API clients and state."""
        load_dotenv(self.BASE_DIR / ".env")

        # Load API credentials
        self.api_key = os.getenv("ALPACA_KEY")
        self.api_secret = os.getenv("ALPACA_SECRET")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_KEY and ALPACA_SECRET must be set in .env")

        # Initialize Alpaca clients (Paper Mode)
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=True
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        self.stream_client = StockDataStream(
            api_key=self.api_key,
            secret_key=self.api_secret
        )

        # Initialize state
        self.price_history: deque = deque(maxlen=self.WARMUP_BARS)
        self.has_position = False
        self.entry_price = 0.0
        self.position_qty = 0.0
        self.current_config = self._load_config()
        self.last_config_reload = datetime.now()
        self.last_heartbeat = datetime.now()

        # Database connection
        self.db_conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=WAL;")

        print(f"[INIT] ReflexEngine initialized for symbol: {self.current_config['symbol']}")

    def _load_config(self) -> dict:
        """
        Safely load configuration from config.json.
        Returns default values if read fails (race condition protection).
        """
        default_config = {
            "symbol": "SPY",
            "buy_enabled": False,
            "buy_rsi_threshold": 30,
            "stop_loss_pct": 0.02,
            "dampening_factor": 0.1,
            "last_update": "UNKNOWN"
        }

        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
                return {**default_config, **config}
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            print(f"[WARN] Config read failed ({e}), using defaults")
            return default_config

    def _reconcile_state(self):
        """
        Reconcile internal state with actual Alpaca positions.
        Prevents flying blind on restart.
        """
        try:
            positions = self.trading_client.get_all_positions()
            symbol = self.current_config["symbol"]

            for pos in positions:
                if pos.symbol == symbol:
                    self.has_position = True
                    self.entry_price = float(pos.avg_entry_price)
                    self.position_qty = float(pos.qty)
                    print(f"[RECONCILE] Found existing position: {symbol} @ ${self.entry_price:.2f} x {self.position_qty}")
                    return

            self.has_position = False
            self.entry_price = 0.0
            self.position_qty = 0.0
            print(f"[RECONCILE] No existing position for {symbol}")

        except Exception as e:
            print(f"[ERROR] Position reconciliation failed: {e}")

    async def _warmup_data(self):
        """
        Fetch historical bars to pre-fill price history.
        Solves the 'cold start' problem for technical indicators.
        """
        symbol = self.current_config["symbol"]
        print(f"[WARMUP] Fetching last {self.WARMUP_BARS} bars for {symbol}...")

        try:
            # Request last 50 1-minute bars
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=datetime.now() - timedelta(hours=2),  # Extra buffer for market hours
                limit=self.WARMUP_BARS
            )

            bars = self.data_client.get_stock_bars(request)

            if symbol in bars:
                bar_list = list(bars[symbol])
                for bar in bar_list:
                    self.price_history.append(float(bar.close))

                print(f"[WARMUP] Loaded {len(self.price_history)} historical prices")
                if self.price_history:
                    print(f"[WARMUP] Price range: ${min(self.price_history):.2f} - ${max(self.price_history):.2f}")
            else:
                print(f"[WARN] No historical data returned for {symbol}")

        except Exception as e:
            print(f"[ERROR] Warmup failed: {e}")

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        """
        Calculate RSI (Relative Strength Index) for the given price series.
        Protected against division by zero.
        """
        if len(prices) < self.RSI_PERIOD + 1:
            return 50.0  # Neutral RSI when insufficient data

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        # Use exponential moving average for smoothing
        avg_gain = np.mean(gains[-self.RSI_PERIOD:])
        avg_loss = np.mean(losses[-self.RSI_PERIOD:])

        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_bollinger_bands(self, prices: np.ndarray) -> tuple[float, float, float]:
        """
        Calculate Bollinger Bands (Upper, Middle, Lower).
        Returns (upper, middle, lower) tuple.
        """
        if len(prices) < self.BB_PERIOD:
            # Not enough data - return wide bands that won't trigger
            if len(prices) > 0:
                current = prices[-1]
                return (current * 1.1, current, current * 0.9)
            return (0.0, 0.0, 0.0)

        period_prices = prices[-self.BB_PERIOD:]
        middle = np.mean(period_prices)
        std = np.std(period_prices)

        upper = middle + (self.BB_STD_DEV * std)
        lower = middle - (self.BB_STD_DEV * std)

        return (float(upper), float(middle), float(lower))

    def _calculate_signals(self) -> tuple[str | None, dict]:
        """
        Calculate trading signals based on technical indicators.
        Returns (signal, metrics) where signal is 'BUY', 'SELL', or None.
        """
        if len(self.price_history) < 2:
            return None, {}

        prices = np.array(self.price_history)
        current_price = prices[-1]

        # Calculate indicators
        rsi = self._calculate_rsi(prices)
        upper_bb, middle_bb, lower_bb = self._calculate_bollinger_bands(prices)

        metrics = {
            "price": current_price,
            "rsi": rsi,
            "bb_upper": upper_bb,
            "bb_middle": middle_bb,
            "bb_lower": lower_bb
        }

        config = self.current_config

        # BUY Signal Logic
        if (current_price < lower_bb and
            rsi < config["buy_rsi_threshold"] and
            config["buy_enabled"] and
            not self.has_position):
            return "BUY", metrics

        # SELL Signal Logic
        if self.has_position:
            stop_loss_price = self.entry_price * (1 - config["stop_loss_pct"])

            if current_price > upper_bb or current_price < stop_loss_price:
                return "SELL", metrics

        return None, metrics

    async def _execute_trade(self, signal: str, metrics: dict):
        """
        Execute a trade and log to database.
        """
        symbol = self.current_config["symbol"]
        current_price = metrics["price"]

        try:
            if signal == "BUY":
                # Calculate position size (use available cash, simplified)
                account = self.trading_client.get_account()
                buying_power = float(account.buying_power)

                # Use dampening factor to limit position size
                dampening = self.current_config["dampening_factor"]
                allocation = buying_power * dampening
                qty = int(allocation / current_price)

                if qty < 1:
                    print(f"[TRADE] Insufficient funds for BUY (need ${current_price:.2f}, have ${allocation:.2f})")
                    return

                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                )

                result = self.trading_client.submit_order(order)

                self.has_position = True
                self.entry_price = current_price
                self.position_qty = qty

                print(f"[TRADE] BUY {qty} {symbol} @ ~${current_price:.2f}")
                trigger_reason = f"RSI={metrics['rsi']:.1f}, Price<LowerBB"

                self._log_execution(symbol, "BUY", current_price, qty, trigger_reason, None)

            elif signal == "SELL":
                if self.position_qty <= 0:
                    print(f"[TRADE] No position to sell")
                    return

                order = MarketOrderRequest(
                    symbol=symbol,
                    qty=self.position_qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY
                )

                result = self.trading_client.submit_order(order)

                profit_loss = (current_price - self.entry_price) * self.position_qty

                print(f"[TRADE] SELL {self.position_qty} {symbol} @ ~${current_price:.2f} | P/L: ${profit_loss:.2f}")

                # Determine trigger reason
                stop_loss_price = self.entry_price * (1 - self.current_config["stop_loss_pct"])
                if current_price < stop_loss_price:
                    trigger_reason = f"STOP_LOSS (Entry=${self.entry_price:.2f})"
                else:
                    trigger_reason = f"Price>UpperBB (RSI={metrics['rsi']:.1f})"

                self._log_execution(symbol, "SELL", current_price, self.position_qty, trigger_reason, profit_loss)

                self.has_position = False
                self.entry_price = 0.0
                self.position_qty = 0.0

        except Exception as e:
            print(f"[ERROR] Trade execution failed: {e}")

    def _log_execution(self, symbol: str, side: str, price: float, qty: float,
                       trigger_reason: str, profit_loss: float | None):
        """Log trade execution to SQLite database."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO executions (timestamp, symbol, side, price, qty, trigger_reason, profit_loss)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                symbol,
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
        """Print periodic heartbeat with current state."""
        now = datetime.now()
        if (now - self.last_heartbeat).seconds < 60:
            return

        self.last_heartbeat = now
        symbol = self.current_config["symbol"]

        price = metrics.get("price", 0)
        rsi = metrics.get("rsi", 0)

        print(f"[HEARTBEAT] {symbol} ${price:.2f} | RSI: {rsi:.1f} | "
              f"Pos: {self.has_position} | Buy Enabled: {self.current_config['buy_enabled']}")

    async def _on_bar(self, bar):
        """
        Callback for new bar data from WebSocket stream.
        """
        # Append new price to history
        close_price = float(bar.close)
        self.price_history.append(close_price)

        # Reload config periodically
        now = datetime.now()
        if (now - self.last_config_reload).seconds >= self.CONFIG_RELOAD_INTERVAL:
            self.current_config = self._load_config()
            self.last_config_reload = now

        # Calculate signals
        signal, metrics = self._calculate_signals()

        # Print heartbeat
        self._print_heartbeat(metrics)

        # Execute trade if signal generated
        if signal:
            await self._execute_trade(signal, metrics)

    async def run(self):
        """
        Main entry point - start the execution engine.
        """
        print("=" * 60)
        print("  REFLEX ENGINE - Bi-Cameral Trading Bot")
        print("=" * 60)

        # Step 1: Reconcile state with Alpaca
        self._reconcile_state()

        # Step 2: Warmup historical data
        await self._warmup_data()

        # Step 3: Subscribe to real-time bars
        symbol = self.current_config["symbol"]
        self.stream_client.subscribe_bars(self._on_bar, symbol)

        print(f"\n[START] Listening for {symbol} bar updates...")
        print(f"[CONFIG] Buy Enabled: {self.current_config['buy_enabled']}")
        print(f"[CONFIG] RSI Threshold: {self.current_config['buy_rsi_threshold']}")
        print(f"[CONFIG] Stop Loss: {self.current_config['stop_loss_pct'] * 100}%")
        print("-" * 60)

        # Step 4: Start the stream (runs indefinitely)
        try:
            await self.stream_client._run_forever()
        except KeyboardInterrupt:
            print("\n[SHUTDOWN] Received interrupt, closing...")
        except Exception as e:
            print(f"[ERROR] Stream error: {e}")
            print("[RETRY] Attempting to reconnect in 5 seconds...")
            await asyncio.sleep(5)
            await self.run()  # Recursive reconnect
        finally:
            self.db_conn.close()
            print("[SHUTDOWN] Database connection closed")


async def main():
    """Entry point for the Reflex Engine."""
    engine = ReflexEngine()
    await engine.run()


if __name__ == "__main__":
    asyncio.run(main())
