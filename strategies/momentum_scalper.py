"""
Momentum Scalper Strategy - "The Sniper"

A high-frequency momentum scalping strategy designed for trending markets.
Uses EMA crossover + VWAP + Volume confirmation with Double-Lock ADX filter.

Best Market Regime: TREND (ADX > 25)
Typical Hold Time: 5-60 minutes
Win Rate Target: 55-65%
Risk/Reward: 1:2

Key Features:
    - Double-Lock ADX: Requires BOTH leveraged ETF AND underlying to confirm trend
    - Dynamic ATR Stops: Adapts to current volatility
    - Volume Spike Confirmation: Only enters on institutional activity
    - VWAP Bias: Bullish only above VWAP

Author: Bi-Cameral Quant Team
"""

from typing import Optional
import pandas as pd

from strategies.base import (
    BaseStrategy,
    StrategySignal,
    StrategyConfig,
    SignalType,
    PositionSizing,
)
from strategies.shared_utils import (
    calc_adx,
    calc_atr,
    calc_ema,
    calc_vwap,
    calc_volume_ratio,
    calc_rsi,
    calc_plus_minus_di,
)


class MomentumScalperStrategy(BaseStrategy):
    """
    The Sniper - Momentum Scalping for Trending Markets.

    Entry Logic:
        1. Price > VWAP (institutional bullish bias)
        2. EMA 9 > EMA 21 (short-term uptrend)
        3. Volume > 2x SMA (momentum confirmation)
        4. ADX > 30 (strong trend in leveraged ETF)
        5. QQQ ADX > 25 (Double-Lock confirmation)
        6. +DI > -DI (bullish directional movement)

    Exit Logic:
        1. Take Profit: 2x ATR from entry
        2. Stop Loss: 1.5x ATR from entry
        3. Trailing Stop: 0.5x ATR when in profit
        4. ADX Collapse: ADX drops below 20
        5. Trend Break: EMA 9 < EMA 21 crossover

    Risk Management:
        - Dynamic ATR-based stops adapt to volatility
        - Max position: 10% of portfolio
        - Time filter: Avoid lunch hour and EOD
    """

    name = "momentum_scalper"
    description = "EMA + VWAP momentum scalping with Double-Lock ADX"
    version = "2.0.0"
    preferred_regime = "TREND"

    # Strategy-specific parameters
    EMA_FAST = 9
    EMA_SLOW = 21
    VOLUME_SPIKE_THRESHOLD = 2.0
    ATR_STOP_MULTIPLIER = 2.0  # Wider stop to avoid noise
    ATR_TP_MULTIPLIER = 6.0  # Increased from 3.0 - let winners run
    ATR_TRAILING_MULTIPLIER = 0.5
    RSI_OVERBOUGHT = 80  # Don't buy if already overbought

    # Exit tuning parameters
    ADX_EXIT_THRESHOLD = 18  # Lower than entry (was 20) - give trades room
    MIN_BARS_BEFORE_DI_EXIT = 5  # Don't exit on DI crossover too quickly
    MIN_PROFIT_FOR_DI_EXIT = 0.005  # 0.5% min profit before DI exit allowed
    MIN_PROFIT_FOR_TP = 0.008  # 0.8% minimum take profit (don't exit for tiny gains)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators needed for the strategy."""
        df = df.copy()

        # Trend indicators
        df["ema_fast"] = calc_ema(df["Close"], self.EMA_FAST)
        df["ema_slow"] = calc_ema(df["Close"], self.EMA_SLOW)
        df["adx"] = calc_adx(df, period=14)
        df["+di"], df["-di"] = calc_plus_minus_di(df, period=14)

        # Volatility
        df["atr"] = calc_atr(df, period=14)
        df["atr_pct"] = df["atr"] / df["Close"] * 100

        # Volume
        df["vwap"] = calc_vwap(df, reset_daily=True)
        df["volume_ratio"] = calc_volume_ratio(df, window=20)

        # Momentum
        df["rsi"] = calc_rsi(df["Close"], period=14)

        # Forward fill NaN from warmup
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
        underlying_df: Optional[pd.DataFrame] = None
    ) -> StrategySignal:
        """
        Generate trading signal based on momentum criteria.

        Args:
            df: Prepared DataFrame with indicators
            current_position: Current position (None if flat)
            underlying_df: QQQ DataFrame for Double-Lock

        Returns:
            StrategySignal with action and risk parameters
        """
        if df.empty:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason="Insufficient data"
            )

        # Get latest bar
        current = df.iloc[-1]
        current_price = current["Close"]
        current_time = df.index[-1]

        # Time filter
        if not self.is_trading_time(current_time):
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Outside trading hours: {current_time.strftime('%H:%M')}",
                metadata={"time": str(current_time)}
            )

        # If we have a position, check for exit
        if current_position is not None:
            return self._check_exit(current, current_position, underlying_df)

        # Check for entry
        return self._check_entry(current, underlying_df)

    def _check_entry(
        self,
        current: pd.Series,
        underlying_df: Optional[pd.DataFrame]
    ) -> StrategySignal:
        """Check if entry conditions are met."""
        current_price = current["Close"]
        atr = current["atr"]

        # Collect all conditions
        conditions = {}

        # 1. Price > VWAP (bullish institutional bias)
        conditions["price_above_vwap"] = current_price > current["vwap"]

        # 2. EMA Fast > EMA Slow (uptrend)
        conditions["ema_bullish"] = current["ema_fast"] > current["ema_slow"]

        # 3. Volume spike (momentum confirmation)
        conditions["volume_spike"] = current["volume_ratio"] >= self.VOLUME_SPIKE_THRESHOLD

        # 4. ADX > 30 (strong trend - stricter for leveraged ETFs)
        adx_threshold = self.config.double_lock_leveraged_adx
        conditions["adx_strong"] = current["adx"] > adx_threshold

        # 5. +DI > -DI (bullish directional movement)
        conditions["di_bullish"] = current["+di"] > current["-di"]

        # 6. Not overbought (avoid buying at top)
        conditions["not_overbought"] = current["rsi"] < self.RSI_OVERBOUGHT

        # 7. Double-Lock: QQQ ADX confirmation
        qqq_adx_confirms = True
        qqq_adx = 0
        if self.config.use_double_lock and underlying_df is not None:
            qqq_adx = self._get_underlying_adx(underlying_df, current.name)
            qqq_adx_confirms = qqq_adx > self.config.double_lock_underlying_adx
            conditions["qqq_confirms"] = qqq_adx_confirms

        # Check all conditions
        all_conditions_met = all(conditions.values())

        if not all_conditions_met:
            # Find which condition failed
            failed = [k for k, v in conditions.items() if not v]
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Entry blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "adx": current["adx"],
                    "qqq_adx": qqq_adx,
                    "volume_ratio": current["volume_ratio"],
                    "rsi": current["rsi"],
                }
            )

        # Calculate dynamic stop loss and take profit
        stop_loss = self.get_dynamic_stop_loss(
            current_price, atr, self.ATR_STOP_MULTIPLIER
        )
        take_profit = self.get_dynamic_take_profit(
            current_price, atr, 2.0, self.ATR_STOP_MULTIPLIER
        )
        trailing_pct = (atr * self.ATR_TRAILING_MULTIPLIER) / current_price

        # Calculate confidence based on signal strength
        confidence = self._calculate_confidence(current, qqq_adx)

        return StrategySignal(
            signal=SignalType.BUY,
            reason=f"Sniper Entry: ADX={current['adx']:.1f}, Vol={current['volume_ratio']:.1f}x, RSI={current['rsi']:.0f}",
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_pct,
            position_size=PositionSizing.FULL,
            metadata={
                "entry_price": current_price,
                "atr": atr,
                "adx": current["adx"],
                "qqq_adx": qqq_adx,
                "ema_fast": current["ema_fast"],
                "ema_slow": current["ema_slow"],
                "vwap": current["vwap"],
                "volume_ratio": current["volume_ratio"],
                "rsi": current["rsi"],
                "+di": current["+di"],
                "-di": current["-di"],
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: dict,
        underlying_df: Optional[pd.DataFrame]
    ) -> StrategySignal:
        """Check if exit conditions are met for current position."""
        entry_price = position.get("entry_price", current["Close"])
        entry_time = position.get("entry_time")
        current_price = current["Close"]
        current_time = current.name
        atr = current["atr"]

        pnl_pct = (current_price - entry_price) / entry_price

        # Calculate bars held (approximate)
        bars_held = 0
        if entry_time is not None:
            try:
                time_diff = (current_time - entry_time).total_seconds() / 60  # minutes
                bars_held = int(time_diff)
            except:
                bars_held = 10  # Default assumption

        # Exit condition 0: Take Profit (ATR-based) - let winners run!
        # Use MAX of ATR-based target OR minimum profit threshold
        atr_tp_distance = atr * self.ATR_TP_MULTIPLIER / entry_price
        tp_target = max(atr_tp_distance, self.MIN_PROFIT_FOR_TP)
        if pnl_pct >= tp_target:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Take Profit: +{pnl_pct*100:.2f}% >= {tp_target*100:.2f}% target",
                confidence=0.95,
                metadata={"exit_reason": "TAKE_PROFIT", "pnl_pct": pnl_pct}
            )

        # Exit condition 1: Stop Loss (hard stop)
        atr_sl_distance = atr * self.ATR_STOP_MULTIPLIER / entry_price
        if pnl_pct <= -atr_sl_distance:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Stop Loss: {pnl_pct*100:.2f}% <= -{atr_sl_distance*100:.2f}%",
                confidence=0.95,
                metadata={"exit_reason": "STOP_LOSS", "pnl_pct": pnl_pct}
            )

        # Exit condition 2: ADX Collapse (trend dying) - use lower threshold
        if current["adx"] < self.ADX_EXIT_THRESHOLD:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"ADX Collapse: {current['adx']:.1f} < {self.ADX_EXIT_THRESHOLD}",
                confidence=0.9,
                metadata={"exit_reason": "ADX_COLLAPSE", "pnl_pct": pnl_pct}
            )

        # Exit condition 3: Trend Break (EMA crossover bearish) - only if losing
        if current["ema_fast"] < current["ema_slow"] and pnl_pct < 0:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Trend Break: EMA{self.EMA_FAST} < EMA{self.EMA_SLOW}",
                confidence=0.85,
                metadata={"exit_reason": "TREND_BREAK", "pnl_pct": pnl_pct}
            )

        # Exit condition 4: -DI crosses above +DI (bearish takeover)
        # Only exit if: enough bars held AND enough profit locked in
        if current["-di"] > current["+di"]:
            if bars_held >= self.MIN_BARS_BEFORE_DI_EXIT and pnl_pct >= self.MIN_PROFIT_FOR_DI_EXIT:
                return StrategySignal(
                    signal=SignalType.EXIT,
                    reason=f"Bearish DI Crossover: -DI={current['-di']:.1f} > +DI={current['+di']:.1f}",
                    confidence=0.75,
                    metadata={"exit_reason": "DI_CROSSOVER", "pnl_pct": pnl_pct}
                )

        # Hold position
        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"Holding: PnL={pnl_pct*100:.2f}%, ADX={current['adx']:.1f}",
            metadata={
                "pnl_pct": pnl_pct,
                "adx": current["adx"],
                "ema_fast": current["ema_fast"],
                "ema_slow": current["ema_slow"],
                "bars_held": bars_held,
            }
        )

    def _get_underlying_adx(self, underlying_df: pd.DataFrame, timestamp) -> float:
        """Get QQQ ADX for Double-Lock confirmation."""
        if underlying_df is None or underlying_df.empty:
            return 0

        # Calculate ADX if not present
        if "adx" not in underlying_df.columns:
            underlying_df["adx"] = calc_adx(underlying_df, period=14)

        try:
            # Try exact match
            if timestamp in underlying_df.index:
                return underlying_df.loc[timestamp, "adx"]

            # Find nearest previous
            mask = underlying_df.index <= timestamp
            if mask.any():
                nearest = underlying_df.index[mask][-1]
                return underlying_df.loc[nearest, "adx"]

            return 0
        except Exception:
            return 0

    def _calculate_confidence(self, current: pd.Series, qqq_adx: float) -> float:
        """Calculate signal confidence based on indicator strength."""
        confidence = 0.5  # Base confidence

        # ADX strength bonus
        if current["adx"] > 40:
            confidence += 0.15
        elif current["adx"] > 30:
            confidence += 0.10

        # Volume spike bonus
        if current["volume_ratio"] > 3.0:
            confidence += 0.10
        elif current["volume_ratio"] > 2.5:
            confidence += 0.05

        # RSI sweet spot (not oversold, not overbought)
        if 40 <= current["rsi"] <= 60:
            confidence += 0.10

        # Strong DI spread
        di_spread = current["+di"] - current["-di"]
        if di_spread > 15:
            confidence += 0.10

        # QQQ confirmation bonus
        if qqq_adx > 30:
            confidence += 0.05

        return min(confidence, 0.95)  # Cap at 95%
