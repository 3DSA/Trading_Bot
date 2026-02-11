"""
Volatility Breakout Strategy - "The News Trader"

An Opening Range Breakout (ORB) strategy for capturing morning momentum.
Designed to catch the explosive moves in the first hours of trading.

Best Market Regime: HIGH VOLATILITY / NEWS DAYS
Typical Hold Time: 30-180 minutes
Win Rate Target: 45-55%
Risk/Reward: 1:3

Key Features:
    - Opening Range: Calculates High/Low of first 30 minutes
    - Breakout Confirmation: Requires price break + volume spike
    - ATR Filter: Only trades when volatility is elevated
    - Time-Based Exit: Closes all positions before 15:55
    - Wide Trailing Stops: Allows winners to run

Author: Bi-Cameral Quant Team
"""

from typing import Optional, Tuple
from datetime import time as dt_time
import pandas as pd
import numpy as np

from deprecated.strategies.base import (
    BaseStrategy,
    StrategySignal,
    StrategyConfig,
    SignalType,
    PositionSizing,
)
from strategies.shared_utils import (
    calc_adx,
    calc_atr,
    calc_atr_percent,
    calc_volume_ratio,
    calc_vwap,
    calc_ema,
)


class VolatilityBreakoutStrategy(BaseStrategy):
    """
    The News Trader - Opening Range Breakout for Morning Momentum.

    Setup Phase (09:30-10:00):
        - Calculate Opening Range High (ORH)
        - Calculate Opening Range Low (ORL)
        - Wait for range to establish

    Entry Logic (10:00-14:00):
        1. Price breaks ORH + (0.1 * ATR) = LONG
        2. Volume > 2x average (breakout confirmation)
        3. ATR% > 1.0% (sufficient volatility)
        4. Not during lunch (11:45-13:15)

    Exit Logic:
        1. Trailing Stop: 2x ATR (let winners run)
        2. Time Stop: Close at 15:55 (no overnight risk)
        3. Range Failure: Price returns inside range
        4. Take Profit: 3x ATR from entry

    Risk Management:
        - Only trades breakouts, not fades
        - Wide trailing stop for volatility
        - Full position on confirmed breakouts
    """

    name = "volatility_breakout"
    description = "Opening Range Breakout (ORB) for morning momentum"
    version = "1.0.0"
    preferred_regime = "VOLATILE"

    # Strategy-specific parameters
    ORB_MINUTES = 30  # First 30 minutes
    BREAKOUT_ATR_BUFFER = 0.1  # 10% of ATR above ORH
    VOLUME_SPIKE_THRESHOLD = 2.0
    MIN_ATR_PERCENT = 1.0  # Minimum 1% ATR for sufficient volatility
    TRAILING_ATR_MULTIPLIER = 2.0
    TP_ATR_MULTIPLIER = 3.0
    STOP_ATR_MULTIPLIER = 1.5

    # Time windows
    ORB_START = dt_time(9, 30)
    ORB_END = dt_time(10, 0)
    TRADING_START = dt_time(10, 0)
    TRADING_END = dt_time(14, 0)
    FORCE_CLOSE = dt_time(15, 55)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators and calculate Opening Range."""
        df = df.copy()

        # Volatility
        df["atr"] = calc_atr(df, period=14)
        df["atr_pct"] = calc_atr_percent(df, period=14)

        # Volume
        df["volume_ratio"] = calc_volume_ratio(df, window=20)

        # VWAP for trend bias
        df["vwap"] = calc_vwap(df, reset_daily=True)

        # EMA for trend confirmation
        df["ema_20"] = calc_ema(df["Close"], span=20)

        # ADX for trend strength
        df["adx"] = calc_adx(df, period=14)

        # Calculate Opening Range for each day
        df["orb_high"], df["orb_low"] = self._calculate_orb(df)

        # Calculate breakout levels
        df["breakout_long"] = df["orb_high"] + (df["atr"] * self.BREAKOUT_ATR_BUFFER)
        df["breakout_short"] = df["orb_low"] - (df["atr"] * self.BREAKOUT_ATR_BUFFER)

        # Forward fill
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def _calculate_orb(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Opening Range High/Low for each trading day.

        Returns:
            Tuple of (orb_high, orb_low) Series
        """
        df = df.copy()
        df["date"] = df.index.date
        df["time"] = df.index.time

        # Filter to ORB period (09:30-10:00)
        orb_mask = (df["time"] >= self.ORB_START) & (df["time"] < self.ORB_END)

        # Calculate ORB high/low per day
        orb_data = df[orb_mask].groupby("date").agg({
            "High": "max",
            "Low": "min"
        })
        orb_data.columns = ["orb_high", "orb_low"]

        # Map back to full dataframe
        orb_high = df["date"].map(orb_data["orb_high"])
        orb_low = df["date"].map(orb_data["orb_low"])

        return orb_high, orb_low

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
        underlying_df: Optional[pd.DataFrame] = None
    ) -> StrategySignal:
        """
        Generate breakout signal based on Opening Range.

        Args:
            df: Prepared DataFrame with ORB levels
            current_position: Current position (None if flat)
            underlying_df: Not used for this strategy

        Returns:
            StrategySignal with action and risk parameters
        """
        if df.empty:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason="Insufficient data"
            )

        current = df.iloc[-1]
        current_price = current["Close"]
        current_time = df.index[-1]
        time_only = current_time.time()

        # Check if we're in ORB setup phase (don't trade yet)
        if self.ORB_START <= time_only < self.ORB_END:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"ORB Setup Phase: {time_only.strftime('%H:%M')} - calculating range",
                metadata={
                    "phase": "ORB_SETUP",
                    "current_high": current["High"],
                    "current_low": current["Low"],
                }
            )

        # Force close at end of day
        if time_only >= self.FORCE_CLOSE:
            if current_position is not None:
                return StrategySignal(
                    signal=SignalType.EXIT,
                    reason="Force Close: End of day",
                    confidence=1.0,
                    metadata={"exit_reason": "EOD_CLOSE"}
                )
            return StrategySignal(
                signal=SignalType.HOLD,
                reason="End of day - no new positions"
            )

        # If we have a position, check for exit
        if current_position is not None:
            return self._check_exit(current, current_position, current_time)

        # Check if we're in trading window
        if time_only < self.TRADING_START or time_only >= self.TRADING_END:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Outside breakout window: {time_only.strftime('%H:%M')}"
            )

        # Avoid lunch hour
        lunch_start = dt_time(11, 45)
        lunch_end = dt_time(13, 15)
        if lunch_start <= time_only <= lunch_end:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason="Lunch hour - no breakout trades"
            )

        # Check for entry
        return self._check_entry(current)

    def _check_entry(self, current: pd.Series) -> StrategySignal:
        """Check for breakout entry conditions."""
        current_price = current["Close"]
        atr = current["atr"]

        # Validate ORB levels exist
        if pd.isna(current["orb_high"]) or pd.isna(current["orb_low"]):
            return StrategySignal(
                signal=SignalType.HOLD,
                reason="ORB levels not yet established"
            )

        conditions = {}

        # 1. Check for breakout above ORH + buffer
        breakout_long_level = current["breakout_long"]
        conditions["breakout_above_orh"] = current_price > breakout_long_level

        # 2. Volume confirmation
        conditions["volume_spike"] = current["volume_ratio"] >= self.VOLUME_SPIKE_THRESHOLD

        # 3. Sufficient volatility (ATR% > 1%)
        conditions["sufficient_volatility"] = current["atr_pct"] >= self.MIN_ATR_PERCENT

        # 4. Price above VWAP (institutional bullish bias)
        conditions["above_vwap"] = current_price > current["vwap"]

        # All conditions for LONG breakout
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]

            # Check how close we are to breakout
            distance_to_breakout = (breakout_long_level - current_price) / current_price * 100

            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Breakout blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "orb_high": current["orb_high"],
                    "orb_low": current["orb_low"],
                    "breakout_level": breakout_long_level,
                    "distance_to_breakout_pct": distance_to_breakout,
                    "volume_ratio": current["volume_ratio"],
                    "atr_pct": current["atr_pct"],
                }
            )

        # Calculate risk parameters
        stop_loss = current_price - (atr * self.STOP_ATR_MULTIPLIER)
        take_profit = current_price + (atr * self.TP_ATR_MULTIPLIER)
        trailing_pct = (atr * self.TRAILING_ATR_MULTIPLIER) / current_price

        # Calculate confidence
        confidence = self._calculate_confidence(current)

        return StrategySignal(
            signal=SignalType.BUY,
            reason=f"ORB Breakout: Price ${current_price:.2f} > ORH ${current['orb_high']:.2f} + buffer",
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_pct,
            position_size=PositionSizing.FULL,
            metadata={
                "entry_price": current_price,
                "orb_high": current["orb_high"],
                "orb_low": current["orb_low"],
                "breakout_level": breakout_long_level,
                "atr": atr,
                "atr_pct": current["atr_pct"],
                "volume_ratio": current["volume_ratio"],
                "vwap": current["vwap"],
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: dict,
        current_time
    ) -> StrategySignal:
        """Check exit conditions for breakout position."""
        entry_price = position.get("entry_price", current["Close"])
        current_price = current["Close"]
        atr = current["atr"]

        pnl_pct = (current_price - entry_price) / entry_price

        # Exit 1: Range Failure (price falls back inside ORB)
        orb_mid = (current["orb_high"] + current["orb_low"]) / 2
        if current_price < orb_mid:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Range Failure: Price ${current_price:.2f} < ORB midpoint ${orb_mid:.2f}",
                confidence=0.9,
                metadata={"exit_reason": "RANGE_FAILURE", "pnl_pct": pnl_pct}
            )

        # Exit 2: Price falls below VWAP (momentum loss)
        if current_price < current["vwap"] and pnl_pct < 0.005:  # Only if not in good profit
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Below VWAP: Momentum lost",
                confidence=0.75,
                metadata={"exit_reason": "VWAP_BREAK", "pnl_pct": pnl_pct}
            )

        # Exit 3: Volume dies (breakout failing)
        if current["volume_ratio"] < 0.5 and pnl_pct < 0.01:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Volume Collapse: {current['volume_ratio']:.1f}x < 0.5x average",
                confidence=0.7,
                metadata={"exit_reason": "VOLUME_COLLAPSE", "pnl_pct": pnl_pct}
            )

        # Hold position - breakout still valid
        distance_from_orh = (current_price - current["orb_high"]) / current["orb_high"] * 100
        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"Breakout Active: PnL={pnl_pct*100:.2f}%, +{distance_from_orh:.1f}% from ORH",
            metadata={
                "pnl_pct": pnl_pct,
                "distance_from_orh_pct": distance_from_orh,
                "volume_ratio": current["volume_ratio"],
                "orb_high": current["orb_high"],
            }
        )

    def _calculate_confidence(self, current: pd.Series) -> float:
        """Calculate confidence based on breakout strength."""
        confidence = 0.5

        # Strong volume = higher confidence
        if current["volume_ratio"] > 3.0:
            confidence += 0.20
        elif current["volume_ratio"] > 2.5:
            confidence += 0.15
        elif current["volume_ratio"] > 2.0:
            confidence += 0.10

        # High ATR% = explosive move
        if current["atr_pct"] > 2.0:
            confidence += 0.10
        elif current["atr_pct"] > 1.5:
            confidence += 0.05

        # Price well above VWAP = strong bullish bias
        vwap_distance = (current["Close"] - current["vwap"]) / current["vwap"]
        if vwap_distance > 0.01:  # 1% above VWAP
            confidence += 0.10

        # ADX showing trend developing
        if current["adx"] > 25:
            confidence += 0.05

        return min(confidence, 0.90)
