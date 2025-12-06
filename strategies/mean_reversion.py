"""
Mean Reversion Strategy - "The Rubber Band"

A statistical mean reversion strategy for choppy/ranging markets.
Uses Z-Score and Bollinger Bands to identify oversold conditions.

Best Market Regime: CHOP (ADX < 20)
Typical Hold Time: 15-120 minutes
Win Rate Target: 60-70%
Risk/Reward: 1:1.5

Key Features:
    - Z-Score Entry: Only buys at statistically extreme levels (-2 sigma)
    - RSI Confluence: Requires momentum oversold confirmation
    - Bollinger Band Squeeze: Identifies low volatility before reversal
    - Fixed Stops: Tight stops for knife-catching protection
    - Mean Exit: Exits when price returns to SMA (mean)

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
    calc_zscore,
    calc_rsi,
    calc_sma,
    calc_bollinger_bands,
    calc_volume_ratio,
    calc_squeeze_indicator,
)


class MeanReversionStrategy(BaseStrategy):
    """
    The Rubber Band - Mean Reversion for Choppy Markets.

    Entry Logic:
        1. Z-Score < -2.0 (price is 2 standard deviations below mean)
        2. RSI < 30 (momentum is oversold)
        3. ADX < 20 (no trend - chop mode)
        4. Price < Lower Bollinger Band (volatility confirmation)
        5. Volume > 1.5x average (selling exhaustion)

    Exit Logic:
        1. Take Profit: Price crosses above SMA20 (mean)
        2. Stop Loss: Fixed 1% (knife-catching protection)
        3. Z-Score > 0 (returned to mean)
        4. RSI > 50 (momentum neutralized)
        5. Time Stop: 2 hours max hold

    Risk Management:
        - Half position sizing (high risk of "catching falling knife")
        - Fixed 1% stop loss (not ATR-based, we need hard floor)
        - Quick exit at mean (don't get greedy)
    """

    name = "mean_reversion"
    description = "Z-Score + Bollinger mean reversion for chop markets"
    version = "1.0.0"
    preferred_regime = "CHOP"

    # Strategy-specific parameters
    ZSCORE_WINDOW = 20
    ZSCORE_ENTRY_THRESHOLD = -2.0  # 2 sigma oversold
    ZSCORE_EXIT_THRESHOLD = 0.0   # Back to mean
    RSI_OVERSOLD = 30
    RSI_NEUTRAL = 50
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0
    VOLUME_CONFIRMATION = 1.5
    MAX_HOLD_MINUTES = 120
    FIXED_STOP_LOSS_PCT = 0.01  # 1% hard stop

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators needed for mean reversion."""
        df = df.copy()

        # Statistical indicators
        df["zscore"] = calc_zscore(df["Close"], window=self.ZSCORE_WINDOW)
        df["sma20"] = calc_sma(df["Close"], window=20)

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = calc_bollinger_bands(
            df["Close"], window=self.BOLLINGER_PERIOD, num_std=self.BOLLINGER_STD
        )
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # Trend strength (we want LOW ADX for mean reversion)
        df["adx"] = calc_adx(df, period=14)

        # Momentum
        df["rsi"] = calc_rsi(df["Close"], period=14)

        # Volume
        df["volume_ratio"] = calc_volume_ratio(df, window=20)

        # Volatility
        df["atr"] = calc_atr(df, period=14)

        # Squeeze indicator (low vol = good for mean reversion)
        df["squeeze_on"], df["squeeze_momentum"] = calc_squeeze_indicator(df)

        # Forward fill NaN
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
        Generate mean reversion signal.

        Args:
            df: Prepared DataFrame with indicators
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

        # Time filter
        if not self.is_trading_time(current_time):
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Outside trading hours: {current_time.strftime('%H:%M')}",
            )

        # If we have a position, check for exit
        if current_position is not None:
            return self._check_exit(current, current_position, current_time)

        # Check for entry
        return self._check_entry(current)

    def _check_entry(self, current: pd.Series) -> StrategySignal:
        """Check mean reversion entry conditions."""
        current_price = current["Close"]

        conditions = {}

        # 1. Z-Score deeply oversold (2 sigma event)
        conditions["zscore_oversold"] = current["zscore"] < self.ZSCORE_ENTRY_THRESHOLD

        # 2. RSI oversold (momentum confirmation)
        conditions["rsi_oversold"] = current["rsi"] < self.RSI_OVERSOLD

        # 3. ADX low (no trend - we want chop)
        conditions["adx_chop"] = current["adx"] < self.config.adx_chop_threshold

        # 4. Price below lower Bollinger Band
        conditions["below_bb_lower"] = current_price < current["bb_lower"]

        # 5. Volume confirmation (selling exhaustion)
        conditions["volume_confirmed"] = current["volume_ratio"] >= self.VOLUME_CONFIRMATION

        # All conditions must be met
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Mean Rev blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "zscore": current["zscore"],
                    "rsi": current["rsi"],
                    "adx": current["adx"],
                }
            )

        # Calculate exits
        stop_loss = current_price * (1 - self.FIXED_STOP_LOSS_PCT)
        take_profit = current["sma20"]  # Exit at mean

        # Confidence based on how extreme the Z-Score is
        confidence = self._calculate_confidence(current)

        return StrategySignal(
            signal=SignalType.BUY,
            reason=f"Rubber Band: Z={current['zscore']:.2f}, RSI={current['rsi']:.0f}, ADX={current['adx']:.1f}",
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=PositionSizing.HALF,  # Half size for safety
            metadata={
                "entry_price": current_price,
                "zscore": current["zscore"],
                "rsi": current["rsi"],
                "adx": current["adx"],
                "bb_lower": current["bb_lower"],
                "bb_position": current["bb_position"],
                "sma20": current["sma20"],
                "volume_ratio": current["volume_ratio"],
                "target_mean": current["sma20"],
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: dict,
        current_time
    ) -> StrategySignal:
        """Check mean reversion exit conditions."""
        entry_price = position.get("entry_price", current["Close"])
        entry_time = position.get("entry_time")
        current_price = current["Close"]

        pnl_pct = (current_price - entry_price) / entry_price

        # Exit 1: Price returned to mean (Z-Score >= 0)
        if current["zscore"] >= self.ZSCORE_EXIT_THRESHOLD:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Mean Achieved: Z-Score={current['zscore']:.2f} >= 0",
                confidence=0.9,
                metadata={"exit_reason": "MEAN_REVERSION_COMPLETE", "pnl_pct": pnl_pct}
            )

        # Exit 2: Price above SMA20 (mean)
        if current_price > current["sma20"]:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Above Mean: Price > SMA20 (${current['sma20']:.2f})",
                confidence=0.85,
                metadata={"exit_reason": "ABOVE_MEAN", "pnl_pct": pnl_pct}
            )

        # Exit 3: RSI neutralized
        if current["rsi"] > self.RSI_NEUTRAL and pnl_pct > 0:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"RSI Neutral: {current['rsi']:.0f} > 50",
                confidence=0.75,
                metadata={"exit_reason": "RSI_NEUTRAL", "pnl_pct": pnl_pct}
            )

        # Exit 4: ADX starts trending (regime change)
        if current["adx"] > 25:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Regime Change: ADX={current['adx']:.1f} > 25 (trending now)",
                confidence=0.8,
                metadata={"exit_reason": "REGIME_CHANGE", "pnl_pct": pnl_pct}
            )

        # Exit 5: Time stop (max hold time)
        if entry_time:
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            if hold_minutes > self.MAX_HOLD_MINUTES:
                return StrategySignal(
                    signal=SignalType.EXIT,
                    reason=f"Time Stop: {hold_minutes:.0f} min > {self.MAX_HOLD_MINUTES} max",
                    confidence=0.7,
                    metadata={"exit_reason": "TIME_STOP", "pnl_pct": pnl_pct}
                )

        # Hold position
        distance_to_mean = (current["sma20"] - current_price) / current_price * 100
        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"Holding: PnL={pnl_pct*100:.2f}%, {distance_to_mean:.1f}% to mean",
            metadata={
                "pnl_pct": pnl_pct,
                "zscore": current["zscore"],
                "rsi": current["rsi"],
                "distance_to_mean_pct": distance_to_mean,
            }
        )

    def _calculate_confidence(self, current: pd.Series) -> float:
        """Calculate confidence based on how extreme conditions are."""
        confidence = 0.5

        # More extreme Z-Score = higher confidence
        if current["zscore"] < -3.0:
            confidence += 0.20  # 3 sigma event
        elif current["zscore"] < -2.5:
            confidence += 0.15
        elif current["zscore"] < -2.0:
            confidence += 0.10

        # More extreme RSI = higher confidence
        if current["rsi"] < 20:
            confidence += 0.15
        elif current["rsi"] < 25:
            confidence += 0.10

        # Squeeze on = imminent reversal
        if current["squeeze_on"]:
            confidence += 0.10

        # Low ADX bonus (very choppy)
        if current["adx"] < 15:
            confidence += 0.05

        return min(confidence, 0.90)  # Cap at 90% (mean reversion is risky)
