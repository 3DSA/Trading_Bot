"""
Mean Reversion Strategy - "The Rubber Band" (v3.0)

A statistical mean reversion strategy for choppy/ranging markets.
Uses Z-Score and Bollinger Bands to identify oversold conditions.

Best Market Regime: CHOP (ADX < 20)
Typical Hold Time: 15-120 minutes
Win Rate Target: 55-65%
Risk/Reward: 1:0.8 (small but consistent gains)

Key Features:
    - Z-Score Entry: Only buys at statistically extreme levels (-2 sigma)
    - RSI Confluence: Requires momentum oversold confirmation
    - Bollinger Band Squeeze: Identifies low volatility before reversal
    - FIXED PRICE TARGETS: 0.8% take profit, 1% stop loss (v3.0 fix)
    - Trailing Stop: Locks in gains on extended moves

v3.0 Change: Replaced Z-Score-based exits with fixed price targets.
The old Z-Score exit was flawed - it could trigger exits at a loss
because the rolling window shifts down with falling prices.

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

    v3.0: Fixed Price Target Exits (no more Z-Score exit trap!)

    Entry Logic:
        1. Z-Score < -2.0 (price is 2 standard deviations below mean)
        2. RSI < 30 (momentum is oversold)
        3. ADX < 20 (no trend - chop mode)
        4. Price < Lower Bollinger Band (volatility confirmation)
        5. Volume > 1.5x average (selling exhaustion)

    Exit Logic (FIXED PRICE TARGETS - v3.0):
        1. Stop Loss: Fixed 1% hard floor (non-negotiable)
        2. Take Profit: Fixed 0.8% target (small but consistent)
        3. Trailing Stop: Activates at 0.5% profit, trails by 0.3%
        4. Time Stop: 2 hours max hold

    Why Fixed Targets? (The Z-Score Trap):
        - Old problem: Z-Score exits when rolling window shifts, not when profitable
        - Example: Buy at $100, price drops to $98, Z-Score returns to 0
          because the 20-period mean dropped too. Trade exits at a LOSS
          despite "returning to mean"
        - Solution: Fixed percentage targets guarantee profit when hit

    Risk Management:
        - Half position sizing (high risk of "catching falling knife")
        - Fixed 1% stop loss (hard floor, non-negotiable)
        - 0.8% take profit (realistic for mean reversion)
        - Trailing stop locks in gains on runners
    """

    name = "mean_reversion"
    description = "Z-Score + Bollinger mean reversion for chop markets"
    version = "3.0.0"
    preferred_regime = "CHOP"

    # Strategy-specific parameters - v3.1 TIGHT QUALITY ENTRIES
    ZSCORE_WINDOW = 20
    ZSCORE_ENTRY_THRESHOLD = -2.5  # TIGHTER (was -2.0) - only extreme 2.5 sigma events
    ZSCORE_EXIT_THRESHOLD = 0.0   # Back to mean (not used in v3.0)
    RSI_OVERSOLD = 25  # TIGHTER (was 30) - require truly oversold
    RSI_NEUTRAL = 50
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2.0
    VOLUME_CONFIRMATION = 2.0  # TIGHTER (was 1.5) - need strong selling exhaustion
    MAX_HOLD_MINUTES = 60  # SHORTER (was 120) - quick scalp or get out
    FIXED_STOP_LOSS_PCT = 0.015  # 1.5% hard stop

    # v3.0 Exit improvements - FIXED PRICE TARGETS instead of Z-Score exits
    # The core fix: Z-Score can return to 0 even when losing (rolling window shifts down)
    # Solution: Use fixed percentage targets that guarantee profit when hit
    ADX_EXIT_THRESHOLD = 30  # Higher - regime changes need to be clear
    FIXED_TAKE_PROFIT_PCT = 0.01  # 1% fixed profit target (quick scalp)
    FIXED_TRAILING_ACTIVATION = 0.005  # Activate trailing stop at 0.5% profit
    FIXED_TRAILING_DISTANCE = 0.003  # Trail by 0.3% (tight trailing)

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
        """
        Check mean reversion exit conditions.

        v3.0: Uses FIXED PRICE TARGETS instead of Z-Score exits.
        The problem with Z-Score exits: rolling window shifts down with price,
        so Z-Score can return to 0 even when the trade is losing money.

        New exit logic priority:
        1. Stop Loss (hard floor protection)
        2. Take Profit (fixed percentage target)
        3. Trailing Stop (lock in gains)
        4. Time Stop (don't hold forever)
        """
        entry_price = position.get("entry_price", current["Close"])
        entry_time = position.get("entry_time")
        current_price = current["Close"]
        highest_price = position.get("highest_price", entry_price)

        pnl_pct = (current_price - entry_price) / entry_price

        # Track highest price for trailing stop
        if current_price > highest_price:
            highest_price = current_price
            position["highest_price"] = highest_price

        # Exit 1: STOP LOSS - Hard floor (non-negotiable)
        if pnl_pct <= -self.FIXED_STOP_LOSS_PCT:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Stop Loss: {pnl_pct*100:.2f}% <= -{self.FIXED_STOP_LOSS_PCT*100:.1f}%",
                confidence=0.95,
                metadata={"exit_reason": "STOP_LOSS", "pnl_pct": pnl_pct}
            )

        # Exit 2: TAKE PROFIT - Fixed percentage target
        if pnl_pct >= self.FIXED_TAKE_PROFIT_PCT:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Take Profit: +{pnl_pct*100:.2f}% >= +{self.FIXED_TAKE_PROFIT_PCT*100:.1f}% target",
                confidence=0.9,
                metadata={"exit_reason": "TAKE_PROFIT", "pnl_pct": pnl_pct}
            )

        # Exit 3: TRAILING STOP - Lock in gains once activated
        if highest_price > entry_price * (1 + self.FIXED_TRAILING_ACTIVATION):
            # Trailing stop is active
            trail_stop_price = highest_price * (1 - self.FIXED_TRAILING_DISTANCE)
            if current_price < trail_stop_price:
                return StrategySignal(
                    signal=SignalType.EXIT,
                    reason=f"Trailing Stop: Price ${current_price:.2f} < Trail ${trail_stop_price:.2f}",
                    confidence=0.85,
                    metadata={
                        "exit_reason": "TRAILING_STOP",
                        "pnl_pct": pnl_pct,
                        "highest_price": highest_price,
                    }
                )

        # Exit 4: TIME STOP - Don't hold mean reversion trades forever
        if entry_time:
            hold_minutes = (current_time - entry_time).total_seconds() / 60
            if hold_minutes > self.MAX_HOLD_MINUTES:
                return StrategySignal(
                    signal=SignalType.EXIT,
                    reason=f"Time Stop: {hold_minutes:.0f} min > {self.MAX_HOLD_MINUTES} max",
                    confidence=0.7,
                    metadata={"exit_reason": "TIME_STOP", "pnl_pct": pnl_pct}
                )

        # Hold position - show progress toward target
        progress_pct = (pnl_pct / self.FIXED_TAKE_PROFIT_PCT) * 100 if self.FIXED_TAKE_PROFIT_PCT > 0 else 0
        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"Holding: PnL={pnl_pct*100:.2f}%, {progress_pct:.0f}% to TP target",
            metadata={
                "pnl_pct": pnl_pct,
                "progress_to_target_pct": progress_pct,
                "zscore": current["zscore"],
                "rsi": current["rsi"],
                "highest_price": highest_price,
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
