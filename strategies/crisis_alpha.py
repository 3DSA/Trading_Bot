"""
Crisis Alpha Strategy - "The Bear"

A VIX-weighted trend following strategy for bearish market conditions.
Designed specifically for trading SQQQ during market turmoil.

Best Market Regime: CRISIS / HIGH VIX (>25)
Typical Hold Time: Hours to Days
Win Rate Target: 40-50%
Risk/Reward: 1:4

Key Features:
    - VIX Trigger: Only activates when VIX > 25 (fear in market)
    - Trend Confirmation: EMA 10 for short-term direction
    - Pyramiding: Adds to winners as trend extends
    - Wide Stops: Allows for volatility during crisis
    - SQQQ Focus: Inverse ETF for bearish bets

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
    calc_rsi,
    calc_sma,
    calc_vwap,
    calc_macd,
    calc_bollinger_bands,
    calc_plus_minus_di,
)


class CrisisAlphaStrategy(BaseStrategy):
    """
    The Bear - VIX-Weighted Trend Following for SQQQ.

    This strategy is designed for bearish market conditions when
    fear (VIX) is elevated. It trades SQQQ (inverse QQQ) to profit
    from market declines.

    Activation Criteria:
        1. VIX > 25 (elevated fear)
        2. Market is in downtrend (QQQ below EMA 10)

    Entry Logic:
        1. SQQQ Price > EMA 10 (inverse is trending up)
        2. ADX > 25 (strong trend)
        3. +DI > -DI (bullish on SQQQ = bearish on QQQ)
        4. Volume confirmation

    Pyramiding (Adding to Winners):
        - If in profit > 1% AND price > EMA 10, add 25% position
        - Maximum 3 pyramid levels

    Exit Logic:
        1. VIX drops below 20 (fear subsiding)
        2. Price < EMA 10 (trend reversal)
        3. Trailing stop: 3x ATR (wide for volatility)
        4. Time-based: Review at EOD

    Risk Management:
        - Starts with half position (crisis is unpredictable)
        - Wide ATR stops (3x) to avoid whipsaws
        - Pyramids only into winners
    """

    name = "crisis_alpha"
    description = "VIX-weighted trend following for SQQQ"
    version = "1.0.0"
    preferred_regime = "CRISIS"

    # Strategy-specific parameters
    VIX_ACTIVATION_THRESHOLD = 25.0  # VIX must be above this to trade
    VIX_EXIT_THRESHOLD = 20.0        # Exit when VIX drops below
    EMA_TREND_PERIOD = 10
    MIN_ADX_FOR_ENTRY = 25
    PYRAMID_PROFIT_THRESHOLD = 0.01  # 1% profit before pyramiding
    MAX_PYRAMID_LEVELS = 3
    ATR_STOP_MULTIPLIER = 3.0        # Wide stops for crisis volatility
    ATR_TP_MULTIPLIER = 6.0          # 2:1 R/R with wide stops

    def __init__(self, config: Optional[StrategyConfig] = None):
        """Initialize with SQQQ-specific defaults."""
        super().__init__(config)
        # Override for SQQQ
        if self.config.symbol != "SQQQ":
            self.config.symbol = "SQQQ"
            self.config.underlying_symbol = "QQQ"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crisis-specific indicators."""
        df = df.copy()

        # Trend indicators
        df["ema_10"] = calc_ema(df["Close"], span=self.EMA_TREND_PERIOD)
        df["ema_21"] = calc_ema(df["Close"], span=21)
        df["sma_50"] = calc_sma(df["Close"], window=50)

        # Trend strength
        df["adx"] = calc_adx(df, period=14)
        df["+di"], df["-di"] = calc_plus_minus_di(df, period=14)

        # Momentum
        df["rsi"] = calc_rsi(df["Close"], period=14)
        df["macd"], df["macd_signal"], df["macd_hist"] = calc_macd(df["Close"])

        # Volatility
        df["atr"] = calc_atr(df, period=14)
        df["atr_pct"] = df["atr"] / df["Close"] * 100

        # Bollinger for squeeze detection
        df["bb_upper"], df["bb_mid"], df["bb_lower"] = calc_bollinger_bands(
            df["Close"], window=20, num_std=2.0
        )

        # VWAP
        df["vwap"] = calc_vwap(df, reset_daily=True)

        # Forward fill
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
        underlying_df: Optional[pd.DataFrame] = None,
        vix_value: Optional[float] = None
    ) -> StrategySignal:
        """
        Generate crisis alpha signal.

        Args:
            df: SQQQ DataFrame with indicators
            current_position: Current position (None if flat)
            underlying_df: QQQ DataFrame for confirmation
            vix_value: Current VIX level (required for this strategy)

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

        # Get VIX value (simulated from ATR if not provided)
        if vix_value is None:
            vix_value = self._estimate_vix(df, underlying_df)

        # Check VIX activation
        vix_active = vix_value >= self.VIX_ACTIVATION_THRESHOLD

        # Time filter
        if not self.is_trading_time(current_time):
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Outside trading hours: {current_time.strftime('%H:%M')}",
                metadata={"vix": vix_value, "vix_active": vix_active}
            )

        # If we have a position, check for exit or pyramid
        if current_position is not None:
            exit_signal = self._check_exit(current, current_position, vix_value)
            if exit_signal.signal == SignalType.EXIT:
                return exit_signal

            # Check for pyramid opportunity
            pyramid_signal = self._check_pyramid(current, current_position, vix_value)
            if pyramid_signal.signal == SignalType.BUY:
                return pyramid_signal

            return exit_signal  # Hold signal

        # Check if VIX is active for new entry
        if not vix_active:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"VIX {vix_value:.1f} < {self.VIX_ACTIVATION_THRESHOLD} - Crisis mode inactive",
                metadata={"vix": vix_value, "vix_active": False}
            )

        # Check for entry
        return self._check_entry(current, underlying_df, vix_value)

    def _check_entry(
        self,
        current: pd.Series,
        underlying_df: Optional[pd.DataFrame],
        vix_value: float
    ) -> StrategySignal:
        """Check crisis entry conditions."""
        current_price = current["Close"]
        atr = current["atr"]

        conditions = {}

        # 1. SQQQ above EMA 10 (inverse trending up = QQQ trending down)
        conditions["above_ema10"] = current_price > current["ema_10"]

        # 2. ADX showing strong trend
        conditions["adx_strong"] = current["adx"] > self.MIN_ADX_FOR_ENTRY

        # 3. +DI > -DI on SQQQ (bullish on inverse = bearish on QQQ)
        conditions["di_bullish"] = current["+di"] > current["-di"]

        # 4. MACD histogram positive (momentum confirmation)
        conditions["macd_bullish"] = current["macd_hist"] > 0

        # 5. Check underlying (QQQ should be weak)
        qqq_bearish = True
        if underlying_df is not None and not underlying_df.empty:
            qqq_current = underlying_df.iloc[-1]
            qqq_ema10 = calc_ema(underlying_df["Close"], span=10).iloc[-1]
            qqq_bearish = qqq_current["Close"] < qqq_ema10
        conditions["qqq_bearish"] = qqq_bearish

        # All conditions check
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Crisis entry blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "vix": vix_value,
                    "adx": current["adx"],
                    "ema_10": current["ema_10"],
                }
            )

        # Calculate risk parameters (wide for crisis)
        stop_loss = current_price - (atr * self.ATR_STOP_MULTIPLIER)
        take_profit = current_price + (atr * self.ATR_TP_MULTIPLIER)
        trailing_pct = (atr * self.ATR_STOP_MULTIPLIER) / current_price

        confidence = self._calculate_confidence(current, vix_value)

        return StrategySignal(
            signal=SignalType.BUY,
            reason=f"Crisis Alpha Entry: VIX={vix_value:.1f}, ADX={current['adx']:.1f}",
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_pct,
            position_size=PositionSizing.HALF,  # Start with half (crisis is risky)
            metadata={
                "entry_price": current_price,
                "vix": vix_value,
                "atr": atr,
                "adx": current["adx"],
                "ema_10": current["ema_10"],
                "+di": current["+di"],
                "-di": current["-di"],
                "pyramid_level": 1,
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: dict,
        vix_value: float
    ) -> StrategySignal:
        """Check crisis exit conditions."""
        entry_price = position.get("entry_price", current["Close"])
        current_price = current["Close"]

        pnl_pct = (current_price - entry_price) / entry_price

        # Exit 1: VIX subsiding (crisis over)
        if vix_value < self.VIX_EXIT_THRESHOLD:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"VIX Subsiding: {vix_value:.1f} < {self.VIX_EXIT_THRESHOLD}",
                confidence=0.9,
                metadata={"exit_reason": "VIX_SUBSIDING", "pnl_pct": pnl_pct}
            )

        # Exit 2: Trend reversal (price below EMA 10)
        if current_price < current["ema_10"]:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"Trend Reversal: Price < EMA10 (${current['ema_10']:.2f})",
                confidence=0.85,
                metadata={"exit_reason": "TREND_REVERSAL", "pnl_pct": pnl_pct}
            )

        # Exit 3: -DI crosses above +DI (bearish on SQQQ = bullish on QQQ)
        if current["-di"] > current["+di"] and pnl_pct > 0.005:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason=f"DI Crossover Bearish: -DI > +DI",
                confidence=0.75,
                metadata={"exit_reason": "DI_CROSSOVER", "pnl_pct": pnl_pct}
            )

        # Exit 4: MACD histogram turns negative
        if current["macd_hist"] < 0 and pnl_pct > 0.01:
            return StrategySignal(
                signal=SignalType.EXIT,
                reason="MACD Momentum Loss",
                confidence=0.7,
                metadata={"exit_reason": "MACD_NEGATIVE", "pnl_pct": pnl_pct}
            )

        # Hold
        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"Crisis Hold: VIX={vix_value:.1f}, PnL={pnl_pct*100:.2f}%",
            metadata={
                "pnl_pct": pnl_pct,
                "vix": vix_value,
                "adx": current["adx"],
                "pyramid_level": position.get("pyramid_level", 1),
            }
        )

    def _check_pyramid(
        self,
        current: pd.Series,
        position: dict,
        vix_value: float
    ) -> StrategySignal:
        """Check if we should add to winning position (pyramid)."""
        entry_price = position.get("entry_price", current["Close"])
        current_price = current["Close"]
        pyramid_level = position.get("pyramid_level", 1)

        pnl_pct = (current_price - entry_price) / entry_price

        # Don't pyramid if already at max
        if pyramid_level >= self.MAX_PYRAMID_LEVELS:
            return StrategySignal(
                signal=SignalType.HOLD,
                reason=f"Max pyramid level reached ({pyramid_level})",
            )

        # Pyramid conditions
        conditions = {
            "in_profit": pnl_pct >= self.PYRAMID_PROFIT_THRESHOLD,
            "above_ema10": current_price > current["ema_10"],
            "adx_strong": current["adx"] > 25,
            "vix_elevated": vix_value > self.VIX_ACTIVATION_THRESHOLD,
        }

        if all(conditions.values()):
            return StrategySignal(
                signal=SignalType.BUY,
                reason=f"Pyramid Level {pyramid_level + 1}: PnL={pnl_pct*100:.2f}%",
                confidence=0.7,
                position_size=PositionSizing.QUARTER,  # Add 25% each pyramid
                metadata={
                    "pyramid_level": pyramid_level + 1,
                    "pnl_pct": pnl_pct,
                    "vix": vix_value,
                }
            )

        return StrategySignal(
            signal=SignalType.HOLD,
            reason=f"No pyramid: {[k for k, v in conditions.items() if not v]}",
        )

    def _estimate_vix(
        self,
        df: pd.DataFrame,
        underlying_df: Optional[pd.DataFrame]
    ) -> float:
        """
        Estimate VIX from ATR% when actual VIX not available.

        This is a rough approximation:
        - Normal market: ATR% ~0.8-1.2% -> VIX ~15-20
        - Elevated: ATR% ~1.5-2.5% -> VIX ~25-35
        - Crisis: ATR% ~3%+ -> VIX ~40+
        """
        if underlying_df is not None and not underlying_df.empty:
            # Use QQQ ATR% for more accurate VIX proxy
            atr_pct = calc_atr(underlying_df, 14).iloc[-1] / underlying_df["Close"].iloc[-1] * 100
        else:
            # Use SQQQ ATR% / 3 (since it's 3x leveraged)
            atr_pct = df["atr_pct"].iloc[-1] / 3

        # Rough VIX estimation
        estimated_vix = 12 + (atr_pct * 15)

        return min(max(estimated_vix, 10), 80)  # Clamp between 10-80

    def _calculate_confidence(self, current: pd.Series, vix_value: float) -> float:
        """Calculate confidence based on crisis indicators."""
        confidence = 0.4  # Lower base (crisis is unpredictable)

        # Higher VIX = higher confidence (more fear = more opportunity)
        if vix_value > 35:
            confidence += 0.15
        elif vix_value > 30:
            confidence += 0.10
        elif vix_value > 25:
            confidence += 0.05

        # Strong ADX
        if current["adx"] > 35:
            confidence += 0.10
        elif current["adx"] > 30:
            confidence += 0.05

        # Wide DI spread
        di_spread = current["+di"] - current["-di"]
        if di_spread > 15:
            confidence += 0.10

        # MACD confirmation
        if current["macd_hist"] > 0:
            confidence += 0.05

        return min(confidence, 0.80)  # Cap at 80% (crisis is risky)
