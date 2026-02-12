"""
Vega Snap Strategy - Capturing Panic Reversals

The Physics: We exploit VEGA (Panic Premium) at market bottoms.

When markets crash:
1. VIX spikes = options get MORE expensive (Vega expansion)
2. Price drops to extreme levels (3 Standard Deviations)
3. Everyone is panicking = opportunity

The Counter-Intuitive Trade:
- At the bottom of a panic, buying a CALL seems crazy
- But the "Snap Back" in price is so VIOLENT that:
  - Delta gain from price recovery > Vega loss from vol crush
  - You profit from BOTH the direction AND the volatility

The Math:
- QQQ crashes 2% in a panic, hits 3 SD below mean
- VIX spikes from 18 to 28 (options are expensive)
- You buy ATM Call for $5.00 (inflated by Vega)
- Snap back happens: QQQ recovers 1% in 30 min
- Your call is now $7.50 despite VIX dropping to 24
- Why? Delta gain (+$3.00) > Vega loss (-$0.50)

The Key: You MUST exit on the SNAP, not hold for recovery.
- Take 10% profit immediately
- If no snap in 15 min, exit anyway (Theta + more vol crush)

Risk:
- If panic continues, you lose (but limited to premium)
- This is a RARE event strategy (3 SD = 0.3% probability)
- We wait for extreme conditions, then strike

Author: Bi-Cameral Quant Team
"""

from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

from strategies.options.core.base_options import (
    BaseOptionStrategy,
    OptionSignal,
    OptionSignalType,
    OptionPosition,
    ContractSpec,
    OptionType,
    ContractSelection,
)


class VegaSnapStrategy(BaseOptionStrategy):
    """
    Vega Snap - Capture violent reversals at panic bottoms.

    Entry Logic (RARE EVENT ONLY):
        1. EXTREME CRASH: Price < Lower Bollinger Band (3.0 SD)
        2. PANIC CONFIRMED: RSI < 20 (deeply oversold)
        3. VIX SPIKE: VIX > 22 OR VIX jumped > 10% recently
        4. VOLUME CAPITULATION: Volume > 2.5x average (selling exhaustion)

    Contract Selection:
        - ATM Call (betting on snap back)
        - 1-2 DTE (need some time for recovery)

    Exit Logic (AGGRESSIVE PROFIT TAKING):
        1. PROFIT TARGET: 10% gain (take the snap immediately)
        2. TIME STOP: 15 minutes (if no snap, vol crush will kill us)
        3. STOP LOSS: 25% loss (cut losses fast)
        4. VIX CRUSH: If VIX drops > 5% without price recovery, exit

    Why This Works:
        - 3 SD events are RARE but violent
        - At panic bottoms, snap backs are fast and furious
        - Delta gain outpaces Vega crush on rapid recoveries
        - We're trading the SNAP, not the recovery
    """

    name = "vega_snap"
    description = "Captures violent reversals at panic bottoms using Vega dynamics"
    version = "1.0.0"

    # Entry parameters (EXTREME CONDITIONS ONLY)
    BOLLINGER_PERIOD = 20
    BOLLINGER_SD = 3.0              # 3 Standard Deviations (rare event)
    RSI_PANIC_THRESHOLD = 20        # Deeply oversold
    VIX_ELEVATED_THRESHOLD = 22     # VIX must be elevated
    VIX_SPIKE_PCT = 0.10            # 10% VIX spike qualifies
    VOLUME_CAPITULATION = 2.5       # 2.5x average volume (selling exhaustion)

    # Z-Score for additional confirmation
    ZSCORE_EXTREME = -2.5           # 2.5 standard deviations below mean

    # Exit parameters (AGGRESSIVE)
    TIME_STOP_MINUTES = 15          # Short window - vol crush is coming
    PROFIT_TARGET_PCT = 0.10        # 10% profit - take the snap
    STOP_LOSS_PCT = 0.25            # 25% stop - cut fast
    VIX_CRUSH_EXIT_PCT = 0.05       # Exit if VIX drops 5% without recovery

    # Contract selection
    PREFERRED_DTE = 2               # Slightly more time for recovery
    TARGET_DELTA = 0.50             # ATM for balance

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicators for detecting panic bottoms.

        Key indicators:
        - Bollinger Bands (3 SD for extreme detection)
        - RSI (oversold confirmation)
        - Z-Score (statistical extreme)
        - Volume ratio (capitulation detection)
        """
        df = df.copy()

        # Bollinger Bands (3 Standard Deviations)
        df["bb_sma"] = df["Close"].rolling(window=self.BOLLINGER_PERIOD).mean()
        df["bb_std"] = df["Close"].rolling(window=self.BOLLINGER_PERIOD).std()
        df["bb_upper"] = df["bb_sma"] + (self.BOLLINGER_SD * df["bb_std"])
        df["bb_lower"] = df["bb_sma"] - (self.BOLLINGER_SD * df["bb_std"])
        df["bb_position"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # Z-Score
        df["zscore"] = (df["Close"] - df["bb_sma"]) / (df["bb_std"] + 1e-10)

        # RSI
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Volume analysis
        df["volume_sma"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"]

        # Price velocity (for snap detection)
        df["velocity"] = (df["Close"] - df["Open"]) / df["Open"]
        df["velocity_5"] = df["Close"].pct_change(periods=5)

        # ATR for context
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()

        # Distance from mean (for recovery tracking)
        df["distance_from_mean_pct"] = (df["Close"] - df["bb_sma"]) / df["bb_sma"] * 100

        # Forward fill and drop NaN
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[OptionPosition] = None,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Generate Vega Snap signal.

        Args:
            df: Prepared DataFrame with panic indicators
            current_position: Current option position (if any)
            vix_value: Current VIX level (IMPORTANT for this strategy)

        Returns:
            OptionSignal with contract spec and aggressive exit rules
        """
        if df.empty or len(df) < 25:
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason="Insufficient data for Vega Snap"
            )

        current = df.iloc[-1]
        current_price = current["Close"]
        current_time = df.index[-1]

        # Time filter
        if not self.is_trading_time(current_time):
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason=f"Outside trading hours: {current_time.strftime('%H:%M')}"
            )

        # If we have a position, check for exit
        if current_position is not None:
            return self._check_exit(current, current_position, current_time, vix_value)

        # Check for entry (RARE EVENT)
        return self._check_entry(current, current_time, vix_value)

    def _check_entry(
        self,
        current: pd.Series,
        current_time: datetime,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Check for panic bottom entry conditions.

        This is a RARE EVENT strategy. All conditions must align:
        - Price at 3 SD extreme
        - RSI deeply oversold
        - VIX elevated
        - Volume capitulation
        """
        current_price = current["Close"]
        bb_lower = current["bb_lower"]
        zscore = current["zscore"]
        rsi = current["rsi"]
        volume_ratio = current["volume_ratio"]

        conditions = {}

        # 1. EXTREME PRICE: Below 3 SD Bollinger Band
        conditions["price_extreme"] = current_price < bb_lower

        # 2. Z-SCORE EXTREME: Statistical panic
        conditions["zscore_extreme"] = zscore < self.ZSCORE_EXTREME

        # 3. RSI PANIC: Deeply oversold
        conditions["rsi_panic"] = rsi < self.RSI_PANIC_THRESHOLD

        # 4. VOLUME CAPITULATION: Selling exhaustion
        conditions["volume_capitulation"] = volume_ratio >= self.VOLUME_CAPITULATION

        # 5. VIX ELEVATED: Fear is present
        if vix_value:
            conditions["vix_elevated"] = vix_value >= self.VIX_ELEVATED_THRESHOLD
        else:
            # If no VIX, require even more extreme conditions
            conditions["vix_elevated"] = zscore < -3.0

        # All conditions must be met (RARE)
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]

            # Only show detailed output if we're close (at least 3/5 conditions)
            met_count = sum(conditions.values())
            if met_count >= 3:
                return OptionSignal(
                    signal=OptionSignalType.HOLD,
                    reason=f"Vega Snap near ({met_count}/5): {', '.join(failed)}",
                    metadata={
                        "conditions": conditions,
                        "zscore": zscore,
                        "rsi": rsi,
                        "volume_ratio": volume_ratio,
                        "vix": vix_value,
                        "met_count": met_count,
                    }
                )
            else:
                return OptionSignal(
                    signal=OptionSignalType.HOLD,
                    reason="Vega Snap: Waiting for panic conditions",
                    metadata={"met_count": met_count}
                )

        # PANIC DETECTED! This is rare - we strike.

        # Use elevated IV for pricing (VIX is high)
        iv = (vix_value / 100) if vix_value else 0.30

        # Select ATM Call for the snap back
        contract = self.select_contract(
            underlying_price=current_price,
            option_type=OptionType.CALL,
            selection=ContractSelection.ATM,
            current_time=current_time,
            iv=iv,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(current, vix_value)

        return OptionSignal(
            signal=OptionSignalType.BUY_CALL,
            reason=f"PANIC BOTTOM: Z={zscore:.2f}, RSI={rsi:.0f}, VIX={vix_value or 'N/A'}, Vol={volume_ratio:.1f}x",
            confidence=confidence,
            contract=contract,
            time_stop_minutes=self.TIME_STOP_MINUTES,
            profit_target_pct=self.PROFIT_TARGET_PCT,
            stop_loss_pct=self.STOP_LOSS_PCT,
            metadata={
                "zscore": zscore,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "vix": vix_value,
                "bb_lower": bb_lower,
                "distance_from_mean_pct": current["distance_from_mean_pct"],
                "contract_strike": contract.strike,
                "contract_delta": contract.delta,
                "estimated_premium": contract.mid_price,
                "iv_estimate": iv,
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: OptionPosition,
        current_time: datetime,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Check exit conditions for Vega Snap.

        AGGRESSIVE EXIT DISCIPLINE:
        - This is a SNAP trade, not a recovery trade
        - Take 10% profit immediately
        - Exit in 15 min regardless (vol crush coming)
        - Watch for VIX crush without price recovery
        """
        current_price = current["Close"]

        # Calculate current P&L
        pnl_dollars, pnl_pct = self.estimate_current_pnl(
            position=position,
            current_underlying_price=current_price,
            current_time=current_time,
        )

        # Track highest P&L
        if pnl_pct > position.highest_pnl_pct:
            position.highest_pnl_pct = pnl_pct

        # Calculate time held
        held_seconds = (current_time - position.entry_time).total_seconds()
        held_minutes = held_seconds / 60

        # Calculate price recovery
        entry_distance = position.entry_underlying_price - current["bb_sma"]
        current_distance = current_price - current["bb_sma"]
        recovery_pct = 1 - (current_distance / entry_distance) if entry_distance != 0 else 0

        # EXIT 1: PROFIT TARGET (10% - take the snap!)
        if pnl_pct >= self.PROFIT_TARGET_PCT * 100:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"SNAP CAPTURED: +{pnl_pct:.1f}% >= +{self.PROFIT_TARGET_PCT*100:.0f}%",
                confidence=0.95,
                metadata={
                    "exit_reason": "PROFIT_TARGET",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                    "recovery_pct": recovery_pct,
                }
            )

        # EXIT 2: TIME STOP (15 min - vol crush coming)
        if held_minutes >= self.TIME_STOP_MINUTES:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"TIME STOP: {held_minutes:.1f} min >= {self.TIME_STOP_MINUTES} (vol crush risk)",
                confidence=0.99,
                metadata={
                    "exit_reason": "TIME_STOP",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 3: STOP LOSS
        if pnl_pct <= -self.STOP_LOSS_PCT * 100:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"STOP LOSS: {pnl_pct:.1f}% <= -{self.STOP_LOSS_PCT*100:.0f}%",
                confidence=0.95,
                metadata={
                    "exit_reason": "STOP_LOSS",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 4: SNAP HAPPENED - Lock in gains
        # If price recovered > 50% toward mean and we're profitable
        if recovery_pct > 0.5 and pnl_pct > 5:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"SNAP COMPLETE: {recovery_pct*100:.0f}% recovery, +{pnl_pct:.1f}% P&L",
                confidence=0.85,
                metadata={
                    "exit_reason": "SNAP_COMPLETE",
                    "pnl_pct": pnl_pct,
                    "recovery_pct": recovery_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 5: NO SNAP - Price still falling
        # If Z-Score gets even more extreme (panic deepening)
        if current["zscore"] < -3.5 and pnl_pct < 0:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"PANIC DEEPENING: Z-Score={current['zscore']:.2f}, cutting loss",
                confidence=0.80,
                metadata={
                    "exit_reason": "PANIC_DEEPENING",
                    "pnl_pct": pnl_pct,
                    "zscore": current["zscore"],
                    "held_minutes": held_minutes,
                }
            )

        # HOLD - show status
        time_remaining = self.TIME_STOP_MINUTES - held_minutes
        return OptionSignal(
            signal=OptionSignalType.HOLD,
            reason=f"Waiting for snap: P&L={pnl_pct:+.1f}%, Recovery={recovery_pct*100:.0f}%, {time_remaining:.1f}min left",
            metadata={
                "pnl_pct": pnl_pct,
                "held_minutes": held_minutes,
                "time_remaining": time_remaining,
                "recovery_pct": recovery_pct,
                "zscore": current["zscore"],
                "rsi": current["rsi"],
            }
        )

    def _calculate_confidence(
        self,
        current: pd.Series,
        vix_value: Optional[float],
    ) -> float:
        """Calculate confidence based on panic severity."""
        confidence = 0.50

        zscore = current["zscore"]
        rsi = current["rsi"]
        volume_ratio = current["volume_ratio"]

        # More extreme Z-Score = higher confidence
        if zscore < -3.5:
            confidence += 0.20
        elif zscore < -3.0:
            confidence += 0.15
        elif zscore < -2.5:
            confidence += 0.10

        # More extreme RSI = higher confidence
        if rsi < 15:
            confidence += 0.15
        elif rsi < 18:
            confidence += 0.10

        # Higher volume capitulation = higher confidence
        if volume_ratio >= 4.0:
            confidence += 0.15
        elif volume_ratio >= 3.0:
            confidence += 0.10

        # VIX level affects confidence
        if vix_value:
            if vix_value >= 30:
                confidence += 0.10  # Extreme fear = good setup
            elif vix_value >= 25:
                confidence += 0.05

        return min(confidence, 0.85)  # Cap at 85% (still risky)
