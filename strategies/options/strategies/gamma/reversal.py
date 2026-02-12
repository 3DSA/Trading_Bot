"""
Reversal Scalper Strategy - Fading Exhausted Explosive Moves

The Physics: When Gamma Scalper would catch a LATE explosion, we FADE it instead.

The Problem with Late Gamma Scalper Entries:
- Exhaustion Score >= 2: 0% win rate in 2018, 2019, 2022, 2024
- Midday session: Consistently 25-35% win rate across all years
- By the time we detect the explosion, it's often about to reverse

The Reversal Scalper Edge:
- Instead of riding an exhausted explosion, we fade it
- Enter OPPOSITE direction when exhaustion signals appear
- Tighter profit target (quick reversal scalp)
- Wider initial stop (let the exhausted move fully extend)

Trigger Logic (compound rules from backtest analysis):
    Rule 1: exhaustion_score >= 2 AND VIX < 25 -> reversal
    Rule 2: session_phase == "midday" AND exhaustion_score >= 1 -> reversal
    Rule 3: exhaustion_score >= 3 (any VIX) -> reversal

Entry Logic (opposite of Gamma Scalper):
    - BULLISH explosion detected + exhaustion -> BUY PUT (fade the up move)
    - BEARISH explosion detected + exhaustion -> BUY CALL (fade the down move)

Exit Logic (faster than Gamma Scalper):
    - TIME STOP: 8 minutes (reversals are quick or fail)
    - PROFIT TARGET: 10% (quick scalp, don't wait for full reversal)
    - STOP LOSS: 25% (tighter than gamma scalper)

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


class ReversalScalperStrategy(BaseOptionStrategy):
    """
    Reversal Scalper - Fade exhausted explosive moves.

    This is a Level 3 sub-strategy of Gamma Scalper. When Gamma Scalper
    detects an explosion but the exhaustion score indicates the move is
    overextended, we enter the OPPOSITE direction to capture the reversal.

    Trigger Conditions (from backtest analysis):
        Rule 1: exhaustion_score >= 2 AND vix < 25
        Rule 2: session_phase == "midday" AND exhaustion_score >= 1
        Rule 3: exhaustion_score >= 3 (any VIX level)

    Entry Logic:
        - Bullish explosion + exhaustion -> BUY PUT (fade up move)
        - Bearish explosion + exhaustion -> BUY CALL (fade down move)

    Exit Logic (tighter than Gamma Scalper):
        - TIME STOP: 8 minutes (reversals happen fast or not at all)
        - PROFIT TARGET: 10% (quick scalp)
        - STOP LOSS: 25% (accept some extension before reversal)

    Why This Works:
        - Exhausted moves have high reversion probability
        - We're not fighting the trend, we're catching the snap-back
        - Quick exits minimize Theta exposure
    """

    name = "reversal_scalper"
    description = "Fades exhausted explosive moves using reversal physics"
    version = "1.0.0"

    # Entry parameters - same velocity detection as Gamma Scalper
    VELOCITY_THRESHOLD = 0.004      # Same as Gamma Scalper
    VOLUME_SPIKE_THRESHOLD = 2.0    # Slightly lower (move is extended anyway)

    # Exhaustion trigger thresholds (from backtest analysis)
    EXHAUSTION_THRESHOLD_NORMAL = 2  # Score >= 2 triggers reversal (VIX < 25)
    EXHAUSTION_THRESHOLD_HIGH_VIX = 3  # Score >= 3 for high VIX (continuation likely)
    EXHAUSTION_THRESHOLD_MIDDAY = 1  # Score >= 1 during midday session
    VIX_THRESHOLD = 25  # Above this, use higher exhaustion threshold

    # Exit parameters (TIGHTER than Gamma Scalper)
    TIME_STOP_MINUTES = 8           # Faster - reversals happen quick
    PROFIT_TARGET_PCT = 0.10        # 10% quick scalp (was 15%)
    STOP_LOSS_PCT = 0.25            # 25% stop (wider to allow extension)
    TRAILING_ACTIVATION_PCT = 0.06  # Activate trail at 6%
    TRAILING_DISTANCE_PCT = 0.03    # Trail by 3%

    # Contract selection
    PREFERRED_DTE = 1               # 1-DTE for Gamma exposure
    TARGET_DELTA = 0.45             # Slightly OTM (cheaper premium for reversal bet)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for reversal detection.

        Uses the same indicators as Gamma Scalper since we're looking for
        the same explosive moves, but with exhaustion signals.
        """
        df = df.copy()

        # Velocity (impulse detection)
        df["velocity"] = (df["Close"] - df["Open"]) / df["Open"]
        df["velocity_abs"] = df["velocity"].abs()

        # Volume analysis
        df["volume_sma"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"]

        # ATR for move validation
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()
        df["atr_pct"] = df["atr"] / df["Close"]

        # RSI for momentum state
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Momentum (rate of change)
        df["momentum"] = df["Close"].pct_change(periods=5) * 100

        # EMA for trend context
        df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()

        # =============================================================================
        # Exhaustion Detection Variables (same as Gamma Scalper)
        # =============================================================================

        # 1. CUMULATIVE MOVE: Total move over last 5 bars
        df["cumulative_move_5"] = df["Close"].pct_change(periods=5)

        # 2. PRIOR BAR BEHAVIOR
        df["prior_bar_velocity"] = df["velocity"].shift(1)
        df["prior_bar_direction"] = np.sign(df["velocity"].shift(1))

        # 3. VOLUME PATTERN
        df["prior_volume_ratio"] = df["volume_ratio"].shift(1)
        df["volume_declining"] = df["volume_ratio"] < df["prior_volume_ratio"]
        df["exhaustion_volume"] = df["volume_ratio"] >= 8.0

        # 4. BARS SINCE EXPLOSION START
        explosion_active = df["velocity_abs"] >= self.VELOCITY_THRESHOLD / 2
        df["bars_in_explosion"] = explosion_active.groupby(
            (~explosion_active).cumsum()
        ).cumsum()

        # 5. SESSION PHASE
        def get_session_phase(timestamp):
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour
                minute = timestamp.minute
                time_minutes = hour * 60 + minute

                if time_minutes < 570:  # Before 9:30
                    return "pre_market"
                elif time_minutes < 600:  # 9:30-10:00
                    return "open_drive"
                elif time_minutes < 870:  # 10:00-14:30
                    return "midday"
                elif time_minutes < 960:  # 14:30-16:00
                    return "close_drive"
                else:
                    return "after_hours"
            return "unknown"

        df["session_phase"] = df.index.map(get_session_phase)

        # 6. EXHAUSTION SCORE
        df["exhaustion_score"] = (
            (df["rsi"] >= 65).astype(int) * 1 +  # RSI overbought
            (df["rsi"] <= 35).astype(int) * 1 +  # RSI oversold
            (df["cumulative_move_5"].abs() >= 0.01).astype(int) * 1 +  # Big cumulative move
            df["exhaustion_volume"].astype(int) * 1 +  # High volume
            df["volume_declining"].astype(int) * 1 +  # Volume fading
            (df["bars_in_explosion"] >= 3).astype(int) * 1  # Late to the move
        )

        # Forward fill and drop NaN
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def should_trigger_reversal(
        self,
        exhaustion_score: int,
        session_phase: str,
        vix_value: float,
    ) -> tuple[bool, str]:
        """
        Check if reversal scalper should be triggered based on compound rules.

        Returns:
            tuple: (should_trigger, rule_triggered)
        """
        # Rule 3: Extreme exhaustion (any VIX) - highest priority
        if exhaustion_score >= self.EXHAUSTION_THRESHOLD_HIGH_VIX:
            return True, "Rule3_ExtremeExhaustion"

        # Rule 2: Midday session + any exhaustion
        if session_phase == "midday" and exhaustion_score >= self.EXHAUSTION_THRESHOLD_MIDDAY:
            return True, "Rule2_MiddayExhaustion"

        # Rule 1: Normal exhaustion in low VIX
        if exhaustion_score >= self.EXHAUSTION_THRESHOLD_NORMAL and vix_value < self.VIX_THRESHOLD:
            return True, "Rule1_NormalExhaustion"

        return False, "NoTrigger"

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[OptionPosition] = None,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Generate Reversal Scalper signal.

        Args:
            df: Prepared DataFrame with velocity and exhaustion indicators
            current_position: Current option position (if any)
            vix_value: Current VIX level

        Returns:
            OptionSignal for fading exhausted moves
        """
        if df.empty or len(df) < 20:
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason="Insufficient data for Reversal Scalper"
            )

        current = df.iloc[-1]
        current_price = current["Close"]
        current_time = df.index[-1]
        vix = vix_value if vix_value is not None else 20.0

        # Time filter
        if not self.is_trading_time(current_time):
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason=f"Outside trading hours: {current_time.strftime('%H:%M')}"
            )

        # If we have a position, check for exit
        if current_position is not None:
            return self._check_exit(current, current_position, current_time)

        # Check for entry
        return self._check_entry(current, current_time, vix)

    def _check_entry(
        self,
        current: pd.Series,
        current_time: datetime,
        vix_value: float,
    ) -> OptionSignal:
        """Check for reversal entry conditions."""
        current_price = current["Close"]
        velocity = current["velocity"]
        velocity_abs = current["velocity_abs"]
        volume_ratio = current["volume_ratio"]
        rsi = current["rsi"]
        exhaustion_score = int(current.get("exhaustion_score", 0))
        session_phase = str(current.get("session_phase", "unknown"))

        # Direction of the explosion (we'll fade this)
        is_bullish_explosion = velocity > 0

        conditions = {}

        # 1. VELOCITY CHECK: Is there an explosion to fade?
        conditions["velocity_impulse"] = velocity_abs >= self.VELOCITY_THRESHOLD

        # 2. VOLUME CHECK: Is there conviction in the move (we need this to fade)
        conditions["volume_spike"] = volume_ratio >= self.VOLUME_SPIKE_THRESHOLD

        # 3. EXHAUSTION CHECK: Is the move exhausted?
        should_trigger, trigger_rule = self.should_trigger_reversal(
            exhaustion_score, session_phase, vix_value
        )
        conditions["exhaustion_triggered"] = should_trigger

        # All conditions must be met
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason=f"Reversal blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "exhaustion_score": exhaustion_score,
                    "session_phase": session_phase,
                    "vix": vix_value,
                    "velocity": velocity,
                }
            )

        # EXHAUSTED EXPLOSION DETECTED! Fade the move (opposite direction)
        # Bullish explosion exhausted -> BUY PUT (fade up)
        # Bearish explosion exhausted -> BUY CALL (fade down)
        option_type = OptionType.PUT if is_bullish_explosion else OptionType.CALL

        # Estimate IV from VIX
        iv = vix_value / 100 if vix_value else 0.25

        # Select slightly OTM contract (cheaper premium for reversal bet)
        selection = ContractSelection.OTM_1 if is_bullish_explosion else ContractSelection.OTM_1
        contract = self.select_contract(
            underlying_price=current_price,
            option_type=option_type,
            selection=selection,
            current_time=current_time,
            iv=iv,
        )

        # Calculate confidence based on exhaustion strength
        confidence = self._calculate_confidence(current, exhaustion_score, session_phase)

        fade_direction = "PUT (fade bullish)" if is_bullish_explosion else "CALL (fade bearish)"
        return OptionSignal(
            signal=OptionSignalType.BUY_PUT if is_bullish_explosion else OptionSignalType.BUY_CALL,
            reason=f"REVERSAL {fade_direction}: ExhScore={exhaustion_score}, {trigger_rule}, RSI={rsi:.0f}",
            confidence=confidence,
            contract=contract,
            time_stop_minutes=self.TIME_STOP_MINUTES,
            profit_target_pct=self.PROFIT_TARGET_PCT,
            stop_loss_pct=self.STOP_LOSS_PCT,
            metadata={
                "exhaustion_score": exhaustion_score,
                "trigger_rule": trigger_rule,
                "session_phase": session_phase,
                "original_velocity": velocity,
                "fade_direction": "PUT" if is_bullish_explosion else "CALL",
                "rsi": rsi,
                "vix": vix_value,
                "volume_ratio": volume_ratio,
                "cumulative_move_5": current.get("cumulative_move_5", 0),
                "contract_strike": contract.strike,
                "contract_delta": contract.delta,
                "estimated_premium": contract.mid_price,
            }
        )

    def _check_exit(
        self,
        current: pd.Series,
        position: OptionPosition,
        current_time: datetime,
    ) -> OptionSignal:
        """
        Check exit conditions for Reversal Scalper.

        Faster exits than Gamma Scalper - reversals happen quick or fail.
        """
        current_price = current["Close"]

        # Calculate current P&L
        pnl_dollars, pnl_pct = self.estimate_current_pnl(
            position=position,
            current_underlying_price=current_price,
            current_time=current_time,
        )

        # Track highest P&L for trailing stop
        if pnl_pct > position.highest_pnl_pct:
            position.highest_pnl_pct = pnl_pct

        # Calculate time held
        held_seconds = (current_time - position.entry_time).total_seconds()
        held_minutes = held_seconds / 60

        # EXIT 1: TIME STOP (STRICT - reversals happen fast)
        if held_minutes >= self.TIME_STOP_MINUTES:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"TIME STOP: {held_minutes:.1f} min >= {self.TIME_STOP_MINUTES} max",
                confidence=0.99,
                metadata={
                    "exit_reason": "TIME_STOP",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 2: PROFIT TARGET (quick scalp)
        if pnl_pct >= self.PROFIT_TARGET_PCT * 100:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"PROFIT TARGET: +{pnl_pct:.1f}% >= +{self.PROFIT_TARGET_PCT*100:.0f}%",
                confidence=0.95,
                metadata={
                    "exit_reason": "PROFIT_TARGET",
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

        # EXIT 4: TRAILING STOP (tighter trail than Gamma Scalper)
        if position.highest_pnl_pct >= self.TRAILING_ACTIVATION_PCT * 100:
            trail_stop = position.highest_pnl_pct - (self.TRAILING_DISTANCE_PCT * 100)
            if pnl_pct < trail_stop:
                return OptionSignal(
                    signal=OptionSignalType.EXIT,
                    reason=f"TRAILING STOP: {pnl_pct:.1f}% < trail {trail_stop:.1f}%",
                    confidence=0.90,
                    metadata={
                        "exit_reason": "TRAILING_STOP",
                        "pnl_pct": pnl_pct,
                        "highest_pnl": position.highest_pnl_pct,
                        "held_minutes": held_minutes,
                    }
                )

        # EXIT 5: REVERSAL CONFIRMED (momentum shifted to our direction)
        # If we faded a bullish move (bought PUT), we want velocity to go negative
        velocity = current["velocity"]
        is_put = position.contract.option_type == OptionType.PUT

        # For PUT positions, we want velocity to be strongly negative (our fade worked)
        # Take profit early if reversal is confirmed
        if is_put and velocity < -self.VELOCITY_THRESHOLD and pnl_pct > 5:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"REVERSAL CONFIRMED: Velocity={velocity*100:.2f}%, P&L=+{pnl_pct:.1f}%",
                confidence=0.85,
                metadata={
                    "exit_reason": "REVERSAL_CONFIRMED",
                    "pnl_pct": pnl_pct,
                    "velocity": velocity,
                    "held_minutes": held_minutes,
                }
            )

        # For CALL positions (fading bearish move), velocity should go positive
        if not is_put and velocity > self.VELOCITY_THRESHOLD and pnl_pct > 5:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"REVERSAL CONFIRMED: Velocity={velocity*100:.2f}%, P&L=+{pnl_pct:.1f}%",
                confidence=0.85,
                metadata={
                    "exit_reason": "REVERSAL_CONFIRMED",
                    "pnl_pct": pnl_pct,
                    "velocity": velocity,
                    "held_minutes": held_minutes,
                }
            )

        # HOLD - show status
        time_remaining = self.TIME_STOP_MINUTES - held_minutes
        return OptionSignal(
            signal=OptionSignalType.HOLD,
            reason=f"Holding reversal: P&L={pnl_pct:+.1f}%, {time_remaining:.1f}min left",
            metadata={
                "pnl_pct": pnl_pct,
                "held_minutes": held_minutes,
                "time_remaining": time_remaining,
                "velocity": current["velocity"],
                "highest_pnl": position.highest_pnl_pct,
            }
        )

    def _calculate_confidence(
        self,
        current: pd.Series,
        exhaustion_score: int,
        session_phase: str,
    ) -> float:
        """Calculate confidence based on exhaustion strength."""
        confidence = 0.50

        # Higher exhaustion score = higher confidence
        if exhaustion_score >= 4:
            confidence += 0.20
        elif exhaustion_score >= 3:
            confidence += 0.15
        elif exhaustion_score >= 2:
            confidence += 0.10

        # Session phase affects confidence
        if session_phase == "midday":
            confidence += 0.10  # Midday reversals are more reliable
        elif session_phase == "close_drive":
            confidence += 0.05  # close_drive also decent

        # Volume declining = higher reversal confidence
        if current.get("volume_declining", False):
            confidence += 0.05

        # RSI extremes = higher reversal confidence
        rsi = current.get("rsi", 50)
        if rsi >= 70 or rsi <= 30:
            confidence += 0.10
        elif rsi >= 65 or rsi <= 35:
            confidence += 0.05

        return min(confidence, 0.85)  # Cap at 85% (reversals are inherently risky)
