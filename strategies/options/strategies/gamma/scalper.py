"""
Gamma Scalper Strategy - Capturing Explosive Moves

The Physics: We want GAMMA (Acceleration), not trend.

Stock momentum strategies wait for trends (slow, sustained moves).
In options, waiting = losing (Theta decay).

Instead, we hunt for EXPLOSIONS:
- Price moves > 0.3% in 1 minute = IMPULSE
- Volume > 2x average = CONVICTION
- Buy ATM Call (highest Gamma)
- Exit in 10 minutes MAX

Why 10 Minutes?
- Gamma works best in the first few minutes of a move
- After that, either the explosion continues (take profit)
- Or it stops, and Theta starts eating your premium
- We're scalping the IMPULSE, not riding the trend

The Math:
- QQQ moves 0.5% = $2.50 on $500 stock
- ATM Call with 0.50 Delta = $1.25 move
- But Gamma accelerates: actual move ≈ $1.50-2.00
- On a $3.00 option = 50-66% gain
- Theta decay in 10 min ≈ 1-2% (negligible vs the gain)

Risk:
- If explosion fizzles, we're out in 10 min with small loss
- Max loss = premium paid (defined risk)
- No overnight holds, no Theta bleed

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


class GammaScalperStrategy(BaseOptionStrategy):
    """
    Gamma Scalper - Capture explosive moves with options leverage.

    Entry Logic:
        1. VELOCITY: Price moves > 0.3% in 1 candle (impulse detected)
        2. VOLUME: Current volume > 2x 20-period average (conviction)
        3. DIRECTION: Close > Open (bullish impulse) for calls
        4. NOT EXHAUSTED: RSI < 75 (room to run)

    Contract Selection:
        - ATM Call (Delta ~0.50, highest Gamma)
        - 1-DTE preferred (maximum Gamma exposure)

    Exit Logic (STRICT TIME DISCIPLINE):
        1. TIME STOP: 10 minutes MAX (non-negotiable)
        2. PROFIT TARGET: 15% gain (take the money)
        3. STOP LOSS: 30% loss (cut and run)
        4. MOMENTUM FADE: Velocity reverses (explosion over)

    Why This Works:
        - Gamma gives us CONVEXITY on explosive moves
        - 10-minute window limits Theta damage
        - We don't need to predict direction long-term
        - Just catch the impulse and get out
    """

    name = "gamma_scalper"
    description = "Captures explosive moves using Gamma acceleration"
    version = "1.1.0"  # Tightened entry parameters

    # Entry parameters - v1.1 TIGHTER for quality explosions only
    VELOCITY_THRESHOLD = 0.004      # 0.4% move in 1 candle (was 0.3%)
    VOLUME_SPIKE_THRESHOLD = 2.5    # 2.5x average volume (was 2.0x)
    RSI_MAX_ENTRY = 70              # Don't chase overbought (was 75)
    RSI_MIN_ENTRY = 30              # Don't catch falling knives (was 25)
    MIN_ATR_MULTIPLIER = 2.0        # Move must be > 2x ATR (was 1.5x)

    # Exit parameters (STRICT)
    TIME_STOP_MINUTES = 10          # Absolute max hold
    PROFIT_TARGET_PCT = 0.15        # 15% profit target
    STOP_LOSS_PCT = 0.30            # 30% stop loss
    TRAILING_ACTIVATION_PCT = 0.10  # Activate trail at 10%
    TRAILING_DISTANCE_PCT = 0.05    # Trail by 5%

    # Contract selection
    PREFERRED_DTE = 1               # 1-DTE for maximum Gamma
    TARGET_DELTA = 0.50             # ATM for highest Gamma

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicators for detecting explosive moves.

        Key indicators:
        - Velocity: (Close - Open) / Open per candle
        - Volume ratio: Current / 20-period average
        - ATR: To validate move significance
        - RSI: To avoid exhausted moves

        Exhaustion indicators (for reversal_scalper routing):
        - cumulative_move_5: Total move over last 5 bars (late entry detection)
        - bars_since_explosion: How long since velocity first exceeded threshold
        - volume_declining: Whether volume is fading during the move
        - prior_bar_velocity: Previous bar's velocity (momentum building vs reversal)
        - session_phase: open_drive, midday, close_drive
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

        # EMA for trend context (not for entry, just info)
        df["ema_9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["Close"].ewm(span=21, adjust=False).mean()

        # =============================================================================
        # NEW: Exhaustion Detection Variables (for reversal_scalper routing)
        # =============================================================================

        # 1. CUMULATIVE MOVE: Total move over last 5 bars
        # If price has already moved significantly, we're catching the end not the start
        df["cumulative_move_5"] = df["Close"].pct_change(periods=5)

        # 2. PRIOR BAR BEHAVIOR: What happened before this bar
        df["prior_bar_velocity"] = df["velocity"].shift(1)
        df["prior_bar_direction"] = np.sign(df["velocity"].shift(1))

        # Check if prior bar was also explosive in same direction (momentum building)
        # vs opposite direction (potential reversal forming)
        df["momentum_building"] = (
            (df["velocity"] > 0) & (df["prior_bar_velocity"] > self.VELOCITY_THRESHOLD / 2)
        ) | (
            (df["velocity"] < 0) & (df["prior_bar_velocity"] < -self.VELOCITY_THRESHOLD / 2)
        )

        # 3. VOLUME PATTERN: Is volume increasing or declining during the move
        df["prior_volume_ratio"] = df["volume_ratio"].shift(1)
        df["volume_declining"] = df["volume_ratio"] < df["prior_volume_ratio"]

        # High volume often precedes reversals (exhaustion volume)
        df["exhaustion_volume"] = df["volume_ratio"] >= 8.0

        # 4. BARS SINCE EXPLOSION START
        # Track how many consecutive bars have had elevated velocity
        explosion_active = df["velocity_abs"] >= self.VELOCITY_THRESHOLD / 2
        df["bars_in_explosion"] = explosion_active.groupby(
            (~explosion_active).cumsum()
        ).cumsum()

        # 5. SESSION PHASE: Different behavior at different times
        # open_drive (9:30-10:00): More continuation
        # midday (10:00-14:30): More chop/reversal
        # close_drive (14:30-16:00): More directional
        def get_session_phase(timestamp):
            if hasattr(timestamp, 'hour'):
                hour = timestamp.hour
                minute = timestamp.minute
                time_minutes = hour * 60 + minute

                # Market hours in Eastern Time (assuming data is in ET)
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

        # 6. EXHAUSTION SCORE: Composite score for reversal likelihood
        # Higher score = more likely to reverse (good for reversal_scalper)
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

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[OptionPosition] = None,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Generate Gamma Scalper signal.

        Args:
            df: Prepared DataFrame with velocity indicators
            current_position: Current option position (if any)
            vix_value: Current VIX (affects IV estimate)

        Returns:
            OptionSignal with contract spec and strict exit rules
        """
        if df.empty or len(df) < 20:
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason="Insufficient data for Gamma Scalper"
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
            return self._check_exit(current, current_position, current_time)

        # Check for entry
        return self._check_entry(current, current_time, vix_value)

    def _check_entry(
        self,
        current: pd.Series,
        current_time: datetime,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """Check for explosive move entry conditions."""
        current_price = current["Close"]
        velocity = current["velocity"]
        velocity_abs = current["velocity_abs"]
        volume_ratio = current["volume_ratio"]
        rsi = current["rsi"]
        atr_pct = current["atr_pct"]

        # Determine direction
        is_bullish = velocity > 0

        conditions = {}

        # 1. VELOCITY CHECK: Is this an impulse?
        conditions["velocity_impulse"] = velocity_abs >= self.VELOCITY_THRESHOLD

        # 2. VOLUME CHECK: Is there conviction?
        conditions["volume_spike"] = volume_ratio >= self.VOLUME_SPIKE_THRESHOLD

        # 3. MOVE SIGNIFICANCE: Is this move meaningful vs noise?
        conditions["significant_move"] = velocity_abs >= atr_pct * self.MIN_ATR_MULTIPLIER

        # 4. NOT EXHAUSTED: Room to run?
        if is_bullish:
            conditions["not_exhausted"] = rsi < self.RSI_MAX_ENTRY
        else:
            conditions["not_exhausted"] = rsi > self.RSI_MIN_ENTRY

        # All conditions must be met
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason=f"Gamma blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "velocity": velocity,
                    "velocity_abs": velocity_abs,
                    "volume_ratio": volume_ratio,
                    "rsi": rsi,
                }
            )

        # EXPLOSION DETECTED! Prepare contract
        option_type = OptionType.CALL if is_bullish else OptionType.PUT

        # Estimate IV from VIX (rough approximation)
        iv = (vix_value / 100) if vix_value else 0.25

        # Select ATM contract for maximum Gamma
        contract = self.select_contract(
            underlying_price=current_price,
            option_type=option_type,
            selection=ContractSelection.ATM,
            current_time=current_time,
            iv=iv,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(current, is_bullish)

        direction = "BULLISH" if is_bullish else "BEARISH"
        return OptionSignal(
            signal=OptionSignalType.BUY_CALL if is_bullish else OptionSignalType.BUY_PUT,
            reason=f"EXPLOSION {direction}: Vel={velocity*100:.2f}%, Vol={volume_ratio:.1f}x, RSI={rsi:.0f}",
            confidence=confidence,
            contract=contract,
            time_stop_minutes=self.TIME_STOP_MINUTES,
            profit_target_pct=self.PROFIT_TARGET_PCT,
            stop_loss_pct=self.STOP_LOSS_PCT,
            metadata={
                "velocity": velocity,
                "velocity_abs": velocity_abs,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "atr_pct": atr_pct,
                "direction": direction,
                "contract_strike": contract.strike,
                "contract_delta": contract.delta,
                "contract_gamma": contract.gamma,
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
        Check exit conditions for Gamma Scalper.

        EXIT DISCIPLINE IS EVERYTHING IN OPTIONS.
        We have strict rules:
        1. TIME STOP (10 min) - Non-negotiable
        2. PROFIT TARGET (15%) - Take the money
        3. STOP LOSS (30%) - Cut losses
        4. MOMENTUM FADE - Explosion over
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

        # EXIT 1: TIME STOP (NON-NEGOTIABLE)
        if held_minutes >= self.TIME_STOP_MINUTES:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"TIME STOP: {held_minutes:.1f} min >= {self.TIME_STOP_MINUTES} max",
                confidence=0.99,  # Must exit
                metadata={
                    "exit_reason": "TIME_STOP",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 2: PROFIT TARGET
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

        # EXIT 4: TRAILING STOP (once activated)
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

        # EXIT 5: MOMENTUM FADE (velocity reversed significantly)
        velocity = current["velocity"]
        is_call = position.contract.option_type == OptionType.CALL

        # If we're long calls and velocity goes negative, momentum is fading
        if is_call and velocity < -self.VELOCITY_THRESHOLD / 2:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"MOMENTUM FADE: Velocity reversed to {velocity*100:.2f}%",
                confidence=0.75,
                metadata={
                    "exit_reason": "MOMENTUM_FADE",
                    "pnl_pct": pnl_pct,
                    "velocity": velocity,
                    "held_minutes": held_minutes,
                }
            )

        # If we're long puts and velocity goes positive
        if not is_call and velocity > self.VELOCITY_THRESHOLD / 2:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"MOMENTUM FADE: Velocity reversed to {velocity*100:.2f}%",
                confidence=0.75,
                metadata={
                    "exit_reason": "MOMENTUM_FADE",
                    "pnl_pct": pnl_pct,
                    "velocity": velocity,
                    "held_minutes": held_minutes,
                }
            )

        # HOLD - show status
        time_remaining = self.TIME_STOP_MINUTES - held_minutes
        return OptionSignal(
            signal=OptionSignalType.HOLD,
            reason=f"Holding: P&L={pnl_pct:+.1f}%, {time_remaining:.1f}min left",
            metadata={
                "pnl_pct": pnl_pct,
                "held_minutes": held_minutes,
                "time_remaining": time_remaining,
                "velocity": current["velocity"],
                "highest_pnl": position.highest_pnl_pct,
            }
        )

    def _calculate_confidence(self, current: pd.Series, is_bullish: bool) -> float:
        """Calculate confidence based on explosion strength."""
        confidence = 0.50

        velocity_abs = current["velocity_abs"]
        volume_ratio = current["volume_ratio"]
        rsi = current["rsi"]

        # Stronger velocity = higher confidence
        if velocity_abs >= 0.005:  # 0.5% move
            confidence += 0.15
        elif velocity_abs >= 0.004:
            confidence += 0.10

        # Higher volume = higher confidence
        if volume_ratio >= 3.0:
            confidence += 0.15
        elif volume_ratio >= 2.5:
            confidence += 0.10

        # RSI sweet spot (not extreme)
        if is_bullish and 40 <= rsi <= 65:
            confidence += 0.10
        elif not is_bullish and 35 <= rsi <= 60:
            confidence += 0.10

        # Trend alignment bonus (EMA check)
        if is_bullish and current["ema_9"] > current["ema_21"]:
            confidence += 0.05
        elif not is_bullish and current["ema_9"] < current["ema_21"]:
            confidence += 0.05

        return min(confidence, 0.90)  # Cap at 90%
