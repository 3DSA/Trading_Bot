"""
Delta Surfer Strategy - Riding Steady Trends with Deep ITM Options

The Physics: We want DELTA (Stock-like exposure), not Gamma or Vega.

When market is trending steadily (not explosively):
- Gamma Scalper: Waits for explosions that may never come
- Vega Snap: Waits for panic that may never come
- Delta Surfer: CAPTURES THE TREND with minimal Theta bleed

Why Deep ITM Options?
- Delta 0.70-0.80: Moves almost 1:1 with stock (like owning shares)
- Minimal Theta decay: Mostly intrinsic value
- Less Gamma: Delta doesn't change much (stable exposure)
- Can HOLD for hours (not minutes like Gamma Scalper)

Entry Logic:
- ADX > 25: Confirmed trend (not choppy market)
- Price > EMA 20: Uptrend for calls
- Velocity < 0.2%: Steady move, not explosion (Gamma Scalper's territory)
- Volume rising: Trend has participation

Exit Logic (Trend-Following):
- ATR Trailing Stop: Trail by 3x ATR (not time-based)
- Hard Stop: -15% on premium (protect capital)
- Trend Break: Price crosses below EMA 20
- NO strict time stop: Trends can run for hours

The Math:
- QQQ in steady uptrend: +1% over 2 hours
- Deep ITM Call (Delta 0.75): captures 0.75%
- On $8.00 option: +$0.60 per share = +7.5%
- Theta decay in 2 hours ≈ 1% (manageable vs the gain)

Risk:
- If trend breaks, ATR stop gets us out
- Max loss = 15% of premium (hard stop)
- Deep ITM = smaller percentage swings vs ATM

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


class DeltaSurferStrategy(BaseOptionStrategy):
    """
    Delta Surfer - Ride steady trends with deep ITM options.

    Entry Logic:
        1. TREND: ADX > 25 (confirmed trend)
        2. DIRECTION: Price > EMA 20 (uptrend) for calls
        3. STEADY: Velocity < 0.2% (not explosive - that's Gamma Scalper)
        4. PARTICIPATION: Volume above average (trend has legs)

    Contract Selection:
        - Deep ITM Call/Put (Delta 0.70-0.80)
        - 2-3 DTE preferred (more time to run)
        - Lower Gamma = stable exposure

    Exit Logic (TREND-FOLLOWING):
        1. ATR TRAILING STOP: Trail by 3x ATR
        2. HARD STOP: -15% on premium
        3. TREND BREAK: Price crosses EMA 20
        4. NO time stop (trends can run)

    Why This Works:
        - Captures sustained directional moves
        - Deep ITM = minimal Theta bleed
        - Can hold for hours (vs minutes for ATM)
        - Fills the gap when market is trending but not exploding
    """

    name = "delta_surfer"
    description = "Rides steady trends with deep ITM options"
    version = "1.1.0"  # Tightened entry (ADX 28, DI separation) and improved exit logic

    # Entry parameters - v1.1 TIGHTER for quality trends only
    ADX_THRESHOLD = 28              # Minimum ADX for trend confirmation (was 25)
    ADX_RISING_PERIODS = 3          # ADX must be rising for 3 periods
    VELOCITY_MAX = 0.002            # Max 0.2% velocity (above this = Gamma Scalper territory)
    VOLUME_MIN_RATIO = 1.3          # Volume must be above average (was 1.2)
    RSI_MAX_ENTRY = 65              # Don't chase overbought (was 70)
    RSI_MIN_ENTRY = 35              # Don't catch falling knives (was 30)
    DI_MIN_SEPARATION = 8           # Minimum +DI/-DI separation for clear trend

    # Trend parameters
    EMA_FAST = 9                    # Fast EMA
    EMA_SLOW = 20                   # Slow EMA (trend reference)

    # Exit parameters (TREND-FOLLOWING, not time-based)
    ATR_TRAILING_MULTIPLIER = 3.0  # Trail by 3x ATR
    HARD_STOP_PCT = 0.15            # -15% hard stop on premium
    PROFIT_TARGET_PCT = 0.25        # 25% profit target (optional take profit)
    MAX_HOLD_MINUTES = 240          # 4 hours max (end of day)

    # Contract selection - DEEP ITM
    PREFERRED_DTE = 2               # 2-DTE for more time
    TARGET_DELTA = 0.75             # Deep ITM

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicators for trend detection.

        Key indicators:
        - ADX: Trend strength
        - EMA 9/20: Trend direction
        - ATR: For trailing stop
        - Velocity: To filter out explosions (Gamma Scalper territory)
        """
        df = df.copy()

        # Price velocity (to distinguish from Gamma Scalper)
        df["velocity"] = (df["Close"] - df["Open"]) / df["Open"]
        df["velocity_abs"] = df["velocity"].abs()

        # EMAs for trend
        df["ema_9"] = df["Close"].ewm(span=self.EMA_FAST, adjust=False).mean()
        df["ema_20"] = df["Close"].ewm(span=self.EMA_SLOW, adjust=False).mean()

        # ADX calculation
        df = self._calculate_adx(df)

        # ATR for trailing stop
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=14).mean()
        df["atr_pct"] = df["atr"] / df["Close"]

        # Volume analysis
        df["volume_sma"] = df["Volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["Volume"] / df["volume_sma"]

        # RSI for overbought/oversold
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Trend direction
        df["uptrend"] = (df["Close"] > df["ema_20"]) & (df["ema_9"] > df["ema_20"])
        df["downtrend"] = (df["Close"] < df["ema_20"]) & (df["ema_9"] < df["ema_20"])

        # Forward fill and drop NaN
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX (Average Directional Index)."""
        # True Range
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = df["High"].diff()
        minus_dm = -df["Low"].diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX and ADX
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[OptionPosition] = None,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Generate Delta Surfer signal.

        Args:
            df: Prepared DataFrame with trend indicators
            current_position: Current option position (if any)
            vix_value: Current VIX (affects IV estimate)

        Returns:
            OptionSignal with contract spec and trend-following exit rules
        """
        if df.empty or len(df) < 30:
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason="Insufficient data for Delta Surfer"
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
            return self._check_exit(df, current, current_position, current_time)

        # Check for entry
        return self._check_entry(current, current_time, vix_value)

    def _check_entry(
        self,
        current: pd.Series,
        current_time: datetime,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """Check for steady trend entry conditions."""
        current_price = current["Close"]
        adx = current["adx"]
        velocity_abs = current["velocity_abs"]
        velocity = current["velocity"]
        volume_ratio = current["volume_ratio"]
        rsi = current["rsi"]
        uptrend = current["uptrend"]
        downtrend = current["downtrend"]
        plus_di = current["plus_di"]
        minus_di = current["minus_di"]

        conditions = {}

        # 1. TREND CHECK: Is there a confirmed trend?
        conditions["trend_confirmed"] = adx >= self.ADX_THRESHOLD

        # 2. STEADY CHECK: Is this NOT an explosion? (Gamma Scalper's territory)
        conditions["steady_move"] = velocity_abs < self.VELOCITY_MAX

        # 3. VOLUME CHECK: Is there participation?
        conditions["volume_ok"] = volume_ratio >= self.VOLUME_MIN_RATIO

        # 4. DIRECTION CHECK: Are we in a clear trend with DI separation?
        di_diff = abs(plus_di - minus_di)
        is_bullish = uptrend and plus_di > minus_di and di_diff >= self.DI_MIN_SEPARATION
        is_bearish = downtrend and minus_di > plus_di and di_diff >= self.DI_MIN_SEPARATION
        conditions["clear_direction"] = is_bullish or is_bearish

        # 5. NOT EXHAUSTED: Room to run?
        if is_bullish:
            conditions["not_exhausted"] = rsi < self.RSI_MAX_ENTRY
        elif is_bearish:
            conditions["not_exhausted"] = rsi > self.RSI_MIN_ENTRY
        else:
            conditions["not_exhausted"] = False

        # 6. ADX RISING: Trend is strengthening (not weakening)
        # This requires looking back at previous ADX values
        # For now, we use the threshold as a proxy

        # All conditions must be met
        all_met = all(conditions.values())

        if not all_met:
            failed = [k for k, v in conditions.items() if not v]
            return OptionSignal(
                signal=OptionSignalType.HOLD,
                reason=f"DeltaSurf blocked: {', '.join(failed)}",
                metadata={
                    "conditions": conditions,
                    "adx": adx,
                    "velocity_abs": velocity_abs,
                    "volume_ratio": volume_ratio,
                    "rsi": rsi,
                }
            )

        # TREND DETECTED! Prepare deep ITM contract
        option_type = OptionType.CALL if is_bullish else OptionType.PUT

        # Estimate IV from VIX
        iv = (vix_value / 100) if vix_value else 0.20

        # Select deep ITM contract for maximum Delta
        contract = self._select_deep_itm_contract(
            underlying_price=current_price,
            option_type=option_type,
            current_time=current_time,
            iv=iv,
        )

        # Calculate confidence
        confidence = self._calculate_confidence(current, is_bullish)

        direction = "BULLISH" if is_bullish else "BEARISH"
        return OptionSignal(
            signal=OptionSignalType.BUY_CALL if is_bullish else OptionSignalType.BUY_PUT,
            reason=f"TREND {direction}: ADX={adx:.1f}, Vel={velocity*100:.2f}%, RSI={rsi:.0f}",
            confidence=confidence,
            contract=contract,
            time_stop_minutes=self.MAX_HOLD_MINUTES,  # Much longer than Gamma Scalper
            profit_target_pct=self.PROFIT_TARGET_PCT,
            stop_loss_pct=self.HARD_STOP_PCT,
            metadata={
                "adx": adx,
                "velocity": velocity,
                "velocity_abs": velocity_abs,
                "volume_ratio": volume_ratio,
                "rsi": rsi,
                "atr": current["atr"],
                "direction": direction,
                "contract_strike": contract.strike,
                "contract_delta": contract.delta,
                "estimated_premium": contract.mid_price,
            }
        )

    def _select_deep_itm_contract(
        self,
        underlying_price: float,
        option_type: OptionType,
        current_time: datetime,
        iv: float,
    ) -> ContractSpec:
        """
        Select a deep ITM contract for Delta Surfer.

        Deep ITM characteristics:
        - Delta 0.70-0.80
        - Minimal Theta decay
        - Can hold for hours
        """
        # For deep ITM, we go 3-4 strikes ITM
        if option_type == OptionType.CALL:
            strike = round(underlying_price) - 4  # 4 strikes ITM
        else:
            strike = round(underlying_price) + 4  # 4 strikes ITM

        # Use longer DTE (2 days)
        expiry = self._get_next_expiry(current_time, self.PREFERRED_DTE)
        dte = max(1, (expiry.date() - current_time.date()).days)

        # Deep ITM Greeks
        delta = 0.75 if option_type == OptionType.CALL else -0.75
        gamma = 0.02  # Much lower gamma than ATM
        theta = -0.03  # Much lower theta decay
        vega = 0.05   # Less sensitive to IV changes

        # Estimate option price (mostly intrinsic for deep ITM)
        if option_type == OptionType.CALL:
            intrinsic = max(0, underlying_price - strike)
        else:
            intrinsic = max(0, strike - underlying_price)

        time_value = underlying_price * 0.004 * iv  # Small time value for deep ITM
        mid_price = intrinsic + time_value
        spread = mid_price * 0.02  # ~2% spread

        # Build contract symbol
        expiry_str = expiry.strftime("%y%m%d")
        type_char = "C" if option_type == OptionType.CALL else "P"
        strike_str = f"{int(strike * 1000):08d}"
        symbol = f"{self.underlying}{expiry_str}{type_char}{strike_str}"

        return ContractSpec(
            underlying=self.underlying,
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            dte=dte,
            delta=abs(delta),
            gamma=gamma,
            theta=theta,
            vega=vega,
            bid=max(0.01, mid_price - spread/2),
            ask=mid_price + spread/2,
            mid_price=mid_price,
            implied_volatility=iv,
            symbol=symbol,
        )

    def _check_exit(
        self,
        df: pd.DataFrame,
        current: pd.Series,
        position: OptionPosition,
        current_time: datetime,
    ) -> OptionSignal:
        """
        Check exit conditions for Delta Surfer.

        EXIT DISCIPLINE FOR TREND-FOLLOWING:
        1. HARD STOP (-15%) - Protect capital (check first)
        2. PROFIT TARGET (25%) - Take profit
        3. UNDERLYING ATR STOP - Trail by 2x ATR on underlying
        4. ADX BREAKDOWN - Trend losing strength
        5. TREND BREAK - Price crosses EMA
        6. MAX TIME (4 hours) - End of day
        """
        current_price = current["Close"]
        atr = current["atr"]
        adx = current["adx"]

        # Calculate current P&L
        pnl_dollars, pnl_pct = self.estimate_current_pnl(
            position=position,
            current_underlying_price=current_price,
            current_time=current_time,
        )

        # Track highest underlying price for ATR trailing
        is_call = position.contract.option_type == OptionType.CALL

        # Track best price seen (not P&L based)
        if not hasattr(position, 'best_underlying_price'):
            position.best_underlying_price = position.entry_underlying_price

        if is_call:
            position.best_underlying_price = max(position.best_underlying_price, current_price)
        else:
            position.best_underlying_price = min(position.best_underlying_price, current_price)

        # Track highest P&L for reference
        if pnl_pct > position.highest_pnl_pct:
            position.highest_pnl_pct = pnl_pct

        # Calculate time held
        held_seconds = (current_time - position.entry_time).total_seconds()
        held_minutes = held_seconds / 60

        # EXIT 1: HARD STOP (protect capital)
        if pnl_pct <= -self.HARD_STOP_PCT * 100:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"HARD STOP: {pnl_pct:.1f}% <= -{self.HARD_STOP_PCT*100:.0f}%",
                confidence=0.95,
                metadata={
                    "exit_reason": "HARD_STOP",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 2: PROFIT TARGET
        if pnl_pct >= self.PROFIT_TARGET_PCT * 100:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"PROFIT TARGET: +{pnl_pct:.1f}% >= +{self.PROFIT_TARGET_PCT*100:.0f}%",
                confidence=0.90,
                metadata={
                    "exit_reason": "PROFIT_TARGET",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 3: ATR TRAILING STOP (on underlying price, not P&L)
        # Only activate after held for at least 5 minutes and have some gain
        atr_trail = self.ATR_TRAILING_MULTIPLIER * atr

        if held_minutes >= 5 and pnl_pct > 3:  # At least 5 min held and 3% gain
            if is_call:
                # For calls, trail below the best price
                trail_stop_price = position.best_underlying_price - atr_trail
                if current_price < trail_stop_price:
                    return OptionSignal(
                        signal=OptionSignalType.EXIT,
                        reason=f"ATR TRAIL: ${current_price:.2f} < ${trail_stop_price:.2f}",
                        confidence=0.85,
                        metadata={
                            "exit_reason": "ATR_TRAILING_STOP",
                            "pnl_pct": pnl_pct,
                            "trail_stop_price": trail_stop_price,
                            "held_minutes": held_minutes,
                        }
                    )
            else:
                # For puts, trail above the best (lowest) price
                trail_stop_price = position.best_underlying_price + atr_trail
                if current_price > trail_stop_price:
                    return OptionSignal(
                        signal=OptionSignalType.EXIT,
                        reason=f"ATR TRAIL: ${current_price:.2f} > ${trail_stop_price:.2f}",
                        confidence=0.85,
                        metadata={
                            "exit_reason": "ATR_TRAILING_STOP",
                            "pnl_pct": pnl_pct,
                            "trail_stop_price": trail_stop_price,
                            "held_minutes": held_minutes,
                        }
                    )

        # EXIT 4: ADX BREAKDOWN (trend losing strength)
        if held_minutes >= 10 and adx < 20:  # ADX dropped below trend threshold
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"ADX BREAKDOWN: ADX={adx:.1f} < 20 (trend weakening)",
                confidence=0.80,
                metadata={
                    "exit_reason": "ADX_BREAKDOWN",
                    "pnl_pct": pnl_pct,
                    "adx": adx,
                    "held_minutes": held_minutes,
                }
            )

        # EXIT 5: TREND BREAK (price crosses EMA)
        # Only if we've been in trade for a while and not already very profitable
        if held_minutes >= 15 and pnl_pct < 10:
            if is_call:
                # For calls, exit if we break below EMA 20
                if current_price < current["ema_20"]:
                    return OptionSignal(
                        signal=OptionSignalType.EXIT,
                        reason=f"TREND BREAK: Price ${current_price:.2f} < EMA20 ${current['ema_20']:.2f}",
                        confidence=0.80,
                        metadata={
                            "exit_reason": "TREND_BREAK",
                            "pnl_pct": pnl_pct,
                            "held_minutes": held_minutes,
                        }
                    )
            else:
                # For puts, exit if we break above EMA 20
                if current_price > current["ema_20"]:
                    return OptionSignal(
                        signal=OptionSignalType.EXIT,
                        reason=f"TREND BREAK: Price ${current_price:.2f} > EMA20 ${current['ema_20']:.2f}",
                        confidence=0.80,
                        metadata={
                            "exit_reason": "TREND_BREAK",
                            "pnl_pct": pnl_pct,
                            "held_minutes": held_minutes,
                        }
                    )

        # EXIT 5: MAX TIME (end of day protection)
        if held_minutes >= self.MAX_HOLD_MINUTES:
            return OptionSignal(
                signal=OptionSignalType.EXIT,
                reason=f"MAX TIME: {held_minutes:.0f}min >= {self.MAX_HOLD_MINUTES}min",
                confidence=0.90,
                metadata={
                    "exit_reason": "MAX_TIME",
                    "pnl_pct": pnl_pct,
                    "held_minutes": held_minutes,
                }
            )

        # HOLD - show status
        adx = current["adx"]
        return OptionSignal(
            signal=OptionSignalType.HOLD,
            reason=f"Surfing: P&L={pnl_pct:+.1f}%, ADX={adx:.0f}, {held_minutes:.0f}min",
            metadata={
                "pnl_pct": pnl_pct,
                "held_minutes": held_minutes,
                "adx": adx,
                "highest_pnl": position.highest_pnl_pct,
            }
        )

    def _calculate_confidence(self, current: pd.Series, is_bullish: bool) -> float:
        """Calculate confidence based on trend strength."""
        confidence = 0.50

        adx = current["adx"]
        volume_ratio = current["volume_ratio"]
        rsi = current["rsi"]
        plus_di = current["plus_di"]
        minus_di = current["minus_di"]

        # Stronger ADX = higher confidence
        if adx >= 35:
            confidence += 0.15
        elif adx >= 30:
            confidence += 0.10
        elif adx >= 25:
            confidence += 0.05

        # Higher volume = higher confidence
        if volume_ratio >= 2.0:
            confidence += 0.10
        elif volume_ratio >= 1.5:
            confidence += 0.05

        # Clear DI separation = higher confidence
        di_diff = abs(plus_di - minus_di)
        if di_diff >= 15:
            confidence += 0.10
        elif di_diff >= 10:
            confidence += 0.05

        # RSI sweet spot
        if is_bullish and 40 <= rsi <= 60:
            confidence += 0.05
        elif not is_bullish and 40 <= rsi <= 60:
            confidence += 0.05

        return min(confidence, 0.85)  # Cap at 85% (trends can fail)
