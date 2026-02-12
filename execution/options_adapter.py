"""
Options Adapter - Translates Equity Signals to Options Contracts

This module acts as a "shadow bot" that converts TQQQ/SQQQ buy signals
into QQQ options trades. It simulates what the P&L would be if we
traded options instead of shares.

Key Concepts:
    - TQQQ BUY -> QQQ CALL (bullish bet)
    - SQQQ BUY -> QQQ PUT (bearish bet)
    - Uses 1-DTE options for high leverage with reduced gamma risk
    - Targets ~0.40-0.45 Delta (slightly OTM for convexity)

Leverage Approximation:
    - TQQQ: 3x leverage
    - Options (1-DTE, 0.45 Delta): ~30-50x leverage
    - A 1% move in QQQ = ~30-50% move in option value

Author: Bi-Cameral Quant Team
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict
import math
import pytz


class OptionType(Enum):
    """Option contract type."""
    CALL = "CALL"
    PUT = "PUT"


@dataclass
class OptionContract:
    """
    Represents a simulated options contract.

    In production, this would come from the options chain API.
    For simulation, we calculate theoretical values.
    """
    underlying: str  # QQQ
    option_type: OptionType
    strike: float
    expiry: datetime
    delta: float  # Price sensitivity to underlying
    gamma: float  # Rate of change of delta
    theta: float  # Time decay per day (negative)
    vega: float   # Sensitivity to IV changes
    implied_volatility: float
    bid: float
    ask: float

    @property
    def mid_price(self) -> float:
        return (self.bid + self.ask) / 2

    @property
    def spread_pct(self) -> float:
        return (self.ask - self.bid) / self.mid_price * 100 if self.mid_price > 0 else 0


@dataclass
class OptionPosition:
    """Tracks an open options position."""
    contract: OptionContract
    entry_price: float  # Premium paid per contract
    entry_time: datetime
    quantity: int  # Number of contracts (each = 100 shares)
    entry_underlying_price: float  # QQQ price at entry

    # For shadow tracking
    original_signal_symbol: str  # TQQQ or SQQQ
    original_signal_reason: str
    strategy_name: str


@dataclass
class OptionTrade:
    """Completed options trade for logging."""
    contract: OptionContract
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    entry_underlying_price: float
    exit_underlying_price: float
    underlying_move_pct: float
    leverage_achieved: float  # Actual leverage = option_pnl_pct / underlying_pnl_pct
    original_signal_symbol: str
    strategy_name: str
    exit_reason: str


class OptionsAdapter:
    """
    Translates equity signals to options contracts.

    This adapter intercepts buy signals for TQQQ/SQQQ and converts them
    to QQQ option trades. It can run in two modes:

    1. SIMULATION: Estimates option P&L using simplified Greeks model
    2. LIVE: Would fetch real option chains and execute (requires Alpaca Options API)

    Usage:
        adapter = OptionsAdapter(mode="simulation")

        # When strategy generates a signal
        if signal.signal == SignalType.BUY and signal.symbol == "TQQQ":
            option_contract = adapter.translate_signal(
                symbol="TQQQ",
                underlying_price=qqq_price,
                signal_confidence=signal.confidence
            )

        # Track shadow P&L
        shadow_pnl = adapter.simulate_option_pnl(
            entry_underlying=450.00,
            current_underlying=454.50,  # +1% move
            entry_option_price=2.50,
            delta=0.45,
            time_held_hours=0.5
        )
    """

    # Default Greeks for simulation (1-DTE ATM options)
    DEFAULT_DELTA = 0.45  # Slightly OTM
    DEFAULT_GAMMA = 0.08  # High gamma for 1-DTE
    DEFAULT_THETA = -0.15  # ~15% decay per day
    DEFAULT_VEGA = 0.10
    DEFAULT_IV = 0.25  # 25% implied volatility

    # Leverage assumptions
    LEVERAGE_MULTIPLIER = 30  # Conservative estimate for 1-DTE options

    # Contract sizing
    SHARES_PER_CONTRACT = 100

    def __init__(
        self,
        mode: str = "simulation",
        target_delta: float = 0.45,
        preferred_dte: int = 1,
        max_spread_pct: float = 5.0,
    ):
        """
        Initialize the options adapter.

        Args:
            mode: "simulation" or "live"
            target_delta: Target delta for contract selection (0.40-0.50 recommended)
            preferred_dte: Preferred days to expiration (1 = next day)
            max_spread_pct: Maximum bid-ask spread to accept (%)
        """
        self.mode = mode
        self.target_delta = target_delta
        self.preferred_dte = preferred_dte
        self.max_spread_pct = max_spread_pct

        # Shadow tracking
        self.shadow_position: Optional[OptionPosition] = None
        self.shadow_trades: List[OptionTrade] = []
        self.shadow_pnl_total: float = 0.0

    def translate_signal(
        self,
        symbol: str,
        underlying_price: float,
        signal_confidence: float = 0.5,
        current_time: Optional[datetime] = None,
        strategy_name: str = "unknown",
        signal_reason: str = "",
    ) -> Optional[OptionContract]:
        """
        Translate an equity signal to an options contract.

        Args:
            symbol: Original signal symbol (TQQQ or SQQQ)
            underlying_price: Current QQQ price
            signal_confidence: Strategy's confidence (0-1)
            current_time: Current timestamp
            strategy_name: Name of strategy generating signal
            signal_reason: Reason for the signal

        Returns:
            OptionContract if translation successful, None otherwise
        """
        current_time = current_time or datetime.now(pytz.utc)

        # Determine option type based on signal
        if symbol.upper() in ["TQQQ", "QQQ"]:
            option_type = OptionType.CALL
        elif symbol.upper() in ["SQQQ"]:
            option_type = OptionType.PUT
        else:
            print(f"[OPTIONS] Unknown symbol: {symbol}")
            return None

        # Calculate strike price (slightly OTM for convexity)
        if option_type == OptionType.CALL:
            # OTM call = strike above current price
            strike = self._round_to_strike(underlying_price * 1.005)  # 0.5% OTM
        else:
            # OTM put = strike below current price
            strike = self._round_to_strike(underlying_price * 0.995)  # 0.5% OTM

        # Calculate expiry (next trading day for 1-DTE)
        expiry = self._get_next_expiry(current_time, self.preferred_dte)

        # Simulate option price and Greeks
        contract = self._simulate_contract(
            underlying_price=underlying_price,
            strike=strike,
            option_type=option_type,
            expiry=expiry,
            current_time=current_time,
        )

        # Log the translation
        print(f"[OPTIONS] Signal Translation:")
        print(f"  Original: {symbol} @ ${underlying_price:.2f}")
        print(f"  Translated: QQQ {option_type.value} ${strike:.0f} exp {expiry.strftime('%m/%d')}")
        print(f"  Delta: {contract.delta:.2f}, Premium: ${contract.mid_price:.2f}")
        print(f"  Strategy: {strategy_name}")

        return contract

    def open_shadow_position(
        self,
        contract: OptionContract,
        underlying_price: float,
        current_time: datetime,
        original_symbol: str,
        signal_reason: str,
        strategy_name: str,
        capital_allocation: float = 1000.0,
    ) -> OptionPosition:
        """
        Open a shadow options position for tracking.

        Args:
            contract: The option contract to "buy"
            underlying_price: Current QQQ price
            current_time: Entry timestamp
            original_symbol: Original signal symbol (TQQQ/SQQQ)
            signal_reason: Why the signal was generated
            strategy_name: Which strategy generated it
            capital_allocation: How much capital to allocate

        Returns:
            OptionPosition tracking object
        """
        # Calculate number of contracts we could buy
        contract_cost = contract.mid_price * self.SHARES_PER_CONTRACT
        quantity = max(1, int(capital_allocation / contract_cost))

        position = OptionPosition(
            contract=contract,
            entry_price=contract.mid_price,
            entry_time=current_time,
            quantity=quantity,
            entry_underlying_price=underlying_price,
            original_signal_symbol=original_symbol,
            original_signal_reason=signal_reason,
            strategy_name=strategy_name,
        )

        self.shadow_position = position

        total_cost = contract.mid_price * quantity * self.SHARES_PER_CONTRACT
        print(f"[OPTIONS] Shadow Position Opened:")
        print(f"  {quantity}x QQQ {contract.option_type.value} ${contract.strike:.0f}")
        print(f"  Entry: ${contract.mid_price:.2f} x {quantity} = ${total_cost:.2f}")

        return position

    def close_shadow_position(
        self,
        current_underlying_price: float,
        current_time: datetime,
        exit_reason: str = "signal_exit",
    ) -> Optional[OptionTrade]:
        """
        Close the shadow position and record the trade.

        Args:
            current_underlying_price: Current QQQ price
            current_time: Exit timestamp
            exit_reason: Why we're closing

        Returns:
            OptionTrade record if position was open
        """
        if self.shadow_position is None:
            return None

        pos = self.shadow_position

        # Calculate time held
        time_held = (current_time - pos.entry_time).total_seconds() / 3600  # hours

        # Simulate exit price
        exit_price = self.simulate_option_price(
            entry_underlying=pos.entry_underlying_price,
            current_underlying=current_underlying_price,
            entry_option_price=pos.entry_price,
            delta=pos.contract.delta,
            time_held_hours=time_held,
            option_type=pos.contract.option_type,
        )

        # Calculate P&L
        pnl_per_contract = (exit_price - pos.entry_price) * self.SHARES_PER_CONTRACT
        total_pnl = pnl_per_contract * pos.quantity
        entry_cost = pos.entry_price * self.SHARES_PER_CONTRACT * pos.quantity
        pnl_pct = (total_pnl / entry_cost) * 100 if entry_cost > 0 else 0

        # Calculate underlying move
        underlying_move_pct = ((current_underlying_price - pos.entry_underlying_price)
                               / pos.entry_underlying_price) * 100

        # Calculate achieved leverage
        leverage = pnl_pct / underlying_move_pct if abs(underlying_move_pct) > 0.01 else 0

        # Create trade record
        trade = OptionTrade(
            contract=pos.contract,
            entry_time=pos.entry_time,
            exit_time=current_time,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=total_pnl,
            pnl_pct=pnl_pct,
            entry_underlying_price=pos.entry_underlying_price,
            exit_underlying_price=current_underlying_price,
            underlying_move_pct=underlying_move_pct,
            leverage_achieved=leverage,
            original_signal_symbol=pos.original_signal_symbol,
            strategy_name=pos.strategy_name,
            exit_reason=exit_reason,
        )

        self.shadow_trades.append(trade)
        self.shadow_pnl_total += total_pnl
        self.shadow_position = None

        # Log the trade
        pnl_sign = "+" if total_pnl >= 0 else ""
        print(f"[OPTIONS] Shadow Position Closed:")
        print(f"  QQQ {pos.contract.option_type.value} ${pos.contract.strike:.0f}")
        print(f"  Entry: ${pos.entry_price:.2f} -> Exit: ${exit_price:.2f}")
        print(f"  P&L: {pnl_sign}${total_pnl:.2f} ({pnl_sign}{pnl_pct:.1f}%)")
        print(f"  Underlying Move: {underlying_move_pct:+.2f}%")
        print(f"  Leverage: {leverage:.1f}x")
        print(f"  Exit Reason: {exit_reason}")

        return trade

    def simulate_option_price(
        self,
        entry_underlying: float,
        current_underlying: float,
        entry_option_price: float,
        delta: float,
        time_held_hours: float,
        option_type: OptionType = OptionType.CALL,
    ) -> float:
        """
        Simulate option price based on underlying move and time decay.

        This uses a simplified model:
        1. Delta effect: price_change = delta * underlying_change
        2. Gamma effect: delta increases as we go ITM (simplified)
        3. Theta effect: lose ~0.5-1% per hour for 1-DTE options

        Args:
            entry_underlying: Underlying price at entry
            current_underlying: Current underlying price
            entry_option_price: Premium paid at entry
            delta: Option delta at entry
            time_held_hours: Time held in hours
            option_type: CALL or PUT

        Returns:
            Estimated current option price
        """
        # Calculate underlying move
        underlying_change = current_underlying - entry_underlying
        underlying_pct_change = underlying_change / entry_underlying

        # For puts, underlying down = option up
        if option_type == OptionType.PUT:
            underlying_change = -underlying_change
            underlying_pct_change = -underlying_pct_change

        # Delta effect (linear approximation)
        delta_pnl = delta * underlying_change

        # Gamma effect (delta increases as we go ITM)
        # Simplified: if move is in our favor, add bonus from gamma
        gamma = self.DEFAULT_GAMMA
        if underlying_change > 0:
            gamma_bonus = 0.5 * gamma * (underlying_change ** 2)
        else:
            gamma_bonus = -0.5 * gamma * (underlying_change ** 2)

        # Theta decay (time decay)
        # For 1-DTE options, decay is ~10-20% per day = ~0.5-1% per hour
        theta_per_hour = 0.008  # 0.8% per hour
        theta_decay = entry_option_price * theta_per_hour * time_held_hours

        # Calculate new price
        new_price = entry_option_price + delta_pnl + gamma_bonus - theta_decay

        # Floor at 0.01 (options can't go negative, but can go to near zero)
        new_price = max(0.01, new_price)

        return new_price

    def simulate_option_pnl_from_equity(
        self,
        equity_pnl_pct: float,
        time_held_hours: float,
        is_bullish: bool = True,
    ) -> float:
        """
        Quick estimation: What would option P&L be given equity P&L?

        Args:
            equity_pnl_pct: P&L percentage from TQQQ/SQQQ trade
            time_held_hours: How long position was held
            is_bullish: True for TQQQ signals, False for SQQQ

        Returns:
            Estimated option P&L percentage
        """
        # TQQQ is 3x leveraged, so underlying move is equity_pnl / 3
        underlying_move_pct = equity_pnl_pct / 3

        # Option leverage approximation
        option_pnl_pct = underlying_move_pct * self.LEVERAGE_MULTIPLIER

        # Subtract theta decay
        theta_decay_pct = 0.8 * time_held_hours  # 0.8% per hour
        option_pnl_pct -= theta_decay_pct

        # Cap at -100% (can't lose more than premium)
        option_pnl_pct = max(-100, option_pnl_pct)

        return option_pnl_pct

    def get_shadow_summary(self) -> Dict:
        """Get summary of shadow options trading."""
        if not self.shadow_trades:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "avg_leverage": 0.0,
            }

        wins = sum(1 for t in self.shadow_trades if t.pnl > 0)
        total_pnl_pct = sum(t.pnl_pct for t in self.shadow_trades)
        avg_leverage = sum(abs(t.leverage_achieved) for t in self.shadow_trades) / len(self.shadow_trades)

        return {
            "total_trades": len(self.shadow_trades),
            "total_pnl": self.shadow_pnl_total,
            "win_rate": wins / len(self.shadow_trades) * 100,
            "avg_pnl_pct": total_pnl_pct / len(self.shadow_trades),
            "avg_leverage": avg_leverage,
            "trades": self.shadow_trades,
        }

    def print_shadow_report(self):
        """Print a formatted report of shadow options trading."""
        summary = self.get_shadow_summary()

        print("\n" + "=" * 70)
        print("  SHADOW OPTIONS TRADING REPORT")
        print("  (What you WOULD have made with options)")
        print("=" * 70)

        if summary["total_trades"] == 0:
            print("  No shadow trades recorded.")
            return

        print(f"\n  Total Trades: {summary['total_trades']}")
        print(f"  Win Rate: {summary['win_rate']:.1f}%")
        print(f"  Total P&L: ${summary['total_pnl']:+,.2f}")
        print(f"  Avg P&L per Trade: {summary['avg_pnl_pct']:+.1f}%")
        print(f"  Avg Leverage Achieved: {summary['avg_leverage']:.1f}x")

        print("\n" + "-" * 70)
        print("  TRADE LOG (Last 10)")
        print("-" * 70)

        for trade in self.shadow_trades[-10:]:
            pnl_str = f"${trade.pnl:+,.2f}" if trade.pnl >= 0 else f"${trade.pnl:,.2f}"
            entry_str = trade.entry_time.strftime("%m/%d %H:%M")
            exit_str = trade.exit_time.strftime("%H:%M")

            print(f"  {entry_str} -> {exit_str} | QQQ {trade.contract.option_type.value} ${trade.contract.strike:.0f}")
            print(f"    Underlying: ${trade.entry_underlying_price:.2f} -> ${trade.exit_underlying_price:.2f} ({trade.underlying_move_pct:+.2f}%)")
            print(f"    Option: ${trade.entry_price:.2f} -> ${trade.exit_price:.2f} | {pnl_str} ({trade.pnl_pct:+.1f}%)")
            print(f"    Leverage: {trade.leverage_achieved:.1f}x | {trade.exit_reason}")

        print("=" * 70)

    # Helper methods

    def _round_to_strike(self, price: float) -> float:
        """Round to nearest valid strike price (QQQ has $1 strikes)."""
        return round(price)

    def _get_next_expiry(self, current_time: datetime, dte: int) -> datetime:
        """Get the next expiration date."""
        # Simple: add DTE days
        expiry = current_time + timedelta(days=dte)
        # Set to market close time (4 PM ET)
        expiry = expiry.replace(hour=16, minute=0, second=0, microsecond=0)
        return expiry

    def _simulate_contract(
        self,
        underlying_price: float,
        strike: float,
        option_type: OptionType,
        expiry: datetime,
        current_time: datetime,
    ) -> OptionContract:
        """
        Create a simulated option contract with estimated Greeks.

        In production, this would fetch from the options chain API.
        For simulation, we use simplified Black-Scholes approximations.
        """
        # Calculate moneyness
        if option_type == OptionType.CALL:
            moneyness = underlying_price / strike
        else:
            moneyness = strike / underlying_price

        # Time to expiry in years
        tte = (expiry - current_time).total_seconds() / (365.25 * 24 * 3600)
        tte = max(tte, 1/365)  # Minimum 1 day

        # Simplified option pricing
        # ATM options: price ≈ 0.4 * underlying * sqrt(tte) * IV
        iv = self.DEFAULT_IV
        base_price = 0.4 * underlying_price * math.sqrt(tte) * iv

        # Adjust for moneyness
        if moneyness > 1:  # ITM
            intrinsic = underlying_price - strike if option_type == OptionType.CALL else strike - underlying_price
            intrinsic = max(0, intrinsic)
            price = base_price + intrinsic * 0.5
        else:  # OTM
            price = base_price * moneyness

        # Add bid-ask spread (~2-3% for liquid options)
        spread = price * 0.025
        bid = price - spread / 2
        ask = price + spread / 2

        # Delta approximation
        if option_type == OptionType.CALL:
            delta = 0.5 + (moneyness - 1) * 2  # Rough approximation
            delta = min(0.95, max(0.05, delta))
        else:
            delta = -0.5 - (moneyness - 1) * 2
            delta = max(-0.95, min(-0.05, delta))

        return OptionContract(
            underlying="QQQ",
            option_type=option_type,
            strike=strike,
            expiry=expiry,
            delta=abs(delta),  # Store absolute delta
            gamma=self.DEFAULT_GAMMA,
            theta=self.DEFAULT_THETA,
            vega=self.DEFAULT_VEGA,
            implied_volatility=iv,
            bid=max(0.01, bid),
            ask=max(0.02, ask),
        )


# Convenience function for quick P&L comparison
def compare_equity_vs_options(
    equity_pnl_dollars: float,
    equity_entry_price: float,
    time_held_hours: float,
    symbol: str = "TQQQ",
) -> Dict:
    """
    Quick comparison: What would options have made?

    Args:
        equity_pnl_dollars: Actual P&L from equity trade
        equity_entry_price: Entry price of equity
        time_held_hours: How long held
        symbol: TQQQ or SQQQ

    Returns:
        Comparison dictionary
    """
    adapter = OptionsAdapter(mode="simulation")

    # Calculate equity P&L percentage
    # Assume 100 shares for comparison
    shares = 100
    equity_pnl_pct = (equity_pnl_dollars / (equity_entry_price * shares)) * 100

    # Estimate options P&L
    options_pnl_pct = adapter.simulate_option_pnl_from_equity(
        equity_pnl_pct=equity_pnl_pct,
        time_held_hours=time_held_hours,
        is_bullish=(symbol == "TQQQ"),
    )

    # Assume $1000 options allocation
    options_allocation = 1000
    options_pnl_dollars = options_allocation * (options_pnl_pct / 100)

    return {
        "equity_pnl_dollars": equity_pnl_dollars,
        "equity_pnl_pct": equity_pnl_pct,
        "options_pnl_pct": options_pnl_pct,
        "options_pnl_dollars": options_pnl_dollars,
        "leverage_factor": options_pnl_pct / equity_pnl_pct if abs(equity_pnl_pct) > 0.01 else 0,
        "theta_cost_pct": 0.8 * time_held_hours,
    }
