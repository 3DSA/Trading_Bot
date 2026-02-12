"""
Base Options Strategy - The Physics Engine for Options

This module provides the foundation for options-native strategies.
Unlike stock strategies that care about direction, options strategies
must respect the Greeks:

    Delta (Δ): Direction sensitivity (like stocks, but leveraged)
    Gamma (Γ): ACCELERATION - how fast Delta changes (our friend in explosions)
    Theta (Θ): TIME DECAY - the ice cube melting (our enemy)
    Vega (ν): Volatility sensitivity (friend in panics, enemy in calm)

Key Principle: In options, TIME IS TOXIC. Every minute you hold,
Theta is eating your premium. We must be surgical: get in, capture
the move, get out BEFORE the ice cube melts.

Author: Bi-Cameral Quant Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np
import pytz


class OptionType(Enum):
    """Option contract type."""
    CALL = "CALL"
    PUT = "PUT"


class OptionSignalType(Enum):
    """Signal types for options strategies."""
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    EXIT = "EXIT"
    HOLD = "HOLD"


class ContractSelection(Enum):
    """How to select the strike price."""
    ATM = "ATM"      # At the money (highest Gamma)
    OTM_1 = "OTM_1"  # 1 strike out of the money
    OTM_2 = "OTM_2"  # 2 strikes out of the money
    ITM_1 = "ITM_1"  # 1 strike in the money


@dataclass
class ContractSpec:
    """
    Specification for an options contract.

    In production, this would be populated from the options chain API.
    For simulation/backtesting, we estimate these values.
    """
    underlying: str           # QQQ
    option_type: OptionType   # CALL or PUT
    strike: float             # Strike price
    expiry: datetime          # Expiration date
    dte: int                  # Days to expiration

    # Greeks (estimated or from chain)
    delta: float = 0.50       # Price sensitivity
    gamma: float = 0.08       # Delta acceleration
    theta: float = -0.15      # Daily time decay (negative)
    vega: float = 0.10        # Vol sensitivity

    # Pricing
    bid: float = 0.0
    ask: float = 0.0
    mid_price: float = 0.0
    implied_volatility: float = 0.25

    # Contract identifier (for live trading)
    symbol: Optional[str] = None  # e.g., "QQQ250214C00520000"

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as percentage of mid price."""
        if self.mid_price > 0:
            return (self.ask - self.bid) / self.mid_price * 100
        return 0.0

    @property
    def is_atm(self) -> bool:
        """Check if contract is approximately ATM."""
        # Within 0.5% of underlying is considered ATM
        return True  # Placeholder - would check vs underlying price


@dataclass
class OptionSignal:
    """
    Signal from an options strategy.

    Unlike stock signals, option signals include:
    - Contract specification (strike, expiry)
    - Time-based exit rules (mandatory)
    - Greeks-based exit rules
    """
    signal: OptionSignalType
    reason: str
    confidence: float = 0.5

    # Contract details
    contract: Optional[ContractSpec] = None

    # Exit rules (MANDATORY for options)
    time_stop_minutes: int = 10          # Max hold time
    profit_target_pct: float = 0.15      # 15% profit target
    stop_loss_pct: float = 0.50          # 50% loss limit

    # Entry price tracking
    entry_price: Optional[float] = None
    entry_time: Optional[datetime] = None

    # Additional context
    metadata: Dict = field(default_factory=dict)


@dataclass
class OptionPosition:
    """Tracks an open options position."""
    contract: ContractSpec
    entry_price: float
    entry_time: datetime
    quantity: int
    entry_underlying_price: float
    strategy_name: str
    signal_reason: str

    # Tracking
    highest_pnl_pct: float = 0.0
    bars_held: int = 0


class BaseOptionStrategy(ABC):
    """
    Base class for options-native strategies.

    Key differences from stock strategies:
    1. TIME STOPS are MANDATORY (Theta decay)
    2. Signals include contract specification
    3. Exit logic respects Greeks dynamics
    4. Position sizing accounts for leverage

    Subclasses must implement:
    - prepare_data(): Add strategy-specific indicators
    - generate_signal(): Generate option signals
    """

    name: str = "base_option"
    description: str = "Base options strategy"
    version: str = "1.0.0"

    # Universal options constraints
    MAX_DTE = 3           # Maximum days to expiration (1-3 DTE sweet spot)
    MIN_DTE = 1           # Avoid 0-DTE initially (too much gamma risk)
    MAX_HOLD_MINUTES = 30 # Absolute max hold time

    # Greeks targets
    TARGET_DELTA = 0.50   # ATM for maximum Gamma
    MIN_DELTA = 0.30      # Don't go too far OTM
    MAX_DELTA = 0.70      # Don't go too far ITM

    # Risk limits
    MAX_POSITION_PCT = 0.05  # Max 5% of capital per trade
    MAX_SPREAD_PCT = 5.0     # Max 5% bid-ask spread

    def __init__(self, underlying: str = "QQQ"):
        """
        Initialize options strategy.

        Args:
            underlying: Underlying symbol (QQQ for Nasdaq options)
        """
        self.underlying = underlying
        self._position: Optional[OptionPosition] = None

    @abstractmethod
    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add strategy-specific indicators.

        Args:
            df: OHLCV DataFrame for the underlying (QQQ)

        Returns:
            DataFrame with added indicators
        """
        pass

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[OptionPosition] = None,
        vix_value: Optional[float] = None,
    ) -> OptionSignal:
        """
        Generate options signal.

        Args:
            df: Prepared DataFrame with indicators
            current_position: Current option position (if any)
            vix_value: Current VIX level

        Returns:
            OptionSignal with contract spec and exit rules
        """
        pass

    def select_contract(
        self,
        underlying_price: float,
        option_type: OptionType,
        selection: ContractSelection = ContractSelection.ATM,
        current_time: Optional[datetime] = None,
        iv: float = 0.25,
    ) -> ContractSpec:
        """
        Select the appropriate options contract.

        This is the "Chain Manager" - in production, it would query
        the actual options chain. For simulation, we estimate.

        Args:
            underlying_price: Current price of underlying (QQQ)
            option_type: CALL or PUT
            selection: ATM, OTM_1, OTM_2, ITM_1
            current_time: Current timestamp
            iv: Implied volatility estimate

        Returns:
            ContractSpec with estimated Greeks
        """
        current_time = current_time or datetime.now(pytz.utc)

        # Calculate strike based on selection
        # QQQ has $1 strike intervals
        if selection == ContractSelection.ATM:
            strike = round(underlying_price)
        elif selection == ContractSelection.OTM_1:
            if option_type == OptionType.CALL:
                strike = round(underlying_price) + 1
            else:
                strike = round(underlying_price) - 1
        elif selection == ContractSelection.OTM_2:
            if option_type == OptionType.CALL:
                strike = round(underlying_price) + 2
            else:
                strike = round(underlying_price) - 2
        elif selection == ContractSelection.ITM_1:
            if option_type == OptionType.CALL:
                strike = round(underlying_price) - 1
            else:
                strike = round(underlying_price) + 1
        else:
            strike = round(underlying_price)

        # Calculate expiry (prefer 1-DTE)
        expiry = self._get_next_expiry(current_time, self.MIN_DTE)
        dte = max(1, (expiry.date() - current_time.date()).days)

        # Estimate Greeks based on moneyness and DTE
        delta, gamma, theta, vega = self._estimate_greeks(
            underlying_price, strike, option_type, dte, iv
        )

        # Estimate option price
        mid_price = self._estimate_option_price(
            underlying_price, strike, option_type, dte, iv
        )
        spread = mid_price * 0.025  # ~2.5% spread for liquid options

        # Build contract symbol (OCC format)
        # Example: QQQ250214C00520000
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
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            bid=max(0.01, mid_price - spread/2),
            ask=mid_price + spread/2,
            mid_price=mid_price,
            implied_volatility=iv,
            symbol=symbol,
        )

    def check_time_stop(
        self,
        position: OptionPosition,
        current_time: datetime,
        max_minutes: int,
    ) -> bool:
        """
        Check if time stop is triggered.

        TIME STOPS ARE NON-NEGOTIABLE IN OPTIONS.
        Every minute held, Theta is eating your premium.

        Args:
            position: Current position
            current_time: Current timestamp
            max_minutes: Maximum hold time in minutes

        Returns:
            True if time stop triggered
        """
        held_seconds = (current_time - position.entry_time).total_seconds()
        held_minutes = held_seconds / 60
        return held_minutes >= max_minutes

    def estimate_current_pnl(
        self,
        position: OptionPosition,
        current_underlying_price: float,
        current_time: datetime,
    ) -> Tuple[float, float]:
        """
        Estimate current P&L for an options position.

        Uses simplified Greeks model:
        - Delta effect from underlying move
        - Gamma effect (acceleration)
        - Theta decay from time held

        Args:
            position: Current position
            current_underlying_price: Current underlying price
            current_time: Current timestamp

        Returns:
            Tuple of (pnl_dollars, pnl_pct)
        """
        contract = position.contract
        entry_price = position.entry_price

        # Calculate underlying move
        underlying_move = current_underlying_price - position.entry_underlying_price

        # For puts, underlying down = option up
        if contract.option_type == OptionType.PUT:
            underlying_move = -underlying_move

        # Delta effect
        delta_pnl = contract.delta * underlying_move

        # Gamma effect (delta increases as we go ITM)
        if underlying_move > 0:
            gamma_bonus = 0.5 * contract.gamma * (underlying_move ** 2)
        else:
            gamma_bonus = -0.5 * contract.gamma * (underlying_move ** 2)

        # Theta decay
        hours_held = (current_time - position.entry_time).total_seconds() / 3600
        # Theta accelerates in final day - use 0.8% per hour for 1-DTE
        theta_decay = entry_price * 0.008 * hours_held

        # Calculate new price
        new_price = entry_price + delta_pnl + gamma_bonus - theta_decay
        new_price = max(0.01, new_price)  # Floor at $0.01

        # P&L
        pnl_per_contract = (new_price - entry_price) * 100  # 100 shares per contract
        total_pnl = pnl_per_contract * position.quantity
        pnl_pct = ((new_price - entry_price) / entry_price) * 100

        return total_pnl, pnl_pct

    def _get_next_expiry(self, current_time: datetime, dte: int) -> datetime:
        """Get the next expiration date."""
        expiry = current_time + timedelta(days=dte)
        # Set to market close (4 PM ET)
        expiry = expiry.replace(hour=16, minute=0, second=0, microsecond=0)

        # Skip weekends
        while expiry.weekday() >= 5:
            expiry += timedelta(days=1)

        return expiry

    def _estimate_greeks(
        self,
        underlying_price: float,
        strike: float,
        option_type: OptionType,
        dte: int,
        iv: float,
    ) -> Tuple[float, float, float, float]:
        """
        Estimate Greeks using simplified model.

        In production, these would come from the options chain.
        """
        # Moneyness
        if option_type == OptionType.CALL:
            moneyness = underlying_price / strike
        else:
            moneyness = strike / underlying_price

        # Delta approximation (logistic function around ATM)
        if option_type == OptionType.CALL:
            base_delta = 0.5 + (moneyness - 1) * 2
            delta = min(0.95, max(0.05, base_delta))
        else:
            base_delta = -0.5 - (moneyness - 1) * 2
            delta = max(-0.95, min(-0.05, base_delta))

        # Gamma - highest at ATM, increases as DTE decreases
        # Gamma = f(moneyness) * f(DTE)
        atm_factor = 1 - abs(moneyness - 1) * 5  # Peaks at ATM
        atm_factor = max(0.1, atm_factor)
        dte_factor = 1 / max(1, dte)  # Higher gamma for shorter DTE
        gamma = 0.05 * atm_factor * dte_factor

        # Theta - more negative for shorter DTE, highest at ATM
        theta_base = -0.05 * atm_factor  # Base decay
        theta = theta_base / max(1, dte)  # Accelerates near expiry

        # Vega - higher for longer DTE and higher IV
        vega = 0.10 * max(1, dte) * (iv / 0.25)

        return abs(delta), gamma, theta, vega

    def _estimate_option_price(
        self,
        underlying_price: float,
        strike: float,
        option_type: OptionType,
        dte: int,
        iv: float,
    ) -> float:
        """
        Estimate option price using simplified model.

        In production, use actual bid/ask from chain.
        """
        import math

        # Time to expiry in years
        tte = max(dte, 1) / 365.0

        # Simplified ATM approximation: price ≈ 0.4 * S * sqrt(T) * IV
        base_price = 0.4 * underlying_price * math.sqrt(tte) * iv

        # Adjust for moneyness
        if option_type == OptionType.CALL:
            intrinsic = max(0, underlying_price - strike)
            if underlying_price > strike:  # ITM
                price = base_price + intrinsic * 0.8
            else:  # OTM
                otm_factor = underlying_price / strike
                price = base_price * otm_factor
        else:
            intrinsic = max(0, strike - underlying_price)
            if strike > underlying_price:  # ITM
                price = base_price + intrinsic * 0.8
            else:  # OTM
                otm_factor = strike / underlying_price
                price = base_price * otm_factor

        return max(0.05, price)  # Minimum $0.05

    def is_trading_time(self, timestamp: datetime) -> bool:
        """
        Check if within options trading hours.

        Options have specific trading hours and we want to avoid:
        - First 5 minutes (erratic pricing)
        - Last 15 minutes (gamma risk)
        """
        # Convert to Eastern Time
        eastern = pytz.timezone('America/New_York')
        if timestamp.tzinfo is None:
            # Assume UTC if no timezone
            timestamp = pytz.utc.localize(timestamp)
        et_time = timestamp.astimezone(eastern)

        hour = et_time.hour
        minute = et_time.minute

        # Market hours: 9:30 AM - 4:00 PM ET
        if hour < 9 or (hour == 9 and minute < 35):
            return False  # Before 9:35 (skip first 5 min)

        if hour >= 16:
            return False  # After market close

        if hour == 15 and minute >= 45:
            return False  # Last 15 minutes (gamma risk)

        return True

    def calculate_position_size(
        self,
        capital: float,
        option_price: float,
        max_risk_pct: float = 0.05,
    ) -> int:
        """
        Calculate number of contracts to buy.

        Args:
            capital: Available capital
            option_price: Premium per share (multiply by 100 for contract)
            max_risk_pct: Maximum capital to risk (default 5%)

        Returns:
            Number of contracts (minimum 1)
        """
        max_allocation = capital * max_risk_pct
        contract_cost = option_price * 100  # 100 shares per contract

        if contract_cost <= 0:
            return 0

        contracts = int(max_allocation / contract_cost)
        return max(1, contracts)  # Minimum 1 contract
