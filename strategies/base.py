"""
Base Strategy - Abstract Template for All Trading Strategies

All strategy "cartridges" must inherit from BaseStrategy and implement
the generate_signal() method. This ensures uniform execution by reflex.py.

The Strategy Pattern allows the AI Manager to hot-swap strategies
based on market regime without modifying the execution engine.

Author: Bi-Cameral Quant Team
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import pandas as pd


class SignalType(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"


class PositionSizing(Enum):
    """Position sizing modes."""
    FULL = 1.0      # 100% of allocated capital
    HALF = 0.5      # 50% of allocated capital
    QUARTER = 0.25  # 25% of allocated capital
    NONE = 0.0      # No position


@dataclass
class StrategySignal:
    """
    Standardized signal output from all strategies.

    Attributes:
        signal: The trading action (BUY, SELL, HOLD, EXIT)
        reason: Human-readable explanation of why
        confidence: Signal strength 0.0 to 1.0
        stop_loss: Suggested stop loss price
        take_profit: Suggested take profit price
        position_size: Position sizing multiplier
        metadata: Additional strategy-specific data
    """
    signal: SignalType
    reason: str
    confidence: float = 0.5
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    position_size: PositionSizing = PositionSizing.FULL
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert signal to dictionary for JSON serialization."""
        return {
            "signal": self.signal.value,
            "reason": self.reason,
            "confidence": self.confidence,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "trailing_stop_pct": self.trailing_stop_pct,
            "position_size": self.position_size.value,
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        sl_str = f"SL=${self.stop_loss:.2f}" if self.stop_loss else "SL=N/A"
        tp_str = f"TP=${self.take_profit:.2f}" if self.take_profit else "TP=N/A"
        return f"[{self.signal.value}] {self.reason} | {sl_str} {tp_str} | Conf={self.confidence:.0%}"


@dataclass
class StrategyConfig:
    """
    Configuration passed to strategies.

    Loaded from config.json and can be overridden per-symbol.
    """
    # Risk Management
    max_position_pct: float = 0.10      # 10% of portfolio per trade
    default_stop_loss_pct: float = 0.01  # 1% default stop
    default_take_profit_pct: float = 0.02  # 2% default TP

    # ADX Settings
    adx_trend_threshold: float = 25.0
    adx_chop_threshold: float = 20.0

    # Double-Lock (for leveraged ETFs)
    use_double_lock: bool = True
    double_lock_leveraged_adx: float = 30.0
    double_lock_underlying_adx: float = 25.0

    # Time Filters
    trading_start: str = "09:35"  # 5 min after open
    trading_end: str = "15:55"    # 5 min before close
    avoid_lunch_start: str = "11:45"
    avoid_lunch_end: str = "13:15"

    # Symbol Info
    symbol: str = "TQQQ"
    underlying_symbol: str = "QQQ"
    is_leveraged: bool = True
    leverage_factor: float = 3.0


class BaseStrategy(ABC):
    """
    Abstract Base Class for all trading strategies.

    Subclasses must implement:
        - generate_signal(): Core strategy logic
        - name: Strategy identifier

    Optional overrides:
        - prepare_data(): Add custom indicators to DataFrame
        - validate_config(): Check strategy-specific config
    """

    # Class attributes (override in subclasses)
    name: str = "base"
    description: str = "Abstract base strategy"
    version: str = "1.0.0"

    # Market regime this strategy works best in
    preferred_regime: str = "ANY"  # TREND, CHOP, VOLATILE, ANY

    def __init__(self, config: Optional[StrategyConfig] = None):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self.config = config or StrategyConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """
        Validate strategy configuration.
        Override in subclasses for strategy-specific validation.
        """
        if self.config.max_position_pct <= 0 or self.config.max_position_pct > 1:
            raise ValueError("max_position_pct must be between 0 and 1")

    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_position: Optional[dict] = None,
        underlying_df: Optional[pd.DataFrame] = None
    ) -> StrategySignal:
        """
        Generate trading signal based on market data.

        This is the core strategy logic that must be implemented by all strategies.

        Args:
            df: DataFrame with OHLCV data and indicators
            current_position: Current position info (None if flat)
                - entry_price: float
                - entry_time: datetime
                - quantity: int
                - side: 'LONG' or 'SHORT'
            underlying_df: DataFrame for underlying asset (e.g., QQQ for TQQQ)

        Returns:
            StrategySignal with trading action and risk parameters
        """
        pass

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add strategy-specific indicators to DataFrame.

        Override this method to add custom indicators needed by your strategy.
        Called by the execution engine before generate_signal().

        Args:
            df: Raw OHLCV DataFrame

        Returns:
            DataFrame with added indicator columns
        """
        return df

    def get_dynamic_stop_loss(
        self,
        entry_price: float,
        atr: float,
        atr_multiplier: float = 1.5
    ) -> float:
        """
        Calculate dynamic stop loss based on ATR.

        ATR-based stops adapt to current volatility:
        - High volatility = wider stop (avoid noise)
        - Low volatility = tighter stop (capture mean reversion)

        Args:
            entry_price: Trade entry price
            atr: Current ATR value
            atr_multiplier: How many ATRs below entry (default 1.5)

        Returns:
            Stop loss price
        """
        return entry_price - (atr * atr_multiplier)

    def get_dynamic_take_profit(
        self,
        entry_price: float,
        atr: float,
        risk_reward_ratio: float = 2.0,
        atr_multiplier: float = 1.5
    ) -> float:
        """
        Calculate take profit based on risk/reward ratio.

        Example: If stop is 1.5 ATR away, TP is 3.0 ATR away (2:1 R/R)

        Args:
            entry_price: Trade entry price
            atr: Current ATR value
            risk_reward_ratio: Target R/R ratio
            atr_multiplier: ATR multiplier used for stop

        Returns:
            Take profit price
        """
        risk = atr * atr_multiplier
        reward = risk * risk_reward_ratio
        return entry_price + reward

    def is_trading_time(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if current time is within trading hours.

        Filters out:
        - Pre-market
        - Lunch hour (11:45-13:15)
        - End of day (after 15:55)

        Args:
            timestamp: Current bar timestamp

        Returns:
            True if trading is allowed
        """
        time_str = timestamp.strftime("%H:%M")

        # Before trading start
        if time_str < self.config.trading_start:
            return False

        # Lunch hour
        if self.config.avoid_lunch_start <= time_str <= self.config.avoid_lunch_end:
            return False

        # End of day
        if time_str >= self.config.trading_end:
            return False

        return True

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float
    ) -> int:
        """
        Calculate position size based on risk.

        Uses fixed fractional position sizing:
        Position Size = (Portfolio * Risk%) / (Entry - Stop)

        Args:
            portfolio_value: Total portfolio value
            entry_price: Planned entry price
            stop_loss: Planned stop loss price

        Returns:
            Number of shares to buy
        """
        # Risk amount in dollars
        risk_amount = portfolio_value * self.config.max_position_pct

        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)

        if risk_per_share <= 0:
            return 0

        shares = int(risk_amount / risk_per_share)

        # Ensure we don't exceed max position size
        max_shares = int((portfolio_value * self.config.max_position_pct) / entry_price)

        return min(shares, max_shares)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
