"""
Options Strategy Factory - Dynamic Loading for Options-Native Strategies

Similar to the stock strategy factory, but for options strategies
that respect Theta, Gamma, and Vega physics.

Usage:
    from strategies.options import get_option_strategy, list_option_strategies

    # Get Gamma Scalper for explosive moves
    gamma = get_option_strategy("gamma_scalper")
    signal = gamma.generate_signal(df, vix_value=25)

    # Get Vega Snap for panic reversals
    vega = get_option_strategy("vega_snap")
    signal = vega.generate_signal(df, vix_value=28)

Author: Bi-Cameral Quant Team
"""

from typing import Dict, List, Optional, Type
import logging

from strategies.options.base_options import BaseOptionStrategy
from strategies.options.gamma_scalper import GammaScalperStrategy
from strategies.options.vega_snap import VegaSnapStrategy

logger = logging.getLogger(__name__)


# Registry of available options strategies
OPTION_STRATEGY_REGISTRY: Dict[str, Type[BaseOptionStrategy]] = {
    "gamma_scalper": GammaScalperStrategy,
    "vega_snap": VegaSnapStrategy,
    # Aliases
    "gamma": GammaScalperStrategy,
    "explosion": GammaScalperStrategy,
    "snap": VegaSnapStrategy,
    "panic": VegaSnapStrategy,
}

# Mapping of market conditions to strategies
# Unlike stocks where ADX determines regime, options care about:
# - Volatility (VIX) for strategy selection
# - Speed of moves for timing
CONDITION_STRATEGY_MAP: Dict[str, str] = {
    "EXPLOSIVE": "gamma_scalper",    # Fast moves, any direction
    "PANIC": "vega_snap",            # Crash + VIX spike
    "NORMAL": "gamma_scalper",       # Default to gamma
}


def get_option_strategy(
    name: str,
    underlying: str = "QQQ",
) -> BaseOptionStrategy:
    """
    Get an options strategy instance by name.

    Args:
        name: Strategy name (e.g., "gamma_scalper", "vega_snap")
        underlying: Underlying symbol (default QQQ)

    Returns:
        Initialized options strategy instance

    Raises:
        ValueError: If strategy name is not found
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower not in OPTION_STRATEGY_REGISTRY:
        available = list_option_strategies()
        raise ValueError(
            f"Unknown options strategy: '{name}'. Available: {available}"
        )

    strategy_class = OPTION_STRATEGY_REGISTRY[name_lower]
    strategy = strategy_class(underlying=underlying)

    logger.info(f"Loaded options strategy: {strategy.name} v{strategy.version}")

    return strategy


def get_option_strategy_for_condition(
    condition: str,
    underlying: str = "QQQ",
) -> BaseOptionStrategy:
    """
    Get the recommended options strategy for a market condition.

    Args:
        condition: Market condition (EXPLOSIVE, PANIC, NORMAL)
        underlying: Underlying symbol

    Returns:
        Strategy optimized for the given condition
    """
    condition_upper = condition.upper()

    if condition_upper not in CONDITION_STRATEGY_MAP:
        logger.warning(f"Unknown condition '{condition}', defaulting to gamma_scalper")
        condition_upper = "NORMAL"

    strategy_name = CONDITION_STRATEGY_MAP[condition_upper]
    strategy = get_option_strategy(strategy_name, underlying)

    logger.info(f"Selected {strategy.name} for {condition_upper} condition")

    return strategy


def list_option_strategies() -> List[str]:
    """
    List all available options strategy names.

    Returns:
        List of strategy names (excluding aliases)
    """
    return ["gamma_scalper", "vega_snap"]


def get_option_strategy_info(name: str) -> dict:
    """
    Get detailed information about an options strategy.

    Args:
        name: Strategy name

    Returns:
        Dictionary with strategy metadata
    """
    strategy = get_option_strategy(name)

    return {
        "name": strategy.name,
        "description": strategy.description,
        "version": strategy.version,
        "underlying": strategy.underlying,
        "max_hold_minutes": strategy.MAX_HOLD_MINUTES,
        "target_delta": strategy.TARGET_DELTA,
        "class": strategy.__class__.__name__,
    }


def select_option_strategy(
    vix_value: float,
    price_velocity: float,
    zscore: float,
) -> str:
    """
    Intelligent strategy selection based on market conditions.

    This is the "brain" that decides which options strategy to use
    based on current market state.

    Args:
        vix_value: Current VIX level
        price_velocity: Recent price velocity (1-min move %)
        zscore: Current Z-Score of price

    Returns:
        Name of recommended strategy
    """
    # PANIC CONDITIONS: VIX high + extreme price drop
    if vix_value >= 22 and zscore < -2.5:
        logger.info(f"PANIC detected: VIX={vix_value}, Z={zscore} -> vega_snap")
        return "vega_snap"

    # EXPLOSIVE CONDITIONS: Fast move with volume
    if abs(price_velocity) >= 0.003:  # 0.3% in 1 minute
        logger.info(f"EXPLOSION detected: velocity={price_velocity*100:.2f}% -> gamma_scalper")
        return "gamma_scalper"

    # DEFAULT: Gamma scalper (wait for explosions)
    return "gamma_scalper"


class OptionStrategyManager:
    """
    Manages options strategies with intelligent switching.

    Unlike stock strategies that switch based on ADX/regime,
    options strategies switch based on:
    - VIX levels (volatility)
    - Price velocity (explosions)
    - Statistical extremes (panic)
    """

    def __init__(self, underlying: str = "QQQ"):
        """
        Initialize the options strategy manager.

        Args:
            underlying: Underlying symbol for all strategies
        """
        self.underlying = underlying
        self._cached_strategies: Dict[str, BaseOptionStrategy] = {}
        self._current_strategy: Optional[BaseOptionStrategy] = None
        self._current_condition: str = "NORMAL"

    def get_strategy(self, name: str) -> BaseOptionStrategy:
        """Get a strategy, using cache if available."""
        if name not in self._cached_strategies:
            self._cached_strategies[name] = get_option_strategy(name, self.underlying)
        return self._cached_strategies[name]

    def select_strategy(
        self,
        vix_value: float,
        price_velocity: float,
        zscore: float,
    ) -> BaseOptionStrategy:
        """
        Select the optimal strategy based on conditions.

        Args:
            vix_value: Current VIX level
            price_velocity: Recent price velocity
            zscore: Current Z-Score

        Returns:
            The selected strategy
        """
        strategy_name = select_option_strategy(vix_value, price_velocity, zscore)

        # Determine condition for logging
        if strategy_name == "vega_snap":
            new_condition = "PANIC"
        elif abs(price_velocity) >= 0.003:
            new_condition = "EXPLOSIVE"
        else:
            new_condition = "NORMAL"

        if new_condition != self._current_condition:
            logger.info(f"Condition change: {self._current_condition} -> {new_condition}")
            self._current_condition = new_condition

        self._current_strategy = self.get_strategy(strategy_name)
        return self._current_strategy

    @property
    def current_strategy(self) -> Optional[BaseOptionStrategy]:
        """Get the currently active strategy."""
        return self._current_strategy

    @property
    def current_condition(self) -> str:
        """Get the current market condition."""
        return self._current_condition

    def __repr__(self) -> str:
        strategy_name = self._current_strategy.name if self._current_strategy else "None"
        return f"OptionStrategyManager(condition={self._current_condition}, strategy={strategy_name})"


# Convenience function
def create_option_strategy_manager(underlying: str = "QQQ") -> OptionStrategyManager:
    """
    Create a new options strategy manager.

    Args:
        underlying: Underlying symbol

    Returns:
        Initialized OptionStrategyManager
    """
    return OptionStrategyManager(underlying=underlying)
