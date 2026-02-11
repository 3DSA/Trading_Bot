"""
Strategy Factory - Dynamic Strategy Loading

Provides a clean interface for loading and managing trading strategies.
Allows the AI Manager to hot-swap "cartridges" based on market regime.

Usage:
    from strategies import get_strategy, list_strategies

    # Get a specific strategy
    sniper = get_strategy("momentum_scalper")
    signal = sniper.generate_signal(df)

    # List available strategies
    strategies = list_strategies()

    # Get strategy for market regime
    strategy = get_strategy_for_regime("TREND")

Author: Bi-Cameral Quant Team
"""

from typing import Dict, List, Optional, Type
import logging

from deprecated.strategies.base import BaseStrategy, StrategyConfig

# Import all strategy classes
from deprecated.strategies.momentum_scalper import MomentumScalperStrategy
from deprecated.strategies.mean_reversion import MeanReversionStrategy
from deprecated.strategies.volatility_breakout import VolatilityBreakoutStrategy
from deprecated.strategies.crisis_alpha import CrisisAlphaStrategy

logger = logging.getLogger(__name__)


# Registry of available strategies
STRATEGY_REGISTRY: Dict[str, Type[BaseStrategy]] = {
    "momentum_scalper": MomentumScalperStrategy,
    "mean_reversion": MeanReversionStrategy,
    "volatility_breakout": VolatilityBreakoutStrategy,
    "crisis_alpha": CrisisAlphaStrategy,
    # Aliases for convenience
    "sniper": MomentumScalperStrategy,
    "rubber_band": MeanReversionStrategy,
    "news_trader": VolatilityBreakoutStrategy,
    "orb": VolatilityBreakoutStrategy,
    "bear": CrisisAlphaStrategy,
    "vix": CrisisAlphaStrategy,
}

# Mapping of market regimes to recommended strategies
REGIME_STRATEGY_MAP: Dict[str, str] = {
    "TREND": "momentum_scalper",
    "CHOP": "mean_reversion",
    "VOLATILE": "volatility_breakout",
    "CRISIS": "crisis_alpha",
    "BUFFER": "momentum_scalper",  # Default to momentum in buffer zone
}


def get_strategy(
    name: str,
    config: Optional[StrategyConfig] = None
) -> BaseStrategy:
    """
    Get a strategy instance by name.

    Args:
        name: Strategy name (e.g., "momentum_scalper", "sniper")
        config: Optional configuration override

    Returns:
        Initialized strategy instance

    Raises:
        ValueError: If strategy name is not found
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")

    if name_lower not in STRATEGY_REGISTRY:
        available = list_strategies()
        raise ValueError(
            f"Unknown strategy: '{name}'. Available: {available}"
        )

    strategy_class = STRATEGY_REGISTRY[name_lower]
    strategy = strategy_class(config=config)

    logger.info(f"Loaded strategy: {strategy.name} v{strategy.version}")

    return strategy


def get_strategy_for_regime(
    regime: str,
    config: Optional[StrategyConfig] = None
) -> BaseStrategy:
    """
    Get the recommended strategy for a market regime.

    Args:
        regime: Market regime (TREND, CHOP, VOLATILE, CRISIS)
        config: Optional configuration override

    Returns:
        Strategy optimized for the given regime
    """
    regime_upper = regime.upper()

    if regime_upper not in REGIME_STRATEGY_MAP:
        logger.warning(f"Unknown regime '{regime}', defaulting to momentum_scalper")
        regime_upper = "TREND"

    strategy_name = REGIME_STRATEGY_MAP[regime_upper]
    strategy = get_strategy(strategy_name, config)

    logger.info(f"Selected {strategy.name} for {regime_upper} regime")

    return strategy


def list_strategies() -> List[str]:
    """
    List all available strategy names.

    Returns:
        List of strategy names (excluding aliases)
    """
    # Filter out aliases (only return primary names)
    primary_names = [
        "momentum_scalper",
        "mean_reversion",
        "volatility_breakout",
        "crisis_alpha",
    ]
    return primary_names


def get_strategy_info(name: str) -> dict:
    """
    Get detailed information about a strategy.

    Args:
        name: Strategy name

    Returns:
        Dictionary with strategy metadata
    """
    strategy = get_strategy(name)

    return {
        "name": strategy.name,
        "description": strategy.description,
        "version": strategy.version,
        "preferred_regime": strategy.preferred_regime,
        "class": strategy.__class__.__name__,
    }


def get_all_strategy_info() -> List[dict]:
    """
    Get information about all available strategies.

    Returns:
        List of strategy info dictionaries
    """
    return [get_strategy_info(name) for name in list_strategies()]


def validate_strategy(name: str) -> bool:
    """
    Check if a strategy name is valid.

    Args:
        name: Strategy name to validate

    Returns:
        True if valid, False otherwise
    """
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    return name_lower in STRATEGY_REGISTRY


class StrategyManager:
    """
    Manages multiple strategies and regime-based switching.

    Used by the AI Manager to dynamically select strategies
    based on detected market conditions.
    """

    def __init__(self, default_config: Optional[StrategyConfig] = None):
        """
        Initialize the strategy manager.

        Args:
            default_config: Default configuration for all strategies
        """
        self.default_config = default_config or StrategyConfig()
        self._cached_strategies: Dict[str, BaseStrategy] = {}
        self._current_strategy: Optional[BaseStrategy] = None
        self._current_regime: str = "TREND"

    def get_strategy(self, name: str) -> BaseStrategy:
        """Get a strategy, using cache if available."""
        if name not in self._cached_strategies:
            self._cached_strategies[name] = get_strategy(name, self.default_config)
        return self._cached_strategies[name]

    def set_regime(self, regime: str) -> BaseStrategy:
        """
        Switch to the optimal strategy for a regime.

        Args:
            regime: Market regime (TREND, CHOP, VOLATILE, CRISIS)

        Returns:
            The newly selected strategy
        """
        if regime != self._current_regime:
            logger.info(f"Regime change: {self._current_regime} -> {regime}")

        self._current_regime = regime
        strategy_name = REGIME_STRATEGY_MAP.get(regime.upper(), "momentum_scalper")
        self._current_strategy = self.get_strategy(strategy_name)

        return self._current_strategy

    @property
    def current_strategy(self) -> Optional[BaseStrategy]:
        """Get the currently active strategy."""
        return self._current_strategy

    @property
    def current_regime(self) -> str:
        """Get the current market regime."""
        return self._current_regime

    def get_signal(self, df, position=None, underlying_df=None):
        """
        Generate signal using the current strategy.

        Args:
            df: DataFrame with OHLCV data
            position: Current position (None if flat)
            underlying_df: Underlying asset data (for Double-Lock)

        Returns:
            StrategySignal from current strategy
        """
        if self._current_strategy is None:
            self.set_regime("TREND")

        # Prepare data with strategy-specific indicators
        prepared_df = self._current_strategy.prepare_data(df)

        return self._current_strategy.generate_signal(
            prepared_df,
            current_position=position,
            underlying_df=underlying_df
        )

    def __repr__(self) -> str:
        strategy_name = self._current_strategy.name if self._current_strategy else "None"
        return f"StrategyManager(regime={self._current_regime}, strategy={strategy_name})"


# Convenience function to create a pre-configured manager
def create_strategy_manager(config: Optional[StrategyConfig] = None) -> StrategyManager:
    """
    Create a new strategy manager with optional config.

    Args:
        config: Strategy configuration

    Returns:
        Initialized StrategyManager
    """
    return StrategyManager(default_config=config)
