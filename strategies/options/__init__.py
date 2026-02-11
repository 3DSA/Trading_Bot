"""
Options-Native Strategy Library

These strategies are designed specifically for options physics:
- Time is TOXIC (Theta decay)
- Convexity is POWER (Gamma acceleration)
- Volatility is TREACHEROUS (Vega crush/expansion)

DO NOT use stock strategies to trade options. Different physics.

Strategies:
    - GammaScalper: Captures explosive moves (replaces momentum)
    - VegaSnap: Captures panic reversals (replaces mean reversion)

Usage:
    from strategies.options import get_option_strategy, GammaScalperStrategy

    # Get strategy by name
    gamma = get_option_strategy("gamma_scalper")

    # Or instantiate directly
    gamma = GammaScalperStrategy(underlying="QQQ")

    # Generate signal
    signal = gamma.generate_signal(prepared_df, vix_value=25)

Author: Bi-Cameral Quant Team
"""

from strategies.options.base_options import (
    BaseOptionStrategy,
    OptionSignal,
    OptionSignalType,
    ContractSpec,
    OptionType,
    ContractSelection,
    OptionPosition,
)
from strategies.options.gamma_scalper import GammaScalperStrategy
from strategies.options.vega_snap import VegaSnapStrategy
from strategies.options.factory import (
    get_option_strategy,
    get_option_strategy_for_condition,
    list_option_strategies,
    get_option_strategy_info,
    select_option_strategy,
    OptionStrategyManager,
    create_option_strategy_manager,
)

__all__ = [
    # Base classes
    "BaseOptionStrategy",
    "OptionSignal",
    "OptionSignalType",
    "ContractSpec",
    "OptionType",
    "ContractSelection",
    "OptionPosition",
    # Strategies
    "GammaScalperStrategy",
    "VegaSnapStrategy",
    # Factory functions
    "get_option_strategy",
    "get_option_strategy_for_condition",
    "list_option_strategies",
    "get_option_strategy_info",
    "select_option_strategy",
    "OptionStrategyManager",
    "create_option_strategy_manager",
]
