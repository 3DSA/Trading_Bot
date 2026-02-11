"""
Options-Native Strategy Library

The trading bot now focuses EXCLUSIVELY on options trading.
Legacy equity strategies have been moved to deprecated/strategies/.

Options Strategies:
    - GammaScalper: Captures explosive moves with ATM calls (high Gamma)
    - VegaSnap: Captures panic reversals at market bottoms
    - DeltaSurfer: Rides steady trends with deep ITM options

Usage:
    from strategies.options import get_option_strategy, GammaScalperStrategy

    # Get strategy by name
    gamma = get_option_strategy("gamma_scalper")
    signal = gamma.generate_signal(df, vix_value=25)

For legacy equity strategies, see deprecated/strategies/
"""

# Only export options strategies (active)
from strategies.options import (
    # Base classes
    BaseOptionStrategy,
    OptionSignal,
    OptionSignalType,
    ContractSpec,
    OptionType,
    ContractSelection,
    OptionPosition,
    # Strategies
    GammaScalperStrategy,
    VegaSnapStrategy,
    DeltaSurferStrategy,
    # Factory functions
    get_option_strategy,
    get_option_strategy_for_condition,
    list_option_strategies,
    get_option_strategy_info,
    select_option_strategy,
    OptionStrategyManager,
    create_option_strategy_manager,
)

# Shared utilities (used by both options and legacy)
from strategies.shared_utils import calc_adx, calc_atr, calc_ema, calc_rsi

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
    "DeltaSurferStrategy",
    # Factory functions
    "get_option_strategy",
    "get_option_strategy_for_condition",
    "list_option_strategies",
    "get_option_strategy_info",
    "select_option_strategy",
    "OptionStrategyManager",
    "create_option_strategy_manager",
    # Utilities
    "calc_adx",
    "calc_atr",
    "calc_ema",
    "calc_rsi",
]

__version__ = "2.0.0"  # Options-native version
