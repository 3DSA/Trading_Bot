"""
Core components for the Options Strategy Library.

Contains base classes, type definitions, and shared utilities.
"""

from strategies.options.core.base_options import (
    BaseOptionStrategy,
    OptionSignal,
    OptionSignalType,
    ContractSpec,
    OptionType,
    ContractSelection,
    OptionPosition,
)

__all__ = [
    "BaseOptionStrategy",
    "OptionSignal",
    "OptionSignalType",
    "ContractSpec",
    "OptionType",
    "ContractSelection",
    "OptionPosition",
]
