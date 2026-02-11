"""
Execution Module - Order Routing and Adapters

Contains adapters for different execution modes:
- options_adapter: Translates equity signals to options contracts
- futures_adapter: (Future) Translates to MNQ futures
"""

from execution.options_adapter import OptionsAdapter, OptionContract, OptionType

__all__ = ["OptionsAdapter", "OptionContract", "OptionType"]
