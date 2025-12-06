"""
Bi-Cameral Trading Bot - Strategy Library

A modular, professional-grade quantitative strategy library using the Strategy Pattern.
Each strategy is a "cartridge" that can be hot-swapped by the AI Manager based on market regime.

Strategies:
    - MomentumScalper: EMA + VWAP trend following with Double-Lock ADX filter
    - MeanReversion: Bollinger + Z-Score rubber band in chop markets
    - VolatilityBreakout: Opening Range Breakout for morning momentum
    - CrisisAlpha: VIX-weighted trend following for SQQQ

Usage:
    from strategies import get_strategy

    strategy = get_strategy("momentum_scalper")
    signal = strategy.generate_signal(df, config)
"""

from strategies.factory import get_strategy, list_strategies
from strategies.base import BaseStrategy, StrategySignal

__all__ = [
    "get_strategy",
    "list_strategies",
    "BaseStrategy",
    "StrategySignal",
]

__version__ = "1.0.0"
