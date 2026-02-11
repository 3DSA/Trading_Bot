"""
DEPRECATED: Legacy Equity Strategies

These strategies were designed for trading TQQQ/SQQQ equities.
The bot has transitioned to OPTIONS trading.

Strategies (DEPRECATED):
    - MomentumScalper: EMA + VWAP trend following
    - MeanReversion: Bollinger + Z-Score for choppy markets
    - VolatilityBreakout: Opening Range Breakout
    - CrisisAlpha: VIX-weighted SQQQ trading

For active options strategies, use:
    from strategies.options import get_option_strategy

Note: These deprecated files are preserved for reference only.
To use them, you would need to adjust PYTHONPATH or run from project root.
"""

# Imports disabled - files preserved for reference only
# To use legacy strategies, import directly:
#   from deprecated.strategies.momentum_scalper import MomentumScalperStrategy

__all__ = []
