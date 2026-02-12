"""
Level 1: Brain - Market Regime Detection & Strategy Routing

The brain analyzes broad market conditions and routes to the appropriate
Level 2 strategy:

    VIX + Velocity + Z-Score + ADX
              |
              v
    ┌─────────────────────────────────────┐
    │            BRAIN ROUTER              │
    │   select_option_strategy()           │
    └─────────────────────────────────────┘
              |
    ┌─────────┼─────────┬─────────────────┐
    v         v         v                 v
  PANIC   EXPLOSIVE  TRENDING         DEFAULT
    |         |         |                 |
    v         v         v                 v
 VegaSnap  Gamma    Delta           GammaScalper
           Scalper   Surfer          (waiting)

Priority Order:
    1. PANIC: VIX >= 22 AND Z-Score < -2.5 -> Vega Snap
    2. EXPLOSIVE: Velocity >= 0.3% -> Gamma Scalper
    3. TRENDING: ADX >= 28 AND Velocity < 0.2% -> Delta Surfer
    4. DEFAULT: Gamma Scalper (wait for explosions)
"""

from strategies.options.brain.router import (
    get_option_strategy,
    get_option_strategy_for_condition,
    list_option_strategies,
    get_option_strategy_info,
    select_option_strategy,
    OptionStrategyManager,
    create_option_strategy_manager,
    OPTION_STRATEGY_REGISTRY,
    CONDITION_STRATEGY_MAP,
)

__all__ = [
    "get_option_strategy",
    "get_option_strategy_for_condition",
    "list_option_strategies",
    "get_option_strategy_info",
    "select_option_strategy",
    "OptionStrategyManager",
    "create_option_strategy_manager",
    "OPTION_STRATEGY_REGISTRY",
    "CONDITION_STRATEGY_MAP",
]
