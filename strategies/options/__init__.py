"""
Options-Native Strategy Library

Hierarchical Decision Tree Architecture:

┌─────────────────────────────────────────────────────────────────────────────┐
│                        OPTIONS STRATEGY TREE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LEVEL 1: BRAIN (Market Regime Detection)                                   │
│  ├── brain/router.py                                                        │
│  │   └── select_option_strategy(vix, velocity, zscore, adx)                │
│  │                                                                          │
│  └── Routes to:                                                             │
│      ├── PANIC (VIX >= 22, Z < -2.5) ────────────> Vega Snap               │
│      ├── EXPLOSIVE (Velocity >= 0.3%) ───────────> Gamma Scalper           │
│      ├── TRENDING (ADX >= 28, low vel) ──────────> Delta Surfer            │
│      └── DEFAULT ────────────────────────────────> Gamma Scalper (wait)    │
│                                                                              │
│  LEVEL 2: STRATEGIES (Specialized for market conditions)                    │
│  ├── strategies/gamma/          <- Has Level 3 sub-routing                 │
│  │   ├── scalper.py             <- Ride explosions                         │
│  │   └── reversal.py            <- Fade exhausted moves                    │
│  │                                                                          │
│  ├── strategies/vega_snap.py    <- Panic reversals                         │
│  └── strategies/delta_surfer.py <- Trend following                         │
│                                                                              │
│  LEVEL 3: SUB-STRATEGIES (Strategy-specific routing)                        │
│  └── strategies/gamma/                                                      │
│      └── Exhaustion Score triggers reversal_scalper                        │
│          - Rule 1: Score >= 2 AND VIX < 25                                 │
│          - Rule 2: Session == midday AND Score >= 1                        │
│          - Rule 3: Score >= 3 (any VIX)                                    │
│                                                                              │
│  CORE: Shared base classes and types                                        │
│  └── core/base_options.py                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

These strategies are designed specifically for options physics:
- Time is TOXIC (Theta decay)
- Convexity is POWER (Gamma acceleration)
- Volatility is TREACHEROUS (Vega crush/expansion)

DO NOT use stock strategies to trade options. Different physics.

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

# Core components
from strategies.options.core.base_options import (
    BaseOptionStrategy,
    OptionSignal,
    OptionSignalType,
    ContractSpec,
    OptionType,
    ContractSelection,
    OptionPosition,
)

# Level 2 Strategies
from strategies.options.strategies.gamma.scalper import GammaScalperStrategy
from strategies.options.strategies.gamma.reversal import ReversalScalperStrategy
from strategies.options.strategies.vega_snap import VegaSnapStrategy
from strategies.options.strategies.delta_surfer import DeltaSurferStrategy

# Level 1 Brain (Router)
from strategies.options.brain.router import (
    get_option_strategy,
    get_option_strategy_for_condition,
    list_option_strategies,
    get_option_strategy_info,
    select_option_strategy,
    OptionStrategyManager,
    create_option_strategy_manager,
)

__all__ = [
    # Core classes
    "BaseOptionStrategy",
    "OptionSignal",
    "OptionSignalType",
    "ContractSpec",
    "OptionType",
    "ContractSelection",
    "OptionPosition",
    # Level 2 Strategies
    "GammaScalperStrategy",
    "ReversalScalperStrategy",
    "VegaSnapStrategy",
    "DeltaSurferStrategy",
    # Level 1 Brain functions
    "get_option_strategy",
    "get_option_strategy_for_condition",
    "list_option_strategies",
    "get_option_strategy_info",
    "select_option_strategy",
    "OptionStrategyManager",
    "create_option_strategy_manager",
]
