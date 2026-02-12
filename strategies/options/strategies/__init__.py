"""
Level 2: Strategy Library - Specialized Options Strategies

Each strategy is optimized for specific market conditions:

┌─────────────────────────────────────────────────────────────────┐
│                    LEVEL 2 STRATEGIES                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   GAMMA SCALPER  │  │    VEGA SNAP     │  │ DELTA SURFER │  │
│  │   (EXPLOSIVE)    │  │   (VOL SHIFT)    │  │  (TRENDING)  │  │
│  │                  │  │                  │  │              │  │
│  │  High velocity   │  │  VIX spikes      │  │  ADX > 28    │  │
│  │  Volume surge    │  │  Z-Score < -2.5  │  │  Steady move │  │
│  │  Quick scalps    │  │  Panic reversals │  │  Ride trend  │  │
│  └────────┬─────────┘  └──────────────────┘  └──────────────┘  │
│           │                                                     │
│           │ Level 3 Sub-routing                                 │
│           v                                                     │
│  ┌──────────────────┐                                          │
│  │ gamma/           │                                          │
│  │ ├── scalper.py   │  <- Ride explosions                      │
│  │ └── reversal.py  │  <- Fade exhausted moves                 │
│  └──────────────────┘                                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

from strategies.options.strategies.gamma import GammaScalperStrategy, ReversalScalperStrategy
from strategies.options.strategies.vega_snap import VegaSnapStrategy
from strategies.options.strategies.delta_surfer import DeltaSurferStrategy

__all__ = [
    "GammaScalperStrategy",
    "ReversalScalperStrategy",
    "VegaSnapStrategy",
    "DeltaSurferStrategy",
]
