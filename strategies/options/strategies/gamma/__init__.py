"""
Gamma Scalper Strategy Family - Level 2 + Level 3 Sub-routing

This folder contains the Gamma Scalper and its sub-strategies:

┌─────────────────────────────────────────────────────────────────┐
│                    GAMMA SCALPER SUB-TREE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│                  ┌─────────────────────┐                        │
│                  │    GAMMA SCALPER    │                        │
│                  │   (Level 2 Entry)   │                        │
│                  └──────────┬──────────┘                        │
│                             │                                   │
│                    Exhaustion Check                             │
│                             │                                   │
│              ┌──────────────┴──────────────┐                    │
│              │                             │                    │
│              v                             v                    │
│   ┌──────────────────┐          ┌──────────────────┐           │
│   │    SCALPER.PY    │          │   REVERSAL.PY    │           │
│   │ (Score < 2 or    │          │  (Score >= 2 or  │           │
│   │  VIX >= 25)      │          │   Midday+Score1) │           │
│   │                  │          │                  │           │
│   │ Ride explosion   │          │ Fade exhausted   │           │
│   │ continuation     │          │ move reversal    │           │
│   └──────────────────┘          └──────────────────┘           │
│                                                                 │
│  Exhaustion Score Components:                                   │
│    - RSI >= 65 or <= 35 (+1)                                   │
│    - Cumulative move >= 1% (+1)                                │
│    - Exhaustion volume >= 8x (+1)                              │
│    - Volume declining (+1)                                      │
│    - Bars in explosion >= 3 (+1)                               │
│                                                                 │
│  Reversal Trigger (compound logic):                            │
│    Rule 1: Score >= 2 AND VIX < 25                             │
│    Rule 2: Session == midday AND Score >= 1                    │
│    Rule 3: Score >= 3 (any VIX)                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Files:
    - scalper.py: Main gamma scalper (ride explosions)
    - reversal.py: Reversal scalper (fade exhausted moves)
"""

from strategies.options.strategies.gamma.scalper import GammaScalperStrategy
from strategies.options.strategies.gamma.reversal import ReversalScalperStrategy

__all__ = [
    "GammaScalperStrategy",
    "ReversalScalperStrategy",
]
