"""
Deprecated Code - Legacy Equity Trading System

The bot has transitioned to pure algorithmic OPTIONS trading.
This folder contains the old equity-based system.

Contents:
    Execution Layer (deprecated):
    - reflex.py: Universal strategy router (equity execution engine)
    - reflex_websocket.py: WebSocket-based execution (older version)
    - manager.py: Gemini AI brain for regime detection
    - scraper.py: Market data collector (F&G, news, sentiment)

    Strategy Layer (deprecated):
    - strategies/: Equity strategies (momentum_scalper, mean_reversion, etc.)
    - tests/: Equity backtests

Why deprecated:
    - Options trading provides better risk/reward (30x leverage vs 3x TQQQ)
    - Algorithmic brain (factory.py) outperforms AI manager
    - Simpler system = fewer failure points

Active code is now in:
    - strategies/options/ (3 options strategies)
    - tests/backtest_options.py (options backtester)

DO NOT import from this folder in active trading code.
"""
