#!/usr/bin/env python3
"""
Strategy Manager v2 - The "Brain" of the Bi-Cameral Trading Bot

This script runs periodically (via cron) to:
1. Gather market intelligence (sentiment, news, VIX, internals)
2. Synthesize a strategy using Google Gemini AI
3. Select the optimal STRATEGY from the Strategy Library
4. Update config.json with symbol, buy_enabled, AND active_strategy
5. Log all decisions to SQLite for auditability

=== STRATEGY SELECTION ===
The Manager now selects from 4 strategy "cartridges":
- momentum_scalper: For trending markets (ADX > 25)
- mean_reversion: For choppy markets (ADX < 20)
- volatility_breakout: For high volatility / news days
- crisis_alpha: For VIX > 25 bear markets (uses SQQQ)

Author: Bi-Cameral System
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import google.generativeai as genai
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

import scraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class StrategyManager:
    """
    The AI-powered strategy manager that adjusts trading parameters
    based on market intelligence.

    Now also selects active_strategy from the Strategy Library.
    """

    BASE_DIR = Path(__file__).parent
    CONFIG_PATH = BASE_DIR / "config.json"
    CONFIG_TMP_PATH = BASE_DIR / "config.tmp"
    DB_PATH = BASE_DIR / "data" / "trades.db"

    # Dampening constraints (10% rule)
    MAX_DEVIATION = 0.10

    # Gemini model
    MODEL = "gemini-2.0-flash"

    # Available strategies from Strategy Library
    AVAILABLE_STRATEGIES = [
        "momentum_scalper",   # TREND regime - ADX > 25
        "mean_reversion",     # CHOP regime - ADX < 20
        "volatility_breakout", # VOLATILE regime - High ATR / News days
        "crisis_alpha",       # CRISIS regime - VIX > 25, uses SQQQ
    ]

    # Strategy-Symbol mapping
    STRATEGY_SYMBOL_MAP = {
        "momentum_scalper": "TQQQ",
        "mean_reversion": "TQQQ",
        "volatility_breakout": "TQQQ",
        "crisis_alpha": "SQQQ",  # Crisis uses inverse ETF
    }

    def __init__(self):
        """Initialize the Strategy Manager."""
        load_dotenv(self.BASE_DIR / ".env", override=True)

        # Verify Google API key
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY must be set in .env")

        # Configure Google Generative AI
        genai.configure(api_key=self.google_api_key)

        # Initialize Gemini model
        self.model = genai.GenerativeModel(self.MODEL)

        # Initialize Alpaca data client for ADX calculation
        alpaca_key = os.getenv("ALPACA_KEY")
        alpaca_secret = os.getenv("ALPACA_SECRET")
        if alpaca_key and alpaca_secret:
            self.data_client = StockHistoricalDataClient(alpaca_key, alpaca_secret)
            logger.info("[INIT] Alpaca data client initialized for ADX")
        else:
            self.data_client = None
            logger.warning("[INIT] No Alpaca credentials - ADX will use default value")

        # Database connection with WAL mode
        self.db_conn = sqlite3.connect(str(self.DB_PATH), check_same_thread=False)
        self.db_conn.execute("PRAGMA journal_mode=WAL;")
        self.db_conn.row_factory = sqlite3.Row

        # Load current config
        self.current_config = self._load_config()

        logger.info(f"[INIT] StrategyManager initialized for {self.current_config['symbol']}")
        logger.info(f"[INIT] Using Gemini model: {self.MODEL}")

    def _load_config(self) -> dict:
        """Load current configuration from config.json."""
        default_config = {
            "symbol": "SPY",
            "buy_enabled": False,
            "buy_rsi_threshold": 30,
            "stop_loss_pct": 0.02,
            "dampening_factor": 0.1,
            "last_update": "UNKNOWN"
        }

        try:
            with open(self.CONFIG_PATH, "r") as f:
                config = json.load(f)
                return {**default_config, **config}
        except (json.JSONDecodeError, FileNotFoundError, IOError) as e:
            logger.warning(f"Config read failed ({e}), using defaults")
            return default_config

    async def gather_intelligence(self) -> dict:
        """
        Gather all market intelligence from various sources.

        Now includes ADX (Average Directional Index) for trend strength detection.
        This is the "Physics Layer" that determines road conditions.

        Returns:
            Dictionary containing sentiment, news, macro data, ADX, and politician trades.
        """
        logger.info("[INTEL] Gathering market intelligence...")

        # Get market sentiment (Fear & Greed + News)
        sentiment = await scraper.get_market_sentiment()

        # Get market internals (VIX, TNX) - CRITICAL for TQQQ
        internals = scraper.get_market_internals()

        # Calculate ADX - THE PHYSICS LAYER
        # This measures actual trend strength, not sentiment
        symbol = self.current_config.get("symbol", "TQQQ")
        adx = self._calc_adx(symbol)

        # Query recent politician trades from database
        politician_trades = self._get_politician_trades()

        # Query recent execution history
        recent_trades = self._get_recent_executions()

        intelligence = {
            "fear_greed": sentiment["fear_greed"],
            "market_sentiment": sentiment["overall"],
            "headlines": sentiment["headlines"],
            "vix": internals["vix"],
            "vix_change_pct": internals["vix_change_pct"],
            "vix_alert": internals["vix_alert"],
            "tnx": internals["tnx"],
            "tnx_change_pct": internals["tnx_change_pct"],
            "tnx_alert": internals["tnx_alert"],
            "adx": adx,  # NEW: Trend strength indicator
            "politician_trades": politician_trades,
            "recent_executions": recent_trades,
            "timestamp": datetime.now().isoformat()
        }

        logger.info(f"[INTEL] Fear & Greed: {intelligence['fear_greed']}")
        logger.info(f"[INTEL] Sentiment: {intelligence['market_sentiment']}")
        logger.info(f"[INTEL] VIX: {intelligence['vix']} ({intelligence['vix_alert']})")
        logger.info(f"[INTEL] ADX: {intelligence['adx']:.1f} ({'TREND' if adx > 25 else 'CHOP' if adx < 20 else 'TRANSITION'})")
        logger.info(f"[INTEL] TNX: {intelligence['tnx']}% ({intelligence['tnx_alert']})")
        logger.info(f"[INTEL] Headlines: {len(intelligence['headlines'])}")

        return intelligence

    def _get_politician_trades(self) -> list[dict]:
        """Query politician trades from the last 7 days."""
        try:
            cursor = self.db_conn.cursor()
            seven_days_ago = (datetime.now() - timedelta(days=7)).isoformat()

            cursor.execute("""
                SELECT politician_name, symbol, transaction_date, type, amount_bracket
                FROM politician_trades
                WHERE transaction_date >= ?
                ORDER BY transaction_date DESC
                LIMIT 20
            """, (seven_days_ago,))

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.warning(f"Failed to query politician trades: {e}")
            return []

    def _get_recent_executions(self) -> list[dict]:
        """Query recent trade executions for context."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT timestamp, symbol, side, price, qty, trigger_reason, profit_loss
                FROM executions
                ORDER BY timestamp DESC
                LIMIT 10
            """)

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

        except Exception as e:
            logger.warning(f"Failed to query executions: {e}")
            return []

    def _calc_adx(self, symbol: str = "TQQQ", period: int = 14) -> float:
        """
        Calculate current ADX (Average Directional Index) for regime detection.

        ADX measures trend STRENGTH (not direction):
        - ADX > 25: Strong trend (use momentum_scalper)
        - ADX < 20: Weak/no trend (use mean_reversion)
        - ADX 20-25: Transition zone

        Returns:
            Current ADX value, or 20.0 (neutral) if calculation fails
        """
        if not self.data_client:
            logger.warning("[ADX] No data client - using neutral default")
            return 20.0  # Neutral default

        try:
            # Get 50 bars for ADX calculation (need warmup)
            end = datetime.now()
            start = end - timedelta(days=3)  # 3 days covers ~50 minute bars

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end
            )

            bars = self.data_client.get_stock_bars(request)
            bars_data = bars.data if hasattr(bars, 'data') else {}
            symbol_bars = bars_data.get(symbol, [])

            if len(symbol_bars) < period * 2:
                logger.warning(f"[ADX] Insufficient bars ({len(symbol_bars)}) - using neutral default")
                return 20.0

            # Convert to DataFrame
            df = pd.DataFrame({
                'high': [float(b.high) for b in symbol_bars],
                'low': [float(b.low) for b in symbol_bars],
                'close': [float(b.close) for b in symbol_bars]
            })

            # Calculate True Range components
            high = df['high']
            low = df['low']
            close = df['close']

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()

            # Calculate Directional Movement
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low

            plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
            minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

            plus_di = 100 * plus_dm.rolling(window=period).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=period).mean() / atr

            # Calculate ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()

            # Get the latest ADX value
            current_adx = adx.iloc[-1]

            if pd.isna(current_adx):
                logger.warning("[ADX] Calculation returned NaN - using neutral default")
                return 20.0

            logger.info(f"[ADX] {symbol} current ADX: {current_adx:.1f}")
            return float(current_adx)

        except Exception as e:
            logger.error(f"[ADX] Calculation failed: {e}")
            return 20.0  # Neutral default on error

    def _deterministic_regime(self, intelligence: dict) -> dict:
        """
        DETERMINISTIC Regime Detection - The Traffic Controller.

        === 3-LAYER ARCHITECTURE ===

        LAYER 1: SAFETY (The Kill Switch)
        - VIX > 35 OR F&G < 15 OR F&G > 85 → CASH
        - VIX > 28 OR F&G < 20 → CRISIS_ALPHA (SQQQ)
        * AI CANNOT override this layer *

        LAYER 2: PHYSICS (The Road Conditions)
        - ADX < 25 → MEAN_REVERSION (no trend = no momentum)
        - This is the Dec 2 fix: ADX was 11.3, should have blocked momentum
        * AI CANNOT override this layer *

        LAYER 3: OPTIMIZATION (The Driver)
        - ADX > 25 AND F&G > 50 → MOMENTUM_SCALPER
        - VIX 20-28 AND F&G < 40 → VOLATILITY_BREAKOUT
        * AI can VETO (buy_enabled=false) but cannot change strategy *

        Returns:
            Dict with strategy, symbol, buy_enabled, reasoning
        """
        vix = intelligence.get('vix', 15)
        fg = intelligence.get('fear_greed', 50)
        adx = intelligence.get('adx', 20)  # NEW: Physics layer

        logger.info(f"[REGIME] Deterministic check: VIX={vix}, F&G={fg}, ADX={adx:.1f}")

        # ============================================================
        # LAYER 1: SAFETY (The Kill Switch) - AI CANNOT OVERRIDE
        # ============================================================

        # Extreme VIX - Too volatile for any leveraged ETF
        if vix > 35:
            logger.warning(f"[SAFETY] EXTREME VIX={vix} > 35 - Going to CASH")
            return {
                "active_strategy": "crisis_alpha",
                "symbol": "SQQQ",
                "buy_enabled": False,
                "stop_loss_pct": 0.015,
                "reasoning": f"SAFETY: VIX={vix} > 35. Extreme volatility destroys leveraged ETFs. CASH."
            }

        # Extreme Fear - Capitulation zone
        if fg < 15:
            logger.warning(f"[SAFETY] EXTREME F&G={fg} < 15 - Capitulation zone")
            return {
                "active_strategy": "mean_reversion",
                "symbol": "TQQQ",
                "buy_enabled": False,
                "stop_loss_pct": 0.01,
                "reasoning": f"SAFETY: F&G={fg} < 15. Capitulation zone. Wait for reversal signal."
            }

        # Extreme Greed - Euphoria/reversal risk
        if fg > 85:
            logger.warning(f"[SAFETY] EXTREME F&G={fg} > 85 - Euphoria zone")
            return {
                "active_strategy": "momentum_scalper",
                "symbol": "TQQQ",
                "buy_enabled": False,
                "stop_loss_pct": 0.008,
                "reasoning": f"SAFETY: F&G={fg} > 85. Euphoria/reversal risk. Protect gains."
            }

        # Crisis Regime - VIX > 28 OR F&G < 20
        if vix > 28 or fg < 20:
            logger.warning(f"[SAFETY] CRISIS detected: VIX={vix}, F&G={fg}")
            return {
                "active_strategy": "crisis_alpha",
                "symbol": "SQQQ",
                "buy_enabled": True,
                "stop_loss_pct": 0.015,
                "reasoning": f"CRISIS: VIX={vix}, F&G={fg}. Market panic - short via SQQQ."
            }

        # ============================================================
        # LAYER 2: PHYSICS (The Road Conditions) - AI CANNOT OVERRIDE
        # ============================================================
        # This is the Dec 2 fix: If ADX < 25, there's NO TREND to ride
        # It doesn't matter what VIX or F&G say - physics trumps sentiment

        if adx < 25:
            # No trend strength - MUST use mean_reversion
            logger.info(f"[PHYSICS] ADX={adx:.1f} < 25 - No trend strength. Forcing MEAN_REVERSION.")
            return {
                "active_strategy": "mean_reversion",
                "symbol": "TQQQ",
                "buy_enabled": True,
                "stop_loss_pct": 0.008,
                "reasoning": f"PHYSICS: ADX={adx:.1f} < 25. No trend strength. Scalp the ranges."
            }

        # ============================================================
        # LAYER 3: OPTIMIZATION (The Driver) - AI can VETO only
        # ============================================================
        # We've passed Safety and Physics - now optimize

        # Volatile Regime - VIX 20-28 with fear
        if 20 <= vix <= 28 and fg < 40:
            logger.info(f"[OPTIMIZE] VOLATILE: VIX={vix}, F&G={fg}. ORB strategy.")
            return {
                "active_strategy": "volatility_breakout",
                "symbol": "TQQQ",
                "buy_enabled": True,
                "stop_loss_pct": 0.012,
                "reasoning": f"VOLATILE: VIX={vix}, F&G={fg}, ADX={adx:.1f}. ORB for morning breakouts."
            }

        # Trend Regime - ADX > 25 (passed physics) AND F&G > 50 (confirmed greed)
        if adx > 25 and fg > 50:
            logger.info(f"[OPTIMIZE] TREND: ADX={adx:.1f}, F&G={fg}. Momentum scalping.")
            return {
                "active_strategy": "momentum_scalper",
                "symbol": "TQQQ",
                "buy_enabled": True,
                "stop_loss_pct": 0.01,
                "reasoning": f"TREND: ADX={adx:.1f} > 25, F&G={fg} > 50. Strong trend + greed = ride it."
            }

        # Default: Chop Regime (ADX > 25 but F&G neutral 20-50)
        logger.info(f"[OPTIMIZE] CHOP: ADX={adx:.1f}, F&G={fg}. Neutral market.")
        return {
            "active_strategy": "mean_reversion",
            "symbol": "TQQQ",
            "buy_enabled": True,
            "stop_loss_pct": 0.008,
            "reasoning": f"CHOP: ADX={adx:.1f}, F&G={fg}. Sideways/neutral market - scalp the ranges."
        }

    async def synthesize(self, intelligence: dict) -> dict:
        """
        Use DETERMINISTIC regime detection + Gemini for edge cases.

        The Decision Matrix runs FIRST. Gemini only validates or adjusts
        for nuanced situations (headlines, yield curve, etc.).

        Args:
            intelligence: Dictionary from gather_intelligence()

        Returns:
            Dictionary with recommended config changes including active_strategy.
        """
        logger.info("[SYNTH] Running regime detection...")

        current_symbol = self.current_config.get("symbol", "TQQQ")
        current_strategy = self.current_config.get("active_strategy", "momentum_scalper")

        # STEP 1: Run deterministic regime detection FIRST
        regime_decision = self._deterministic_regime(intelligence)
        logger.info(f"[REGIME] Deterministic: {regime_decision['active_strategy']} ({regime_decision['reasoning'][:50]}...)")

        # Build the prompt - Gemini VALIDATES the deterministic decision
        prompt = f"""Role: Senior Portfolio Manager.
Goal: VALIDATE or OVERRIDE the regime detection based on nuanced signals.

=== DETERMINISTIC REGIME DECISION ===
The system has detected:
- Strategy: {regime_decision['active_strategy']}
- Symbol: {regime_decision['symbol']}
- Buy Enabled: {regime_decision['buy_enabled']}
- Reasoning: {regime_decision['reasoning']}

=== CURRENT MARKET DATA ===
- VIX: {intelligence['vix']} (Change: {intelligence['vix_change_pct']:+.1f}%)
- Fear & Greed: {intelligence['fear_greed']}/100 ({intelligence['market_sentiment']})
- TNX (10Y Yield): {intelligence['tnx']}% (Change: {intelligence['tnx_change_pct']:+.1f}%)
- Headlines: {json.dumps(intelligence['headlines'][:3], indent=2)}

=== STRATEGY LIBRARY ===
1. **momentum_scalper** - TREND (VIX < 20, F&G > 40) - TQQQ
2. **mean_reversion** - CHOP (sideways, indecisive) - TQQQ
3. **volatility_breakout** - VOLATILE (VIX 20-28, F&G < 40) - TQQQ
4. **crisis_alpha** - CRISIS (VIX > 28 OR F&G < 20) - SQQQ

=== YOUR TASK ===
Review the deterministic decision. You may:
1. CONFIRM it (recommended 90% of the time)
2. OVERRIDE only if headlines reveal imminent regime change

Output JSON only:
{{
    "active_strategy": "{regime_decision['active_strategy']}",
    "symbol": "{regime_decision['symbol']}",
    "buy_enabled": {str(regime_decision['buy_enabled']).lower()},
    "stop_loss_pct": {regime_decision['stop_loss_pct']},
    "reasoning": "Confirmed: [reason] OR Override: [reason]"
}}
"""

        try:
            # Use Gemini's native JSON mode for structured output
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    max_output_tokens=500
                )
            )

            # Parse JSON response
            response_text = response.text.strip()
            recommendation = json.loads(response_text)

            # Validate active_strategy
            ai_strategy = recommendation.get('active_strategy', regime_decision['active_strategy'])
            if ai_strategy not in self.AVAILABLE_STRATEGIES:
                logger.warning(f"[SYNTH] Invalid strategy '{ai_strategy}', using deterministic: {regime_decision['active_strategy']}")
                ai_strategy = regime_decision['active_strategy']

            # ============================================================
            # AI OVERRIDE PROTECTION - PHYSICS LAYER CANNOT BE VIOLATED
            # ============================================================
            adx = intelligence.get('adx', 20)
            vix = intelligence.get('vix', 15)
            fg = intelligence.get('fear_greed', 50)

            # BLOCK 1: If ADX < 25, AI cannot switch TO momentum_scalper
            if adx < 25 and ai_strategy == "momentum_scalper" and regime_decision['active_strategy'] != "momentum_scalper":
                logger.warning(f"[OVERRIDE BLOCKED] AI tried momentum_scalper but ADX={adx:.1f} < 25. PHYSICS VIOLATION.")
                logger.warning(f"[OVERRIDE BLOCKED] Forcing: {regime_decision['active_strategy']}")
                ai_strategy = regime_decision['active_strategy']
                recommendation['reasoning'] = f"AI Override Blocked: ADX={adx:.1f} < 25 violates Physics Layer. {recommendation.get('reasoning', '')}"

            # BLOCK 2: If VIX > 28 or F&G < 20, AI cannot switch FROM crisis_alpha
            if (vix > 28 or fg < 20) and regime_decision['active_strategy'] == "crisis_alpha" and ai_strategy != "crisis_alpha":
                logger.warning(f"[OVERRIDE BLOCKED] AI tried {ai_strategy} but VIX={vix}/F&G={fg}. SAFETY VIOLATION.")
                logger.warning(f"[OVERRIDE BLOCKED] Forcing: crisis_alpha")
                ai_strategy = "crisis_alpha"
                recommendation['reasoning'] = f"AI Override Blocked: VIX={vix}/F&G={fg} violates Safety Layer. {recommendation.get('reasoning', '')}"

            # BLOCK 3: AI can ONLY change buy_enabled, not strategy, if deterministic was from SAFETY or PHYSICS
            deterministic_source = "SAFETY" if "SAFETY" in regime_decision['reasoning'] or "CRISIS" in regime_decision['reasoning'] else \
                                   "PHYSICS" if "PHYSICS" in regime_decision['reasoning'] else "OPTIMIZE"

            if deterministic_source in ["SAFETY", "PHYSICS"] and ai_strategy != regime_decision['active_strategy']:
                logger.warning(f"[OVERRIDE BLOCKED] AI cannot override {deterministic_source} layer decision.")
                logger.warning(f"[OVERRIDE BLOCKED] Deterministic: {regime_decision['active_strategy']} | AI wanted: {ai_strategy}")
                ai_strategy = regime_decision['active_strategy']
                recommendation['reasoning'] = f"AI Override Blocked: {deterministic_source} layer is immutable. {recommendation.get('reasoning', '')}"

            recommendation['active_strategy'] = ai_strategy

            # Auto-set symbol based on strategy (crisis_alpha uses SQQQ)
            expected_symbol = self.STRATEGY_SYMBOL_MAP.get(ai_strategy, "TQQQ")
            if recommendation.get('symbol') != expected_symbol:
                logger.info(f"[SYNTH] Correcting symbol for {ai_strategy}: {recommendation.get('symbol')} -> {expected_symbol}")
                recommendation['symbol'] = expected_symbol

            # Log if Gemini successfully overrode (only possible in OPTIMIZE layer)
            if ai_strategy != regime_decision['active_strategy']:
                logger.warning(f"[OVERRIDE ACCEPTED] Gemini changed strategy: {regime_decision['active_strategy']} -> {ai_strategy}")
                logger.warning(f"[OVERRIDE ACCEPTED] Reason: {recommendation.get('reasoning', 'N/A')[:80]}")

            logger.info(f"[SYNTH] Final decision: strategy={ai_strategy}, symbol={recommendation['symbol']}, buy_enabled={recommendation.get('buy_enabled')}")
            logger.info(f"[SYNTH] Reasoning: {recommendation.get('reasoning', 'N/A')[:100]}...")

            return recommendation

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.warning(f"[FALLBACK] AI Brain Dead. Using Reflex Logic.")
            # FALLBACK: Use the deterministic decision - DO NOT disable buying
            return regime_decision

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            logger.warning(f"[FALLBACK] AI Brain Dead. Using Reflex Logic.")
            # FALLBACK: Use the deterministic decision - DO NOT disable buying
            return regime_decision

    def apply_dampening(self, recommendation: dict) -> dict:
        """
        Apply dampening constraints to prevent wild config swings.
        NOTE: Symbol and Strategy changes are NOT dampened - regime switches happen instantly.

        Args:
            recommendation: Raw recommendation from Gemini

        Returns:
            Dampened configuration values including active_strategy
        """
        logger.info("[DAMPEN] Applying dampening constraints...")

        current_stop = self.current_config.get("risk_management", {}).get("stop_loss_pct", 0.005)
        current_symbol = self.current_config.get("symbol", "TQQQ")
        current_strategy = self.current_config.get("active_strategy", "momentum_scalper")

        # Get recommended values
        ai_symbol = recommendation.get("symbol", current_symbol)
        ai_stop = recommendation.get("stop_loss_pct", current_stop)
        ai_strategy = recommendation.get("active_strategy", current_strategy)

        # STRATEGY AND SYMBOL ARE NOT DAMPENED - Regime changes happen immediately
        # This is critical for rapid adaptation to market conditions
        dampened_symbol = ai_symbol
        dampened_strategy = ai_strategy

        # Apply 10% rule to stop loss only
        min_stop = current_stop * (1 - self.MAX_DEVIATION)
        max_stop = current_stop * (1 + self.MAX_DEVIATION)
        dampened_stop = max(min_stop, min(ai_stop, max_stop))

        # Enforce absolute bounds for stop loss
        dampened_stop = max(0.003, min(0.02, round(dampened_stop, 4)))

        dampened = {
            "active_strategy": dampened_strategy,
            "symbol": dampened_symbol,
            "buy_enabled": recommendation.get("buy_enabled", False),
            "stop_loss_pct": dampened_stop,
            "reasoning": recommendation.get("reasoning", "")
        }

        # Log strategy change prominently
        if dampened_strategy != current_strategy:
            logger.info(f"[STRATEGY] SWITCH: {current_strategy} -> {dampened_strategy}")

        # Log symbol change prominently
        if dampened_symbol != current_symbol:
            logger.info(f"[SYMBOL] SWITCH: {current_symbol} -> {dampened_symbol}")

        if dampened_stop != ai_stop:
            logger.info(f"[DAMPEN] Stop loss clamped: {ai_stop} -> {dampened_stop}")

        return dampened

    def atomic_write(self, new_config: dict):
        """
        Atomically update config.json using write-then-rename pattern.

        This prevents race conditions with reflex.py reading the config.
        Now includes active_strategy field.
        """
        logger.info("[WRITE] Performing atomic config update...")

        # Merge with current config - preserving nested structure
        updated_config = {
            **self.current_config,
            "symbol": new_config["symbol"],
            "buy_enabled": new_config["buy_enabled"],
            "active_strategy": new_config["active_strategy"],
            "last_update": datetime.now().isoformat()
        }

        # Update nested risk_management stop_loss
        if "risk_management" not in updated_config:
            updated_config["risk_management"] = {}
        updated_config["risk_management"]["stop_loss_pct"] = new_config["stop_loss_pct"]

        try:
            # Write to temporary file
            with open(self.CONFIG_TMP_PATH, "w") as f:
                json.dump(updated_config, f, indent=2)
                f.write("\n")
                # Force flush to OS buffer
                f.flush()
                # Force OS to write to disk
                os.fsync(f.fileno())

            # Atomic rename (POSIX guarantees atomicity)
            os.replace(self.CONFIG_TMP_PATH, self.CONFIG_PATH)

            logger.info("[WRITE] Config updated atomically")
            logger.info(f"[WRITE] New config: strategy={updated_config['active_strategy']}, "
                        f"symbol={updated_config['symbol']}, "
                        f"buy_enabled={updated_config['buy_enabled']}, "
                        f"stop_loss={new_config['stop_loss_pct']}")

        except Exception as e:
            logger.error(f"Atomic write failed: {e}")
            # Clean up temp file if it exists
            if self.CONFIG_TMP_PATH.exists():
                self.CONFIG_TMP_PATH.unlink()
            raise

    def log_decision(self, intelligence: dict, recommendation: dict, dampened: dict):
        """
        Log the decision to SQLite for auditability.
        Now includes active_strategy tracking.
        """
        try:
            cursor = self.db_conn.cursor()

            # Calculate config diff - includes strategy, symbol, buy_enabled
            config_diff = {
                "active_strategy": {
                    "before": self.current_config.get("active_strategy", "momentum_scalper"),
                    "after": dampened.get("active_strategy", "momentum_scalper")
                },
                "symbol": {
                    "before": self.current_config.get("symbol", "TQQQ"),
                    "after": dampened.get("symbol", "TQQQ")
                },
                "buy_enabled": {
                    "before": self.current_config.get("buy_enabled", False),
                    "after": dampened["buy_enabled"]
                },
                "stop_loss_pct": {
                    "before": self.current_config.get("risk_management", {}).get("stop_loss_pct", 0.005),
                    "after": dampened["stop_loss_pct"]
                }
            }

            cursor.execute("""
                INSERT INTO agent_logs (timestamp, sentiment_score, decision_reasoning, config_diff)
                VALUES (?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                intelligence["fear_greed"],
                dampened.get("reasoning", ""),
                json.dumps(config_diff)
            ))

            self.db_conn.commit()
            logger.info("[LOG] Decision logged to database")

        except Exception as e:
            logger.error(f"Failed to log decision: {e}")

    async def run(self):
        """
        Main execution flow for the Strategy Manager.
        """
        print("=" * 60)
        print("  STRATEGY MANAGER v2 - Bi-Cameral Trading Bot")
        print("  Powered by Google Gemini + Strategy Library")
        print("=" * 60)

        try:
            # Step 1: Gather intelligence
            intelligence = await self.gather_intelligence()

            # Step 2: Synthesize strategy with Gemini
            recommendation = await self.synthesize(intelligence)

            # Step 3: Apply dampening constraints
            dampened = self.apply_dampening(recommendation)

            # Step 4: Log the decision (before write, in case write fails)
            self.log_decision(intelligence, recommendation, dampened)

            # Step 5: Atomically update config
            self.atomic_write(dampened)

            # Strategy name mapping for display
            strategy_names = {
                "momentum_scalper": "The Sniper",
                "mean_reversion": "The Rubber Band",
                "volatility_breakout": "The News Trader",
                "crisis_alpha": "The Bear"
            }
            strategy_display = strategy_names.get(dampened['active_strategy'], dampened['active_strategy'])

            print("\n" + "=" * 60)
            print("  STRATEGY UPDATE COMPLETE")
            print("=" * 60)
            print(f"  Active Strategy: {dampened['active_strategy']} ({strategy_display})")
            print(f"  Symbol: {dampened['symbol']} ({'BULL' if dampened['symbol'] == 'TQQQ' else 'BEAR'} mode)")
            print(f"  Buy Enabled: {dampened['buy_enabled']}")
            print(f"  Stop Loss: {dampened['stop_loss_pct'] * 100:.2f}%")
            print(f"  Reasoning: {dampened['reasoning'][:80]}...")
            print("=" * 60)

        except Exception as e:
            logger.error(f"Strategy Manager failed: {e}")
            raise

        finally:
            self.db_conn.close()


async def main():
    """Entry point for the Strategy Manager."""
    manager = StrategyManager()
    await manager.run()


if __name__ == "__main__":
    asyncio.run(main())
