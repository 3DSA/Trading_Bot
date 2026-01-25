# Bi-Cameral Trading Bot

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      BI-CAMERAL ARCHITECTURE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐         ┌─────────────┐         ┌──────────┐ │
│   │   Scraper   │────────▶│   Manager   │────────▶│  config  │ │
│   │   (Eyes)    │         │   (Brain)   │         │   .json  │ │
│   └─────────────┘         └─────────────┘         └────┬─────┘ │
│         │                        │                     │       │
│         │ F&G, News              │ Gemini AI           │       │
│         │ VIX, Sentiment         │ 3-Layer Regime      │       │
│         │                        │ Detection           │       │
│         ▼                        ▼                     ▼       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    STRATEGY LIBRARY                      │  │
│   │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐     │  │
│   │  │  momentum    │ │    mean      │ │  volatility  │     │  │
│   │  │   scalper    │ │  reversion   │ │   breakout   │     │  │
│   │  │  (TREND)     │ │   (CHOP)     │ │  (VOLATILE)  │     │  │
│   │  └──────────────┘ └──────────────┘ └──────────────┘     │  │
│   │                    ┌──────────────┐                      │  │
│   │                    │   crisis     │                      │  │
│   │                    │    alpha     │                      │  │
│   │                    │  (CRISIS)    │                      │  │
│   │                    └──────────────┘                      │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                    REFLEX ENGINE                         │  │
│   │              (Strategy-Agnostic Executor)                │  │
│   │                                                          │  │
│   │    Loads active_strategy from config.json                │  │
│   │    Executes signals via Alpaca API                       │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Core Files

| File | Purpose |
|------|---------|
| `reflex.py` | **Execution Engine** - Loads strategies, streams market data, executes signals via Alpaca |
| `manager.py` | **AI Brain** - Uses Gemini to analyze market conditions and select optimal strategy |
| `scraper.py` | **Data Collector** - Scrapes Fear & Greed Index, news headlines, market sentiment |
| `config.json` | **Configuration** - Active strategy, risk parameters, symbol settings |

### Strategy Library (`strategies/`)

| Strategy | Regime | Description |
|----------|--------|-------------|
| `momentum_scalper.py` | TREND (ADX > 25) | EMA crossover + VWAP confirmation for trending markets |
| `mean_reversion.py` | CHOP (ADX < 20) | Z-Score + Bollinger Bands for range-bound markets |
| `volatility_breakout.py` | VOLATILE | Opening Range Breakout for high volatility / news days |
| `crisis_alpha.py` | CRISIS (VIX > 25) | Inverse ETF (SQQQ) strategy for bear markets |

### 3-Layer Regime Detection

The Manager uses a hierarchical decision system:

```
Layer 1: SAFETY (Cannot be overridden)
├── VIX > 30 → CRISIS regime (uses SQQQ)
├── Fear & Greed < 20 → CRISIS regime
└── Otherwise → proceed to Layer 2

Layer 2: PHYSICS (Cannot be overridden)
├── ADX > 25 → TREND regime (momentum_scalper)
├── ADX < 20 → CHOP regime (mean_reversion)
└── ADX 20-25 → BUFFER zone, proceed to Layer 3

Layer 3: OPTIMIZATION (AI can adjust)
└── Gemini AI fine-tunes within the determined regime
    └── AI can VETO trades but CANNOT override Safety/Physics
```

## Quick Start

### Prerequisites

- Python 3.11+
- Alpaca Paper Trading Account
- Google AI (Gemini) API Key

### Installation

```bash
# Clone the repository
git clone https://github.com/3DSA/Trading_Bot.git
cd Trading_Bot

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for scraping)
playwright install chromium
```

### Configuration

Create a `.env` file in the project root:

```env
# Alpaca API (Paper Trading)
ALPACA_KEY=your_alpaca_api_key
ALPACA_SECRET=your_alpaca_secret_key

# Google Gemini AI
GOOGLE_API_KEY=your_google_api_key
```

### Running

```bash
# Start the execution engine (runs continuously)
python reflex.py

# Run the AI manager (typically via cron every 15-30 min)
python manager.py

# Or use the start script
./scripts/start.sh
```

## Testing & Backtesting

All backtests are in the `tests/` directory:

```bash
# Universal backtester - tests actual strategy code
python tests/backtest_universal.py --strategy momentum_scalper --days 90

# Available strategies: momentum_scalper, mean_reversion, volatility_breakout, crisis_alpha

# Options:
#   --strategy    Strategy to test
#   --symbol      Target symbol (default: TQQQ)
#   --days        Days of history (default: 30)
#   --warmup      Warmup bars for indicators (default: 50)
```

## Project Structure

```
Trading_Bot/
├── reflex.py              # Execution engine
├── manager.py             # AI strategy manager
├── scraper.py             # Market data scraper
├── config.json            # Runtime configuration
├── requirements.txt       # Python dependencies
├── .env                   # API keys (not in repo)
│
├── strategies/            # Strategy library
│   ├── __init__.py
│   ├── base.py            # Base strategy class
│   ├── factory.py         # Strategy factory & manager
│   ├── shared_utils.py    # Common indicator calculations
│   ├── momentum_scalper.py
│   ├── mean_reversion.py
│   ├── volatility_breakout.py
│   └── crisis_alpha.py
│
├── tests/                 # Backtesting suite
│   ├── backtest_universal.py   # Main backtester
│   ├── backtest_scalper.py     # Scalper-specific tests
│   └── backtest_hybrid.py      # Multi-strategy tests
│
├── scripts/               # Shell scripts
│   ├── start.sh           # Start the bot
│   └── status.sh          # Check bot status
│
├── schema/                # Database schemas
│   └── schema.py
│
└── data/                  # Runtime data (gitignored)
    └── trades.db          # Trade history SQLite DB
```

## Strategy Development

### Creating a New Strategy

1. Create a new file in `strategies/`:

```python
from strategies.base import BaseStrategy, StrategySignal, SignalType

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    version = "1.0.0"
    description = "My custom strategy"
    preferred_regime = "TREND"

    def prepare_data(self, df):
        # Add your indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        return df

    def generate_signal(self, df, current_position=None, underlying_df=None):
        # Your signal logic
        if some_condition:
            return StrategySignal(
                signal=SignalType.BUY,
                reason="My buy reason",
                confidence=0.8
            )
        return StrategySignal(signal=SignalType.HOLD, reason="No setup")
```

2. Register in `strategies/factory.py`:

```python
from strategies.my_strategy import MyStrategy

STRATEGY_REGISTRY["my_strategy"] = MyStrategy
```

### Signal Types

| Signal | Action |
|--------|--------|
| `BUY` | Enter long position |
| `EXIT` | Close current position |
| `HOLD` | Do nothing |
| `SELL` | Enter short position (if supported) |

## Risk Management

Configured in `config.json`:

```json
{
  "risk_management": {
    "stop_loss_pct": 0.008,      // 0.8% stop loss
    "take_profit_pct": 0.02,     // 2% take profit
    "use_trailing_stop": true,
    "trailing_stop_pct": 0.005,  // 0.5% trailing stop
    "max_daily_loss_pct": 0.02,  // 2% max daily loss
    "cooldown_minutes": 15       // Minutes between trades
  }
}
```

## Key Concepts

### Double-Lock ADX Filter

The system uses both the leveraged ETF (TQQQ) AND the underlying (QQQ) ADX values to confirm regime:

- Leveraged ADX threshold: 30
- Underlying ADX threshold: 25
- Both must agree for trend confirmation

### AI Override Protection

The AI (Gemini) can suggest strategy changes, but:
- Cannot override Safety Layer (VIX/F&G)
- Cannot override Physics Layer (ADX regime)
- Can only VETO trades, not force them
- All AI decisions are logged for audit

## Troubleshooting

### Common Issues

1. **No trades executing**
   - Check if `buy_enabled` is true in config.json
   - Verify market hours (9:30 AM - 4:00 PM ET)
   - Check ADX values - might be in wrong regime

2. **Alpaca connection errors**
   - Verify API keys in .env
   - Ensure using paper trading keys for testing
   - Check Alpaca status page

3. **Strategy not loading**
   - Verify strategy name matches registry
   - Check for import errors in strategy file
