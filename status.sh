#!/bin/bash
#
# Bi-Cameral Trading Bot - Quick Status Check
# Shows current config, market data, and system health
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}   BI-CAMERAL BOT - STATUS CHECK                 ${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 1. Config Status
echo -e "${CYAN}[CONFIG]${NC}"
echo -e "  Symbol:      $(grep -o '"symbol": "[^"]*"' config.json | cut -d'"' -f4)"
echo -e "  Buy Enabled: $(grep -o '"buy_enabled": [^,]*' config.json | head -1 | cut -d':' -f2 | tr -d ' ')"
echo -e "  Stop Loss:   $(grep -o '"stop_loss_pct": [^,]*' config.json | head -1 | cut -d':' -f2 | tr -d ' ' | awk '{printf "%.1f%%", $1*100}')"
echo -e "  Last Update: $(grep -o '"last_update": "[^"]*"' config.json | cut -d'"' -f4)"
echo ""

# 2. Market Internals
echo -e "${GREEN}[MARKET INTERNALS]${NC}"
python -c "
import scraper
internals = scraper.get_market_internals()
print(f'  VIX:         {internals[\"vix\"]} ({internals[\"vix_alert\"]})')
print(f'  TNX (10Y):   {internals[\"tnx\"]}% ({internals[\"tnx_alert\"]})')
" 2>/dev/null || echo -e "  ${RED}Failed to fetch${NC}"
echo ""

# 3. Sentiment
echo -e "${CYAN}[SENTIMENT]${NC}"
python -c "
import asyncio
import scraper
fg = asyncio.run(scraper.get_fear_and_greed())
if fg < 25:
    status = 'EXTREME FEAR'
elif fg < 40:
    status = 'FEAR'
elif fg < 60:
    status = 'NEUTRAL'
elif fg < 75:
    status = 'GREED'
else:
    status = 'EXTREME GREED'
print(f'  Fear & Greed: {fg}/100 ({status})')
" 2>/dev/null || echo -e "  ${RED}Failed to fetch${NC}"
echo ""

# 4. Database Stats
echo -e "${YELLOW}[DATABASE]${NC}"
if [ -f "data/trades.db" ]; then
    EXEC_COUNT=$(sqlite3 data/trades.db "SELECT COUNT(*) FROM executions;" 2>/dev/null || echo "0")
    LOG_COUNT=$(sqlite3 data/trades.db "SELECT COUNT(*) FROM agent_logs;" 2>/dev/null || echo "0")
    echo -e "  Executions:  $EXEC_COUNT trades"
    echo -e "  Agent Logs:  $LOG_COUNT decisions"
else
    echo -e "  ${RED}Database not found${NC}"
fi
echo ""

echo -e "${YELLOW}=================================================${NC}"
