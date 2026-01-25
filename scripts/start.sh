#!/bin/bash
#
# Bi-Cameral Trading Bot - Tick-Tock Scheduler
# Runs Reflex Engine (Body) at :00 and Strategy Manager (Brain) at :30
#

# Force Python to flush logs immediately (Critical for real-time output)
export PYTHONUNBUFFERED=1

# Define Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get script directory and move to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}   BI-CAMERAL TRADING BOT - TICK-TOCK SCHEDULER  ${NC}"
echo -e "${YELLOW}   Symbol: $(grep -o '"symbol": "[^"]*"' config.json | cut -d'"' -f4)${NC}"
echo -e "${YELLOW}   Schedule: Reflex @ :00 | Manager @ :30       ${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Trap Ctrl+C to kill all background jobs gracefully
cleanup() {
    echo ""
    echo -e "${YELLOW}[SYSTEM] Shutting down all agents...${NC}"
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo -e "${YELLOW}[SYSTEM] All processes terminated. Goodbye.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Activate Virtual Environment
if [ -d "venv" ]; then
    echo -e "${YELLOW}[SYSTEM] Activating virtual environment...${NC}"
    source venv/bin/activate
else
    echo -e "${RED}[ERROR] Virtual environment not found. Run: python -m venv venv${NC}"
    exit 1
fi

# Verify dependencies
echo -e "${YELLOW}[SYSTEM] Checking dependencies...${NC}"
python -c "import alpaca, numpy, yfinance, google.generativeai" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Missing dependencies. Run: pip install -r requirements.txt${NC}"
    exit 1
fi

echo -e "${YELLOW}[SYSTEM] Starting Tick-Tock Scheduler...${NC}"
echo ""

# ===== Start The Body (Reflex Engine) =====
# It handles its own :00 alignment internally
echo -e "${GREEN}[REFLEX] Starting execution engine (aligned to :00)...${NC}"
(
    python reflex.py 2>&1 | while IFS= read -r line; do
        echo -e "${GREEN}[REFLEX]${NC} $line"
    done
) &
REFLEX_PID=$!

# Give reflex a moment to warm up
sleep 2

# ===== Start The Brain (Manager) - Aligned to :30 =====
echo -e "${CYAN}[MANAGER] Starting strategy manager (aligned to :30)...${NC}"
(
    # Run immediately on start for initial sync
    echo -e "${CYAN}[MANAGER] Initial strategy synthesis...${NC}"
    python manager.py 2>&1 | while IFS= read -r line; do
        echo -e "${CYAN}[MANAGER]${NC} $line"
    done

    # Then loop targeting the :30 mark of each minute
    while true; do
        # Calculate seconds to sleep to hit the next :30 mark
        current_sec=$(date +%S)
        # Remove leading zero to avoid octal interpretation
        current_sec=$((10#$current_sec))

        # Logic:
        # If current is 10s, target is 30s -> Sleep 20s
        # If current is 40s, target is next minute's 30s -> Sleep 50s (90-40)
        if [ $current_sec -lt 30 ]; then
            sleep_time=$((30 - current_sec))
        else
            sleep_time=$((90 - current_sec))
        fi

        echo -e "${YELLOW}[SYNC] Aligning Manager to :30 mark (sleeping ${sleep_time}s)...${NC}"
        sleep $sleep_time

        echo -e "${CYAN}[MANAGER] Analyzing market (Time: $(date +%T))...${NC}"
        python manager.py 2>&1 | while IFS= read -r line; do
            echo -e "${CYAN}[MANAGER]${NC} $line"
        done
    done
) &
MANAGER_PID=$!

echo ""
echo -e "${YELLOW}=================================================${NC}"
echo -e "${YELLOW}   TICK-TOCK SYSTEM RUNNING                      ${NC}"
echo -e "${YELLOW}   Reflex PID: $REFLEX_PID  (executes at :00)    ${NC}"
echo -e "${YELLOW}   Manager PID: $MANAGER_PID (analyzes at :30)   ${NC}"
echo -e "${YELLOW}   Press Ctrl+C to stop                          ${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Wait for background processes
wait
