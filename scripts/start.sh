#!/bin/bash
#
# Options Trading Bot - Launcher
# Pure algorithmic trading with brain strategy switching
#

# Force Python to flush logs immediately
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
echo -e "${YELLOW}   OPTIONS TRADING BOT                           ${NC}"
echo -e "${YELLOW}   Brain: Vega Snap -> Gamma Scalper -> Delta Surfer ${NC}"
echo -e "${YELLOW}=================================================${NC}"
echo ""

# Trap Ctrl+C
cleanup() {
    echo ""
    echo -e "${YELLOW}[SYSTEM] Shutting down...${NC}"
    kill $(jobs -p) 2>/dev/null
    wait 2>/dev/null
    echo -e "${YELLOW}[SYSTEM] Goodbye.${NC}"
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
python -c "import alpaca, numpy, pandas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Missing dependencies. Run: pip install -r requirements.txt${NC}"
    exit 1
fi

# Parse arguments
DRY_RUN=""
VIX=""
for arg in "$@"; do
    case $arg in
        --dry-run)
            DRY_RUN="--dry-run"
            echo -e "${CYAN}[MODE] DRY RUN - No real orders${NC}"
            ;;
        --vix=*)
            VIX="--vix ${arg#*=}"
            echo -e "${CYAN}[VIX] Override: ${arg#*=}${NC}"
            ;;
    esac
done

echo ""
echo -e "${GREEN}[START] Launching Options Trading Bot...${NC}"
echo ""

# Run the bot
python options_bot.py $DRY_RUN $VIX

# Exit code
exit $?
