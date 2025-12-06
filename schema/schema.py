import sqlite3
import os

os.makedirs('data', exist_ok=True)
conn = sqlite3.connect('data/trades.db')
cursor = conn.cursor()

# 1. Enable WAL Mode (CRITICAL for multi-process concurrency)
cursor.execute("PRAGMA journal_mode=WAL;")

# 2. Table: Executions (Fast-path logging)
cursor.execute('''
CREATE TABLE IF NOT EXISTS executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    trigger_reason TEXT,
    profit_loss REAL
)
''')

# 3. Table: Politician Trades (Knowledge Base)
cursor.execute('''
CREATE TABLE IF NOT EXISTS politician_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    politician_name TEXT,
    symbol TEXT,
    transaction_date TEXT,
    type TEXT,
    amount_bracket TEXT
)
''')

# 4. Table: Agent Logs (The "Brain's" Diary)
cursor.execute('''
CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    sentiment_score REAL,
    decision_reasoning TEXT,
    config_diff TEXT
)
''')

conn.commit()
conn.close()
print("Database initialized with WAL mode.")