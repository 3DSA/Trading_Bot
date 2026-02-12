#!/usr/bin/env python3
"""
Yearly Backtest Runner with Detailed Trade Logging

Runs options brain backtester for each year from 2018 to present
and saves:
1. Consolidated results to backtest_yearly_results.txt
2. Detailed trade logs per strategy per year to logs/{strategy}/{year}.log

Usage:
    python tests/run_yearly_backtest.py
"""

import sys
from pathlib import Path
from datetime import datetime
import pytz

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.backtest_options import OptionsBacktester, OptionsBacktestResult

# Base directory for logs
BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Timezone for display
EASTERN = pytz.timezone('America/New_York')


def to_est(dt: datetime) -> datetime:
    """Convert datetime to Eastern Time."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return dt.astimezone(EASTERN)


def setup_log_directories():
    """Create log directories for each strategy."""
    strategies = ["gamma_scalper", "reversal_scalper", "vega_snap", "delta_surfer"]
    for strategy in strategies:
        log_dir = LOGS_DIR / strategy
        log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[SETUP] Log directories created at {LOGS_DIR}")


def write_strategy_trade_logs(year: str, result: OptionsBacktestResult):
    """
    Write detailed trade logs for each strategy for a given year.

    Creates logs/{strategy}/{year}.log with full trade details.
    """
    strategies = ["gamma_scalper", "reversal_scalper", "vega_snap", "delta_surfer"]

    for strategy_name in strategies:
        # Filter trades for this strategy
        strategy_trades = [t for t in result.trades if t.strategy_name == strategy_name]

        log_path = LOGS_DIR / strategy_name / f"{year}.log"

        lines = []
        lines.append("=" * 100)
        lines.append(f"  {strategy_name.upper()} TRADE LOG - {year}")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"  VIX Mode: {result.vix_mode.upper()}")
        if result.vix_stats:
            lines.append(f"  VIX Range: {result.vix_stats['min']:.1f} - {result.vix_stats['max']:.1f} (Mean: {result.vix_stats['mean']:.1f})")
            lines.append(f"  VIX >= 22 Days: {result.vix_stats['elevated_days']} / {result.vix_stats['total_days']}")
        lines.append("=" * 100)

        # Summary stats
        if strategy_trades:
            wins = [t for t in strategy_trades if t.pnl > 0]
            losses = [t for t in strategy_trades if t.pnl <= 0]
            total_pnl = sum(t.pnl for t in strategy_trades)
            win_rate = len(wins) / len(strategy_trades) * 100
            avg_hold = sum(t.hold_minutes for t in strategy_trades) / len(strategy_trades)

            # Separate by exit reason
            time_stop_trades = [t for t in strategy_trades if "TIME" in t.exit_reason.upper()]
            profit_target_trades = [t for t in strategy_trades if "PROFIT" in t.exit_reason.upper()]
            stop_loss_trades = [t for t in strategy_trades if "STOP" in t.exit_reason.upper() and "TIME" not in t.exit_reason.upper()]
            momentum_fade_trades = [t for t in strategy_trades if "MOMENTUM" in t.exit_reason.upper() or "FADE" in t.exit_reason.upper()]
            other_trades = [t for t in strategy_trades if t not in time_stop_trades + profit_target_trades + stop_loss_trades + momentum_fade_trades]

            lines.append(f"\n  SUMMARY:")
            lines.append(f"    Total Trades: {len(strategy_trades)}")
            lines.append(f"    Wins: {len(wins)} | Losses: {len(losses)}")
            lines.append(f"    Win Rate: {win_rate:.1f}%")
            lines.append(f"    Total P&L: ${total_pnl:+,.2f}")
            lines.append(f"    Avg Hold Time: {avg_hold:.1f} minutes")

            lines.append(f"\n  EXIT REASON BREAKDOWN:")
            lines.append(f"    Profit Target: {len(profit_target_trades)} trades")
            lines.append(f"    Stop Loss: {len(stop_loss_trades)} trades")
            lines.append(f"    Time Stop: {len(time_stop_trades)} trades")
            lines.append(f"    Momentum Fade: {len(momentum_fade_trades)} trades")
            lines.append(f"    Other: {len(other_trades)} trades")

            # P&L by exit reason
            lines.append(f"\n  P&L BY EXIT REASON:")
            if profit_target_trades:
                pnl = sum(t.pnl for t in profit_target_trades)
                lines.append(f"    Profit Target: ${pnl:+,.2f} ({len(profit_target_trades)} trades)")
            if stop_loss_trades:
                pnl = sum(t.pnl for t in stop_loss_trades)
                lines.append(f"    Stop Loss: ${pnl:+,.2f} ({len(stop_loss_trades)} trades)")
            if time_stop_trades:
                pnl = sum(t.pnl for t in time_stop_trades)
                lines.append(f"    Time Stop: ${pnl:+,.2f} ({len(time_stop_trades)} trades)")
            if momentum_fade_trades:
                pnl = sum(t.pnl for t in momentum_fade_trades)
                lines.append(f"    Momentum Fade: ${pnl:+,.2f} ({len(momentum_fade_trades)} trades)")

            # Win/Loss analysis
            if wins:
                avg_win = sum(t.pnl for t in wins) / len(wins)
                max_win = max(t.pnl for t in wins)
                avg_win_hold = sum(t.hold_minutes for t in wins) / len(wins)
                lines.append(f"\n  WINNING TRADES:")
                lines.append(f"    Count: {len(wins)}")
                lines.append(f"    Avg Win: ${avg_win:+,.2f}")
                lines.append(f"    Max Win: ${max_win:+,.2f}")
                lines.append(f"    Avg Hold (wins): {avg_win_hold:.1f} min")

            if losses:
                avg_loss = sum(t.pnl for t in losses) / len(losses)
                max_loss = min(t.pnl for t in losses)
                avg_loss_hold = sum(t.hold_minutes for t in losses) / len(losses)
                lines.append(f"\n  LOSING TRADES:")
                lines.append(f"    Count: {len(losses)}")
                lines.append(f"    Avg Loss: ${avg_loss:,.2f}")
                lines.append(f"    Max Loss: ${max_loss:,.2f}")
                lines.append(f"    Avg Hold (losses): {avg_loss_hold:.1f} min")

            # Time of day analysis (in EST)
            lines.append(f"\n  TIME OF DAY ANALYSIS (EST):")
            morning_trades = [t for t in strategy_trades if to_est(t.entry_time).hour < 12]
            afternoon_trades = [t for t in strategy_trades if to_est(t.entry_time).hour >= 12]

            if morning_trades:
                morning_pnl = sum(t.pnl for t in morning_trades)
                morning_wr = len([t for t in morning_trades if t.pnl > 0]) / len(morning_trades) * 100
                lines.append(f"    Morning (before 12 PM EST): {len(morning_trades)} trades, ${morning_pnl:+,.2f}, {morning_wr:.1f}% WR")

            if afternoon_trades:
                afternoon_pnl = sum(t.pnl for t in afternoon_trades)
                afternoon_wr = len([t for t in afternoon_trades if t.pnl > 0]) / len(afternoon_trades) * 100
                lines.append(f"    Afternoon (12 PM+ EST): {len(afternoon_trades)} trades, ${afternoon_pnl:+,.2f}, {afternoon_wr:.1f}% WR")

            # Day of week analysis
            lines.append(f"\n  DAY OF WEEK ANALYSIS:")
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            for day_idx, day_name in enumerate(days):
                day_trades = [t for t in strategy_trades if t.entry_time.weekday() == day_idx]
                if day_trades:
                    day_pnl = sum(t.pnl for t in day_trades)
                    day_wr = len([t for t in day_trades if t.pnl > 0]) / len(day_trades) * 100
                    lines.append(f"    {day_name}: {len(day_trades)} trades, ${day_pnl:+,.2f}, {day_wr:.1f}% WR")

            # Direction analysis (CALL vs PUT)
            lines.append(f"\n  DIRECTION ANALYSIS:")
            call_trades = [t for t in strategy_trades if t.contract_type == "CALL"]
            put_trades = [t for t in strategy_trades if t.contract_type == "PUT"]

            if call_trades:
                call_pnl = sum(t.pnl for t in call_trades)
                call_wr = len([t for t in call_trades if t.pnl > 0]) / len(call_trades) * 100
                lines.append(f"    CALLS: {len(call_trades)} trades, ${call_pnl:+,.2f}, {call_wr:.1f}% WR")

            if put_trades:
                put_pnl = sum(t.pnl for t in put_trades)
                put_wr = len([t for t in put_trades if t.pnl > 0]) / len(put_trades) * 100
                lines.append(f"    PUTS: {len(put_trades)} trades, ${put_pnl:+,.2f}, {put_wr:.1f}% WR")

            # Exhaustion score analysis (for reversal_scalper insights)
            lines.append(f"\n  EXHAUSTION SCORE ANALYSIS:")
            # Group trades by exhaustion score
            for score in range(7):  # 0-6
                score_trades = [t for t in strategy_trades if getattr(t, 'exhaustion_score_at_entry', 0) == score]
                if score_trades:
                    score_pnl = sum(t.pnl for t in score_trades)
                    score_wins = len([t for t in score_trades if t.pnl > 0])
                    score_wr = score_wins / len(score_trades) * 100
                    lines.append(f"    Score {score}: {len(score_trades)} trades, ${score_pnl:+,.2f}, {score_wr:.1f}% WR")

            # High exhaustion (score >= 3) vs low exhaustion analysis
            high_exh_trades = [t for t in strategy_trades if getattr(t, 'exhaustion_score_at_entry', 0) >= 3]
            low_exh_trades = [t for t in strategy_trades if getattr(t, 'exhaustion_score_at_entry', 0) < 3]

            if high_exh_trades:
                high_pnl = sum(t.pnl for t in high_exh_trades)
                high_wr = len([t for t in high_exh_trades if t.pnl > 0]) / len(high_exh_trades) * 100
                lines.append(f"    HIGH EXHAUSTION (>=3): {len(high_exh_trades)} trades, ${high_pnl:+,.2f}, {high_wr:.1f}% WR")

            if low_exh_trades:
                low_pnl = sum(t.pnl for t in low_exh_trades)
                low_wr = len([t for t in low_exh_trades if t.pnl > 0]) / len(low_exh_trades) * 100
                lines.append(f"    LOW EXHAUSTION (<3): {len(low_exh_trades)} trades, ${low_pnl:+,.2f}, {low_wr:.1f}% WR")

            # Session phase analysis
            lines.append(f"\n  SESSION PHASE ANALYSIS:")
            for phase in ["open_drive", "midday", "close_drive"]:
                phase_trades = [t for t in strategy_trades if getattr(t, 'session_phase_at_entry', '') == phase]
                if phase_trades:
                    phase_pnl = sum(t.pnl for t in phase_trades)
                    phase_wr = len([t for t in phase_trades if t.pnl > 0]) / len(phase_trades) * 100
                    lines.append(f"    {phase}: {len(phase_trades)} trades, ${phase_pnl:+,.2f}, {phase_wr:.1f}% WR")

            # Detailed trade log
            lines.append("\n" + "=" * 100)
            lines.append("  DETAILED TRADE LOG")
            lines.append("=" * 100)

            for i, trade in enumerate(strategy_trades, 1):
                entry_et = to_est(trade.entry_time)
                exit_et = to_est(trade.exit_time)
                lines.append(f"\n  --- TRADE #{i} ---")
                lines.append(f"  Entry Time: {entry_et.strftime('%Y-%m-%d %I:%M:%S %p EST')} ({entry_et.strftime('%A')})")
                lines.append(f"  Exit Time:  {exit_et.strftime('%Y-%m-%d %I:%M:%S %p EST') if exit_et else 'OPEN'}")
                lines.append(f"  Hold Duration: {trade.hold_minutes:.1f} minutes")
                lines.append(f"  ")
                lines.append(f"  Contract: {trade.contract_type} ${trade.strike:.0f}")
                lines.append(f"  Entry Premium: ${trade.entry_premium:.2f}")
                lines.append(f"  Exit Premium:  ${trade.exit_premium:.2f}")
                lines.append(f"  ")
                lines.append(f"  Underlying Entry: ${trade.entry_underlying:.2f}")
                lines.append(f"  Underlying Exit:  ${trade.exit_underlying:.2f}")
                lines.append(f"  Underlying Move:  ${trade.exit_underlying - trade.entry_underlying:+.2f} ({((trade.exit_underlying - trade.entry_underlying) / trade.entry_underlying) * 100:+.2f}%)")
                lines.append(f"  ")
                lines.append(f"  Greeks at Entry:")
                lines.append(f"    Delta: {trade.entry_delta:.3f}")
                lines.append(f"    Gamma: {trade.entry_gamma:.4f}")
                lines.append(f"  ")
                lines.append(f"  Market Conditions at Entry:")
                lines.append(f"    VIX: {trade.vix_at_entry:.1f}")
                lines.append(f"    Velocity: {trade.velocity_at_entry * 100:.3f}%")
                lines.append(f"    Z-Score: {trade.zscore_at_entry:.2f}")
                lines.append(f"  ")
                # Exhaustion indicators (for reversal_scalper analysis)
                rsi = getattr(trade, 'rsi_at_entry', 50.0)
                vol_ratio = getattr(trade, 'volume_ratio_at_entry', 1.0)
                cum_move = getattr(trade, 'cumulative_move_5_at_entry', 0.0)
                prior_vel = getattr(trade, 'prior_bar_velocity_at_entry', 0.0)
                vol_declining = getattr(trade, 'volume_declining_at_entry', False)
                exh_score = getattr(trade, 'exhaustion_score_at_entry', 0)
                session = getattr(trade, 'session_phase_at_entry', 'unknown')
                bars_expl = getattr(trade, 'bars_in_explosion_at_entry', 0)

                lines.append(f"  Exhaustion Indicators:")
                lines.append(f"    RSI: {rsi:.1f}")
                lines.append(f"    Volume Ratio: {vol_ratio:.1f}x")
                lines.append(f"    Cumulative Move (5 bars): {cum_move * 100:.2f}%")
                lines.append(f"    Prior Bar Velocity: {prior_vel * 100:.3f}%")
                lines.append(f"    Volume Declining: {vol_declining}")
                lines.append(f"    Exhaustion Score: {exh_score}/6")
                lines.append(f"    Session Phase: {session}")
                lines.append(f"    Bars in Explosion: {bars_expl}")
                lines.append(f"  ")
                lines.append(f"  Entry Reason: {trade.entry_reason}")
                lines.append(f"  Exit Reason:  {trade.exit_reason}")
                lines.append(f"  ")
                lines.append(f"  P&L: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.1f}%)")
                lines.append(f"  Result: {'WIN' if trade.pnl > 0 else 'LOSS'}")

        else:
            lines.append(f"\n  NO TRADES for {strategy_name} in {year}")

        lines.append("\n" + "=" * 100)

        # Write to file
        with open(log_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  [LOG] {strategy_name}/{year}.log - {len(strategy_trades)} trades")


def run_yearly_backtests():
    """Run backtest for each year and compile results."""

    # Setup log directories
    setup_log_directories()

    # Years to test
    years = [
        ("2018-01-01", "2018-12-31", "2018"),
        ("2019-01-01", "2019-12-31", "2019"),
        ("2020-01-01", "2020-12-31", "2020"),
        ("2021-01-01", "2021-12-31", "2021"),
        ("2022-01-01", "2022-12-31", "2022"),
        ("2023-01-01", "2023-12-31", "2023"),
        ("2024-01-01", "2024-12-31", "2024"),
        ("2025-01-01", "2025-12-31", "2025"),
    ]

    results = []

    print("=" * 80)
    print("  YEARLY BACKTEST RUNNER WITH DETAILED LOGGING")
    print("  Running Options Brain Backtest for 2018-2025")
    print("=" * 80)

    for start_date, end_date, year_label in years:
        print(f"\n{'='*80}")
        print(f"  RUNNING BACKTEST FOR {year_label}")
        print(f"{'='*80}")

        try:
            bt = OptionsBacktester(
                strategy_name=None,  # Brain mode
                use_real_vix=True,  # Fetch real VIX data
                vix_fallback=20.0,
                initial_capital=10000,
                position_size=1000,
            )

            result = bt.run(
                days=365,  # Ignored when start/end provided
                start_date=start_date,
                end_date=end_date,
                verbose=False,
            )

            results.append((year_label, result))
            print(f"\n  {year_label} COMPLETED: {len(result.trades)} trades, ${result.total_pnl:+,.2f} P&L")

            # Write detailed trade logs for each strategy
            write_strategy_trade_logs(year_label, result)

        except Exception as e:
            print(f"\n  ERROR for {year_label}: {e}")
            import traceback
            traceback.print_exc()
            results.append((year_label, None))

    # Generate consolidated report
    generate_report(results)

    # Generate strategy summary across all years
    generate_strategy_summary(results)


def generate_strategy_summary(results):
    """Generate a summary file for each strategy across all years."""

    strategies = ["gamma_scalper", "reversal_scalper", "vega_snap", "delta_surfer"]

    for strategy_name in strategies:
        summary_path = LOGS_DIR / strategy_name / "summary_all_years.log"

        lines = []
        lines.append("=" * 100)
        lines.append(f"  {strategy_name.upper()} - MULTI-YEAR SUMMARY")
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 100)

        all_trades = []

        for year, result in results:
            if result is None:
                continue
            strategy_trades = [t for t in result.trades if t.strategy_name == strategy_name]
            all_trades.extend(strategy_trades)

        if all_trades:
            wins = [t for t in all_trades if t.pnl > 0]
            losses = [t for t in all_trades if t.pnl <= 0]
            total_pnl = sum(t.pnl for t in all_trades)
            win_rate = len(wins) / len(all_trades) * 100
            avg_hold = sum(t.hold_minutes for t in all_trades) / len(all_trades)

            lines.append(f"\n  OVERALL STATS (2018-2025):")
            lines.append(f"    Total Trades: {len(all_trades)}")
            lines.append(f"    Wins: {len(wins)} | Losses: {len(losses)}")
            lines.append(f"    Win Rate: {win_rate:.1f}%")
            lines.append(f"    Total P&L: ${total_pnl:+,.2f}")
            lines.append(f"    Avg Hold Time: {avg_hold:.1f} minutes")

            if wins:
                avg_win = sum(t.pnl for t in wins) / len(wins)
                lines.append(f"    Avg Win: ${avg_win:+,.2f}")
            if losses:
                avg_loss = sum(t.pnl for t in losses) / len(losses)
                lines.append(f"    Avg Loss: ${avg_loss:,.2f}")

            # Year by year breakdown
            lines.append(f"\n  YEAR BY YEAR:")
            lines.append(f"  {'Year':<8} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'Win Rate':>10} {'P&L':>14}")
            lines.append("  " + "-" * 60)

            for year, result in results:
                if result is None:
                    continue
                strategy_trades = [t for t in result.trades if t.strategy_name == strategy_name]
                if strategy_trades:
                    yr_wins = len([t for t in strategy_trades if t.pnl > 0])
                    yr_losses = len([t for t in strategy_trades if t.pnl <= 0])
                    yr_pnl = sum(t.pnl for t in strategy_trades)
                    yr_wr = yr_wins / len(strategy_trades) * 100
                    lines.append(f"  {year:<8} {len(strategy_trades):>8} {yr_wins:>6} {yr_losses:>8} {yr_wr:>9.1f}% ${yr_pnl:>+12,.2f}")
                else:
                    lines.append(f"  {year:<8} {'0':>8} {'-':>6} {'-':>8} {'-':>10} {'-':>14}")

            # Exit reason analysis across all years
            lines.append(f"\n  EXIT REASON ANALYSIS (ALL YEARS):")
            time_stop_trades = [t for t in all_trades if "TIME" in t.exit_reason.upper()]
            profit_target_trades = [t for t in all_trades if "PROFIT" in t.exit_reason.upper()]
            stop_loss_trades = [t for t in all_trades if "STOP" in t.exit_reason.upper() and "TIME" not in t.exit_reason.upper()]
            momentum_fade_trades = [t for t in all_trades if "MOMENTUM" in t.exit_reason.upper() or "FADE" in t.exit_reason.upper()]

            if time_stop_trades:
                pnl = sum(t.pnl for t in time_stop_trades)
                wr = len([t for t in time_stop_trades if t.pnl > 0]) / len(time_stop_trades) * 100
                lines.append(f"    Time Stop: {len(time_stop_trades)} trades, ${pnl:+,.2f}, {wr:.1f}% WR")

            if profit_target_trades:
                pnl = sum(t.pnl for t in profit_target_trades)
                wr = len([t for t in profit_target_trades if t.pnl > 0]) / len(profit_target_trades) * 100
                lines.append(f"    Profit Target: {len(profit_target_trades)} trades, ${pnl:+,.2f}, {wr:.1f}% WR")

            if stop_loss_trades:
                pnl = sum(t.pnl for t in stop_loss_trades)
                wr = len([t for t in stop_loss_trades if t.pnl > 0]) / len(stop_loss_trades) * 100
                lines.append(f"    Stop Loss: {len(stop_loss_trades)} trades, ${pnl:+,.2f}, {wr:.1f}% WR")

            if momentum_fade_trades:
                pnl = sum(t.pnl for t in momentum_fade_trades)
                wr = len([t for t in momentum_fade_trades if t.pnl > 0]) / len(momentum_fade_trades) * 100
                lines.append(f"    Momentum Fade: {len(momentum_fade_trades)} trades, ${pnl:+,.2f}, {wr:.1f}% WR")

            # Worst trades
            lines.append(f"\n  TOP 10 WORST TRADES:")
            worst_trades = sorted(all_trades, key=lambda t: t.pnl)[:10]
            for i, trade in enumerate(worst_trades, 1):
                entry_et = to_est(trade.entry_time)
                lines.append(f"    {i}. {entry_et.strftime('%Y-%m-%d %I:%M %p EST')} | {trade.contract_type} ${trade.strike:.0f} | ${trade.pnl:,.2f} | {trade.exit_reason[:40]}")

            # Best trades
            lines.append(f"\n  TOP 10 BEST TRADES:")
            best_trades = sorted(all_trades, key=lambda t: t.pnl, reverse=True)[:10]
            for i, trade in enumerate(best_trades, 1):
                entry_et = to_est(trade.entry_time)
                lines.append(f"    {i}. {entry_et.strftime('%Y-%m-%d %I:%M %p EST')} | {trade.contract_type} ${trade.strike:.0f} | ${trade.pnl:+,.2f} | {trade.exit_reason[:40]}")

            # Pattern analysis - losing streaks
            lines.append(f"\n  LOSING STREAK ANALYSIS:")
            current_streak = 0
            max_streak = 0
            streak_start = None
            max_streak_start = None
            max_streak_end = None

            sorted_trades = sorted(all_trades, key=lambda t: t.entry_time)
            for trade in sorted_trades:
                if trade.pnl <= 0:
                    if current_streak == 0:
                        streak_start = trade.entry_time
                    current_streak += 1
                    if current_streak > max_streak:
                        max_streak = current_streak
                        max_streak_start = streak_start
                        max_streak_end = trade.entry_time
                else:
                    current_streak = 0

            lines.append(f"    Max Losing Streak: {max_streak} trades")
            if max_streak_start:
                lines.append(f"    Streak Period: {max_streak_start.strftime('%Y-%m-%d')} to {max_streak_end.strftime('%Y-%m-%d')}")

        else:
            lines.append(f"\n  NO TRADES for {strategy_name} across all years")

        lines.append("\n" + "=" * 100)

        with open(summary_path, "w") as f:
            f.write("\n".join(lines))

        print(f"  [SUMMARY] {strategy_name}/summary_all_years.log")


def generate_report(results):
    """Generate and save the consolidated report."""

    report_path = Path(__file__).parent.parent / "backtest_yearly_results.txt"

    lines = []
    lines.append("=" * 80)
    lines.append("  OPTIONS TRADING BOT - YEARLY BACKTEST RESULTS")
    lines.append("  Brain Mode: Vega Snap -> Gamma Scalper -> Delta Surfer")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)

    # Summary table
    lines.append("\n")
    lines.append("=" * 80)
    lines.append("  YEARLY SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"  {'Year':<8} {'Trades':>8} {'Win Rate':>10} {'Total P&L':>14} {'Profit Factor':>14}")
    lines.append("  " + "-" * 60)

    total_trades = 0
    total_pnl = 0
    total_wins = 0
    total_losses = 0

    for year, result in results:
        if result is None:
            lines.append(f"  {year:<8} {'ERROR':>8} {'-':>10} {'-':>14} {'-':>14}")
        else:
            trades = len(result.trades)
            wr = f"{result.win_rate:.1f}%"
            pnl = f"${result.total_pnl:+,.2f}"
            pf = f"{result.profit_factor:.2f}"
            lines.append(f"  {year:<8} {trades:>8} {wr:>10} {pnl:>14} {pf:>14}")

            total_trades += trades
            total_pnl += result.total_pnl
            total_wins += result.win_count
            total_losses += result.loss_count

    lines.append("  " + "-" * 60)
    overall_wr = f"{total_wins / (total_wins + total_losses) * 100:.1f}%" if (total_wins + total_losses) > 0 else "N/A"
    lines.append(f"  {'TOTAL':<8} {total_trades:>8} {overall_wr:>10} ${total_pnl:+,.2f}{'':>14}")

    # Per-Year Detailed Stats
    for year, result in results:
        if result is None:
            continue

        lines.append("\n")
        lines.append("=" * 80)
        lines.append(f"  {year} DETAILED RESULTS")
        lines.append("=" * 80)

        lines.append(f"\n  Period: {result.start_date.strftime('%Y-%m-%d')} to {result.end_date.strftime('%Y-%m-%d')}")
        lines.append(f"  Total Bars: {result.total_bars:,}")
        lines.append(f"  VIX Mode: {result.vix_mode.upper()}")
        if result.vix_stats:
            lines.append(f"  VIX Range: {result.vix_stats['min']:.1f} - {result.vix_stats['max']:.1f} (Mean: {result.vix_stats['mean']:.1f})")
            lines.append(f"  VIX >= 22 Days: {result.vix_stats['elevated_days']} / {result.vix_stats['total_days']}")

        # Performance
        lines.append("\n  PERFORMANCE:")
        lines.append(f"    Total Trades: {len(result.trades)}")
        lines.append(f"    Win Rate: {result.win_rate:.1f}% ({result.win_count}W / {result.loss_count}L)")
        lines.append(f"    Total P&L: ${result.total_pnl:,.2f} ({result.total_pnl_pct:+.2f}%)")
        lines.append(f"    Profit Factor: {result.profit_factor:.2f}")

        # Strategy Distribution
        lines.append("\n  STRATEGY DISTRIBUTION (Time Active):")
        total_bars = sum(result.strategy_distribution.values())
        for strat, count in result.strategy_distribution.items():
            pct = count / total_bars * 100 if total_bars > 0 else 0
            pnl = result.strategy_pnl.get(strat, 0)
            lines.append(f"    {strat:20} {pct:5.1f}% of time | P&L: ${pnl:+,.2f}")
        lines.append(f"    Strategy Switches: {result.strategy_switches}")

        # Per-Strategy Performance
        lines.append("\n  PER-STRATEGY PERFORMANCE:")
        for strat_name in ["gamma_scalper", "reversal_scalper", "vega_snap", "delta_surfer"]:
            strat_trades = [t for t in result.trades if t.strategy_name == strat_name]
            if strat_trades:
                strat_wins = len([t for t in strat_trades if t.pnl > 0])
                strat_losses = len([t for t in strat_trades if t.pnl <= 0])
                strat_pnl = sum(t.pnl for t in strat_trades)
                strat_wr = strat_wins / len(strat_trades) * 100
                avg_hold = sum(t.hold_minutes for t in strat_trades) / len(strat_trades)
                lines.append(f"    {strat_name:20}")
                lines.append(f"      Trades: {len(strat_trades)} ({strat_wins}W / {strat_losses}L)")
                lines.append(f"      Win Rate: {strat_wr:.1f}%")
                lines.append(f"      P&L: ${strat_pnl:+,.2f}")
                lines.append(f"      Avg Hold: {avg_hold:.1f} minutes")
            else:
                lines.append(f"    {strat_name:20} No trades")

        # Options Metrics
        lines.append("\n  OPTIONS METRICS:")
        lines.append(f"    Avg Hold Time: {result.avg_hold_minutes:.1f} minutes")
        lines.append(f"    Avg Leverage: {result.avg_leverage:.1f}x")
        lines.append(f"    Exit Breakdown:")
        lines.append(f"      - Profit Target: {result.profit_target_exits}")
        lines.append(f"      - Stop Loss: {result.stop_loss_exits}")
        lines.append(f"      - Time Stop: {result.time_stop_exits}")

    # Final Summary
    lines.append("\n")
    lines.append("=" * 80)
    lines.append("  OVERALL SUMMARY")
    lines.append("=" * 80)
    lines.append(f"\n  Total Years Tested: {len([r for _, r in results if r is not None])}")
    lines.append(f"  Total Trades: {total_trades}")
    lines.append(f"  Overall Win Rate: {overall_wr}")
    lines.append(f"  Total P&L: ${total_pnl:+,.2f}")
    lines.append(f"\n  Avg P&L Per Year: ${total_pnl / len([r for _, r in results if r is not None]):+,.2f}")

    profitable_years = len([r for _, r in results if r is not None and r.total_pnl > 0])
    lines.append(f"  Profitable Years: {profitable_years} / {len([r for _, r in results if r is not None])}")

    lines.append(f"\n  Detailed trade logs saved to: {LOGS_DIR}/")

    lines.append("\n" + "=" * 80)

    # Write to file
    report_text = "\n".join(lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + "=" * 80)
    print(f"  REPORT SAVED TO: {report_path}")
    print("=" * 80)
    print(report_text)


if __name__ == "__main__":
    run_yearly_backtests()
