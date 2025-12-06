"""
Shared Utilities - High-Performance Indicator Library

Professional-grade technical indicators using pandas vectorization.
All functions are designed for efficiency with large datasets.

Indicators:
    - ADX: Average Directional Index (Trend Strength)
    - ATR: Average True Range (Volatility)
    - Z-Score: Statistical Deviation from Mean
    - Donchian: Channel High/Low Breakout Levels
    - VWAP: Volume Weighted Average Price
    - RSI: Relative Strength Index
    - EMA: Exponential Moving Average
    - Bollinger Bands: Volatility Bands

Author: Bi-Cameral Quant Team
"""

from typing import Tuple
import numpy as np
import pandas as pd


# =============================================================================
# TREND INDICATORS
# =============================================================================

def calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average Directional Index (ADX) - Trend Strength.

    ADX measures trend strength regardless of direction:
    - ADX < 20: Weak/No trend (CHOP)
    - ADX 20-25: Developing trend (BUFFER)
    - ADX 25-50: Strong trend (TREND)
    - ADX > 50: Very strong trend (EXTREME)

    Uses Wilder's smoothing method for authentic calculation.

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Smoothing period (default 14)

    Returns:
        Series with ADX values (0-100)
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    # Wilder's smoothing (EMA with alpha=1/period)
    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx


def calc_plus_minus_di(df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate +DI and -DI for directional trading.

    +DI > -DI = Bullish pressure
    -DI > +DI = Bearish pressure

    Args:
        df: DataFrame with OHLC data
        period: Smoothing period

    Returns:
        Tuple of (+DI, -DI) Series
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    return plus_di, minus_di


# =============================================================================
# VOLATILITY INDICATORS
# =============================================================================

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) - Volatility Measure.

    ATR is used for:
    - Dynamic stop losses (e.g., 1.5 * ATR)
    - Position sizing (higher ATR = smaller position)
    - Breakout confirmation (price move > ATR is significant)

    Args:
        df: DataFrame with 'High', 'Low', 'Close' columns
        period: Smoothing period (default 14)

    Returns:
        Series with ATR values in price units
    """
    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's smoothing
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


def calc_atr_percent(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR as a percentage of price.

    Useful for comparing volatility across different price levels.
    ATR% of 2% means typical range is 2% of price.

    Args:
        df: DataFrame with OHLC data
        period: ATR period

    Returns:
        Series with ATR as percentage (0-100)
    """
    atr = calc_atr(df, period)
    atr_pct = (atr / df["Close"]) * 100
    return atr_pct


# =============================================================================
# STATISTICAL INDICATORS
# =============================================================================

def calc_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Z-Score - Statistical Deviation from Mean.

    Z-Score measures how many standard deviations price is from its mean:
    - Z < -2: Statistically oversold (2 sigma event)
    - Z < -1: Oversold
    - Z = 0: At mean
    - Z > 1: Overbought
    - Z > 2: Statistically overbought (2 sigma event)

    Args:
        series: Price series (typically Close)
        window: Rolling window for mean/std calculation

    Returns:
        Series with Z-Score values
    """
    mean = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    zscore = (series - mean) / (std + 1e-10)

    return zscore


def calc_zscore_volume(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Volume Z-Score for spike detection.

    Used to identify unusual volume:
    - Z > 2: Volume spike (institutional activity)
    - Z > 3: Extreme volume (major event)

    Args:
        df: DataFrame with 'Volume' column
        window: Rolling window

    Returns:
        Series with Volume Z-Score
    """
    return calc_zscore(df["Volume"], window)


# =============================================================================
# CHANNEL INDICATORS
# =============================================================================

def calc_donchian(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Donchian Channels - Breakout Levels.

    Donchian Channels show:
    - Upper: Highest high over N periods (resistance breakout)
    - Lower: Lowest low over N periods (support breakdown)
    - Middle: Midpoint (equilibrium)

    Used in Turtle Trading and breakout strategies.

    Args:
        df: DataFrame with 'High', 'Low' columns
        window: Lookback period (default 20)

    Returns:
        Tuple of (upper, middle, lower) Series
    """
    upper = df["High"].rolling(window=window).max()
    lower = df["Low"].rolling(window=window).min()
    middle = (upper + lower) / 2

    return upper, middle, lower


def calc_bollinger_bands(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands - Volatility Bands.

    Bands contract in low volatility, expand in high volatility.
    Price touching bands is NOT a signal - context matters.

    Args:
        series: Price series (typically Close)
        window: SMA period
        num_std: Number of standard deviations

    Returns:
        Tuple of (upper, middle, lower) Series
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()

    upper = middle + (std * num_std)
    lower = middle - (std * num_std)

    return upper, middle, lower


def calc_keltner_channels(
    df: pd.DataFrame,
    ema_period: int = 20,
    atr_period: int = 14,
    atr_mult: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Keltner Channels - ATR-based Volatility Bands.

    Unlike Bollinger (std-based), Keltner uses ATR for smoother bands.
    Squeeze: Bollinger inside Keltner = low volatility, breakout coming.

    Args:
        df: DataFrame with OHLC data
        ema_period: EMA period for middle line
        atr_period: ATR calculation period
        atr_mult: ATR multiplier for bands

    Returns:
        Tuple of (upper, middle, lower) Series
    """
    middle = df["Close"].ewm(span=ema_period, adjust=False).mean()
    atr = calc_atr(df, atr_period)

    upper = middle + (atr * atr_mult)
    lower = middle - (atr * atr_mult)

    return upper, middle, lower


# =============================================================================
# VOLUME INDICATORS
# =============================================================================

def calc_vwap(df: pd.DataFrame, reset_daily: bool = True) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP).

    VWAP is the "fair value" benchmark used by institutions:
    - Price > VWAP: Bullish bias
    - Price < VWAP: Bearish bias

    Args:
        df: DataFrame with 'High', 'Low', 'Close', 'Volume' columns
        reset_daily: If True, VWAP resets at market open each day

    Returns:
        Series with VWAP values
    """
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    tp_volume = typical_price * df["Volume"]

    if reset_daily:
        # Group by date for daily reset
        date = df.index.date
        cumulative_tp_vol = tp_volume.groupby(date).cumsum()
        cumulative_vol = df["Volume"].groupby(date).cumsum()
    else:
        cumulative_tp_vol = tp_volume.cumsum()
        cumulative_vol = df["Volume"].cumsum()

    vwap = cumulative_tp_vol / (cumulative_vol + 1e-10)

    return vwap


def calc_volume_sma(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Volume Simple Moving Average.

    Used for volume spike detection:
    - Volume / SMA > 2 = Volume spike
    - Volume / SMA > 3 = Major volume event

    Args:
        df: DataFrame with 'Volume' column
        window: SMA period

    Returns:
        Series with Volume SMA
    """
    return df["Volume"].rolling(window=window).mean()


def calc_volume_ratio(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Calculate Volume Ratio (Current / SMA).

    Args:
        df: DataFrame with Volume
        window: SMA window

    Returns:
        Series with volume ratio (1.0 = average, 2.0 = 2x average)
    """
    vol_sma = calc_volume_sma(df, window)
    return df["Volume"] / (vol_sma + 1e-10)


# =============================================================================
# MOMENTUM INDICATORS
# =============================================================================

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).

    RSI measures momentum on a 0-100 scale:
    - RSI < 30: Oversold (potential bounce)
    - RSI 30-70: Neutral
    - RSI > 70: Overbought (potential pullback)

    Uses Wilder's smoothing for authentic calculation.

    Args:
        series: Price series (typically Close)
        period: RSI period (default 14)

    Returns:
        Series with RSI values (0-100)
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # Wilder's smoothing
    alpha = 1 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).

    EMA reacts faster to recent price than SMA.
    Used for trend direction and dynamic support/resistance.

    Args:
        series: Price series
        span: EMA period

    Returns:
        Series with EMA values
    """
    return series.ewm(span=span, adjust=False).mean()


def calc_sma(series: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).

    Args:
        series: Price series
        window: SMA period

    Returns:
        Series with SMA values
    """
    return series.rolling(window=window).mean()


def calc_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD measures momentum through EMA convergence/divergence:
    - MACD > Signal: Bullish momentum
    - MACD < Signal: Bearish momentum
    - Histogram: Visual momentum strength

    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


# =============================================================================
# SPECIALIZED INDICATORS
# =============================================================================

def calc_opening_range(
    df: pd.DataFrame,
    minutes: int = 30
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Opening Range for ORB Strategy.

    The Opening Range is the High/Low of the first N minutes of trading.
    Breakout above = bullish, breakdown below = bearish.

    Args:
        df: DataFrame with OHLC data (must have datetime index)
        minutes: Minutes after open to calculate range (default 30)

    Returns:
        Tuple of (orb_high, orb_low) Series
    """
    # Get the date for each row
    df = df.copy()
    df["date"] = df.index.date
    df["time"] = df.index.time

    # Market open time
    from datetime import time
    market_open = time(9, 30)
    orb_end = time(10, 0) if minutes == 30 else time(9, 30 + minutes // 60, minutes % 60)

    # Filter to opening range period
    orb_mask = (df["time"] >= market_open) & (df["time"] < orb_end)

    # Calculate high/low for opening range by date
    orb_data = df[orb_mask].groupby("date").agg({"High": "max", "Low": "min"})
    orb_data.columns = ["orb_high", "orb_low"]

    # Map back to full dataframe
    df["orb_high"] = df["date"].map(orb_data["orb_high"])
    df["orb_low"] = df["date"].map(orb_data["orb_low"])

    return df["orb_high"], df["orb_low"]


def calc_pivot_points(df: pd.DataFrame) -> dict:
    """
    Calculate Daily Pivot Points.

    Pivot points are key support/resistance levels used by floor traders.

    Args:
        df: DataFrame with previous day's OHLC

    Returns:
        Dict with pivot, r1, r2, r3, s1, s2, s3
    """
    high = df["High"].iloc[-1]
    low = df["Low"].iloc[-1]
    close = df["Close"].iloc[-1]

    pivot = (high + low + close) / 3

    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)

    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    return {
        "pivot": pivot,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3
    }


# =============================================================================
# REGIME DETECTION
# =============================================================================

def detect_market_regime(df: pd.DataFrame, adx_period: int = 14) -> pd.Series:
    """
    Detect Market Regime based on ADX.

    Returns:
        Series with regime labels: 'TREND', 'CHOP', or 'BUFFER'
    """
    adx = calc_adx(df, adx_period)

    conditions = [
        adx >= 25,
        adx < 20,
    ]
    choices = ["TREND", "CHOP"]

    regime = pd.Series(
        np.select(conditions, choices, default="BUFFER"),
        index=df.index
    )

    return regime


def calc_squeeze_indicator(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate TTM Squeeze Indicator.

    Squeeze occurs when Bollinger Bands are inside Keltner Channels.
    This indicates low volatility and an imminent breakout.

    Returns:
        Tuple of (squeeze_on: bool Series, momentum: float Series)
    """
    bb_upper, bb_mid, bb_lower = calc_bollinger_bands(df["Close"], 20, 2.0)
    kc_upper, kc_mid, kc_lower = calc_keltner_channels(df, 20, 14, 1.5)

    # Squeeze is on when BB is inside KC
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # Momentum (simplified - using linear regression would be more accurate)
    momentum = df["Close"] - calc_sma(df["Close"], 20)

    return squeeze_on, momentum
