#!/usr/bin/env python3
"""
Scraper Module - The "Eyes" of the Bi-Cameral Trading Bot

Gathers market intelligence from external sources:
- Fear & Greed Index
- News Headlines
- Market Sentiment

All functions are fault-tolerant and return neutral defaults on error.
This module must NEVER crash the parent process.

Author: Bi-Cameral System
"""

import asyncio
import logging
from typing import Optional

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Browser, Page

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stealth configuration
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Default timeout (ms)
TIMEOUT = 15000


async def setup_browser() -> tuple[Browser, "async_playwright"]:
    """
    Initialize a headless browser with stealth configuration.

    Returns:
        Tuple of (browser instance, playwright context manager)
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ]
    )
    return browser, playwright


async def create_stealth_page(browser: Browser) -> Page:
    """
    Create a new page with stealth settings.
    """
    context = await browser.new_context(
        user_agent=USER_AGENT,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
    )
    page = await context.new_page()

    # Additional stealth: override navigator.webdriver
    await page.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
    """)

    return page


async def get_fear_and_greed() -> int:
    """
    Fetch the Fear & Greed Index value.

    Returns:
        Integer 0-100 (0=Extreme Fear, 100=Extreme Greed)
        Returns 50 (neutral) on any error.
    """
    import aiohttp

    # Try multiple sources for reliability
    sources = [
        # Alternative.me API (crypto but correlates with market sentiment)
        ("https://api.alternative.me/fng/", "api"),
        # CNN Fear & Greed (scraping fallback)
        ("https://www.cnn.com/markets/fear-and-greed", "scrape"),
    ]

    # Method 1: Try Alternative.me API first (fast, reliable)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(sources[0][0], timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    value = int(data["data"][0]["value"])
                    logger.info(f"[SCRAPER] Fear & Greed Index (API): {value}")
                    return max(0, min(100, value))
    except Exception as e:
        logger.warning(f"[SCRAPER] Alternative.me API failed: {e}")

    # Method 2: Scrape CNN as fallback
    try:
        browser, playwright = await setup_browser()

        try:
            page = await create_stealth_page(browser)

            await page.goto(
                sources[1][0],
                timeout=TIMEOUT,
                wait_until="domcontentloaded"
            )

            # Try multiple selectors (CNN changes their structure often)
            selectors = [
                ".market-fng-gauge__dial-number",
                "[class*='FearGreedIndex'] [class*='value']",
                "[data-testid='fear-greed-value']",
                ".fear-greed-index__value",
            ]

            for selector in selectors:
                try:
                    await page.wait_for_selector(selector, timeout=5000)
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        # Extract digits from the text
                        import re
                        match = re.search(r'\d+', text)
                        if match:
                            value = int(match.group())
                            logger.info(f"[SCRAPER] Fear & Greed Index (CNN): {value}")
                            return max(0, min(100, value))
                except Exception:
                    continue

        finally:
            await browser.close()
            await playwright.stop()

    except Exception as e:
        logger.warning(f"[SCRAPER] CNN scrape failed: {e}")

    # Return neutral on any error
    logger.info("[SCRAPER] Returning default Fear & Greed: 50")
    return 50


async def get_news_headlines(symbol: str = "SPY") -> list[str]:
    """
    Fetch top news headlines for a given symbol from Yahoo Finance.

    Args:
        symbol: Stock ticker symbol (default: SPY)

    Returns:
        List of headline strings (max 5).
        Returns empty list on any error.
    """
    headlines = []

    try:
        browser, playwright = await setup_browser()

        try:
            page = await create_stealth_page(browser)

            # Navigate to Yahoo Finance news page
            url = f"https://finance.yahoo.com/quote/{symbol}/news/"
            await page.goto(url, timeout=TIMEOUT, wait_until="domcontentloaded")

            # Wait for news items to load
            await asyncio.sleep(2)  # Allow dynamic content to render

            # Get page content and parse with BeautifulSoup
            content = await page.content()
            soup = BeautifulSoup(content, "html.parser")

            # Find headline elements (Yahoo Finance structure)
            headline_elements = soup.select("h3.clamp")
            if not headline_elements:
                # Fallback selectors
                headline_elements = soup.select("a[data-testid='article-link'] h3")
            if not headline_elements:
                headline_elements = soup.select(".news-stream h3")

            for elem in headline_elements[:5]:
                text = elem.get_text(strip=True)
                if text and len(text) > 10:
                    headlines.append(text)

            logger.info(f"[SCRAPER] Found {len(headlines)} headlines for {symbol}")

        finally:
            await browser.close()
            await playwright.stop()

    except Exception as e:
        logger.warning(f"[SCRAPER] News fetch failed for {symbol}: {e}")

    return headlines


async def get_market_sentiment() -> dict:
    """
    Aggregate market sentiment from multiple sources.

    Returns:
        Dictionary with sentiment data:
        {
            "fear_greed": int (0-100),
            "headlines": list[str],
            "overall": str ("bullish", "bearish", "neutral")
        }
    """
    # Gather data concurrently
    fear_greed, headlines = await asyncio.gather(
        get_fear_and_greed(),
        get_news_headlines(),
        return_exceptions=True
    )

    # Handle any exceptions from gather
    if isinstance(fear_greed, Exception):
        logger.warning(f"[SCRAPER] Fear & Greed exception: {fear_greed}")
        fear_greed = 50
    if isinstance(headlines, Exception):
        logger.warning(f"[SCRAPER] Headlines exception: {headlines}")
        headlines = []

    # Determine overall sentiment
    if fear_greed >= 70:
        overall = "bullish"
    elif fear_greed <= 30:
        overall = "bearish"
    else:
        overall = "neutral"

    return {
        "fear_greed": fear_greed,
        "headlines": headlines,
        "overall": overall
    }


def get_market_internals() -> dict:
    """
    Fetch VIX (Volatility) and TNX (10-Year Treasury Yield).

    These are CRITICAL for TQQQ trading:
    - VIX > 25: High volatility = dangerous for leveraged ETFs
    - VIX > 30: Extreme volatility = DISABLE buying
    - Rising TNX: Tech stocks suffer when yields spike

    Returns:
        Dictionary with current values and % changes.
    """
    import yfinance as yf

    try:
        # Fetch 5 days of data to calculate trend
        vix = yf.Ticker("^VIX")
        tnx = yf.Ticker("^TNX")

        vix_hist = vix.history(period="5d")
        tnx_hist = tnx.history(period="5d")

        if vix_hist.empty or tnx_hist.empty:
            logger.warning("[SCRAPER] VIX/TNX data unavailable")
            return {
                "vix": 20.0,
                "vix_change_pct": 0.0,
                "tnx": 4.0,
                "tnx_change_pct": 0.0,
                "vix_alert": "unknown",
                "tnx_alert": "unknown"
            }

        # Calculate current values and changes
        vix_now = float(vix_hist['Close'].iloc[-1])
        vix_prev = float(vix_hist['Close'].iloc[-2])
        vix_chg = ((vix_now - vix_prev) / vix_prev) * 100 if vix_prev > 0 else 0

        tnx_now = float(tnx_hist['Close'].iloc[-1])
        tnx_prev = float(tnx_hist['Close'].iloc[-2])
        tnx_chg = ((tnx_now - tnx_prev) / tnx_prev) * 100 if tnx_prev > 0 else 0

        # Determine alert levels
        if vix_now > 30:
            vix_alert = "EXTREME"
        elif vix_now > 25:
            vix_alert = "HIGH"
        elif vix_now > 20:
            vix_alert = "ELEVATED"
        else:
            vix_alert = "NORMAL"

        # TNX alert based on rate of change
        if tnx_chg > 3:
            tnx_alert = "SPIKING"
        elif tnx_chg > 1:
            tnx_alert = "RISING"
        elif tnx_chg < -1:
            tnx_alert = "FALLING"
        else:
            tnx_alert = "STABLE"

        result = {
            "vix": round(vix_now, 2),
            "vix_change_pct": round(vix_chg, 2),
            "tnx": round(tnx_now, 2),
            "tnx_change_pct": round(tnx_chg, 2),
            "vix_alert": vix_alert,
            "tnx_alert": tnx_alert
        }

        logger.info(f"[SCRAPER] VIX: {result['vix']} ({result['vix_alert']}), "
                   f"TNX: {result['tnx']}% ({result['tnx_alert']})")

        return result

    except Exception as e:
        logger.warning(f"[SCRAPER] Market internals fetch failed: {e}")
        return {
            "vix": 20.0,
            "vix_change_pct": 0.0,
            "tnx": 4.0,
            "tnx_change_pct": 0.0,
            "vix_alert": "unknown",
            "tnx_alert": "unknown"
        }


# Convenience function for synchronous usage
def get_market_sentiment_sync() -> dict:
    """
    Synchronous wrapper for get_market_sentiment().
    """
    return asyncio.run(get_market_sentiment())


if __name__ == "__main__":
    # Test the scraper
    async def test():
        print("=" * 60)
        print("  SCRAPER TEST - Bi-Cameral Trading Bot")
        print("=" * 60)

        print("\n[TEST] Fetching Fear & Greed Index...")
        fg = await get_fear_and_greed()
        print(f"  Result: {fg}")

        print("\n[TEST] Fetching News Headlines...")
        news = await get_news_headlines("SPY")
        for i, headline in enumerate(news, 1):
            print(f"  {i}. {headline[:60]}...")

        print("\n[TEST] Full Market Sentiment...")
        sentiment = await get_market_sentiment()
        print(f"  Fear & Greed: {sentiment['fear_greed']}")
        print(f"  Overall: {sentiment['overall']}")
        print(f"  Headlines: {len(sentiment['headlines'])}")

        print("\n" + "=" * 60)

    asyncio.run(test())
