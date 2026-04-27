"""
Day Trading Scanner — Momentum Breakout with Multi-Indicator Confirmation
Scrapes top gainers, applies strategy filters, and scores stocks for the day.

Runs every 5 minutes from 9:30–10:30 AM (market open hour), then stops.
Sends an SMS to your iPhone via Gmail → T-Mobile gateway (free, no third-party service).

Requirements:
    pip install requests beautifulsoup4 pandas yfinance ta rich

Setup:
    1. Enable 2-Step Verification on your Google account (myaccount.google.com/security)
    2. Create a Gmail App Password at myaccount.google.com/apppasswords
       → App name: "Trading Scanner" → copy the 16-char password
    3. Set these GitHub Secrets (repo Settings → Secrets → Actions):
         GMAIL_ADDRESS       yourname@gmail.com
         GMAIL_APP_PASSWORD  abcdefghijklmnop   # 16 chars, no spaces
         TMOBILE_NUMBER      2025551234          # your 10-digit T-Mobile number

Usage:
    python day_trading_scanner.py                        # uses env vars
    python day_trading_scanner.py --account 25000 --risk 1.5 --top 5
    python day_trading_scanner.py --no-sms              # disable texts (dry run)
    python day_trading_scanner.py --run-once            # single scan, no loop
"""

import argparse
import os
import sys
import time
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9+

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import ta

# ── Gmail → T-Mobile SMS config — set via GitHub Secrets ──────────────────────
GMAIL_ADDRESS      = os.environ.get("GMAIL_ADDRESS",      "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
TMOBILE_NUMBER     = os.environ.get("TMOBILE_NUMBER",     "")

# ── Timezone ──────────────────────────────────────────────────────────────────
ET = ZoneInfo("America/New_York")

# ── Scheduler config ──────────────────────────────────────────────────────────
SCAN_INTERVAL_MINS  = 5
# FIX #1: TRADING_START was dtime(9, 30) but cron fires at 9:25, causing
# immediate "outside window" exit before a single scan ran. Cron is now
# fixed to 9:30 in the yml, and we keep TRADING_START at 9:30 so they match.
TRADING_START       = dtime(9, 30)
TRADING_END         = dtime(10, 30)

# ── HTTP headers ───────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Strategy parameters ────────────────────────────────────────────────────────
MIN_GAP_PCT        = 3.0
MIN_PREMARKET_VOL  = 500_000
MIN_RVOL           = 2.0
MIN_PRICE          = 5.0
MAX_PRICE          = 30.0
RSI_LONG_MIN       = 55
RSI_LONG_MAX       = 75
RSI_SHORT_MIN      = 25
RSI_SHORT_MAX      = 45
VOLUME_MULT        = 1.5
REWARD_RISK_RATIO  = 2.0
MAX_DAILY_TRADES   = 3
MAX_DAILY_LOSS_PCT = 3.0

# force_terminal=True so Rich output is visible in GitHub Actions logs (non-TTY)
console = Console(force_terminal=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCRAPING — Top Gainers
# ══════════════════════════════════════════════════════════════════════════════

def scrape_top_gainers(top_n: int = 5) -> list[dict]:
    """
    Scrape top gainers. Tries Yahoo Finance (JS-rendered so usually fails),
    then Finviz, then falls back to yfinance screener API.
    Returns a list of dicts with ticker, name, price, change_pct, volume.
    """
    tickers = []

    # ── Primary: Yahoo Finance ────────────────────────────────────────────────
    # NOTE: Yahoo Finance is JS-rendered; BeautifulSoup will usually get 0 rows.
    # The code is kept for completeness but the fallbacks below are more reliable.
    try:
        url = "https://finance.yahoo.com/screener/predefined/day_gainers"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("table tbody tr")
        console.print(f"[dim]  Yahoo Finance: found {len(rows)} raw rows[/dim]")
        for row in rows[:top_n * 2]:
            cols = row.find_all("td")
            if len(cols) < 6:
                continue
            try:
                ticker     = cols[0].get_text(strip=True)
                name       = cols[1].get_text(strip=True)
                price      = float(cols[2].get_text(strip=True).replace(",", ""))
                change_pct = float(cols[4].get_text(strip=True).replace("%", ""))
                volume_raw = cols[5].get_text(strip=True).replace(",", "")
                volume     = _parse_volume(volume_raw)
                tickers.append({
                    "ticker":     ticker,
                    "name":       name,
                    "price":      price,
                    "change_pct": change_pct,
                    "volume":     volume,
                    "source":     "Yahoo Finance",
                })
            except (ValueError, IndexError):
                continue

        if tickers:
            console.print(f"[green]✓[/green] Scraped {len(tickers)} gainers from Yahoo Finance")
            return tickers[:top_n]
        else:
            console.print("[yellow]⚠ Yahoo Finance returned 0 parseable rows (JS-rendered page) — trying Finviz[/yellow]")

    except Exception as e:
        console.print(f"[yellow]⚠ Yahoo Finance scrape failed: {e}[/yellow]")

    # ── Fallback: Finviz ──────────────────────────────────────────────────────
    try:
        url = (
            "https://finviz.com/screener.ashx?v=111&s=ta_topgainers"
            "&f=sh_price_o5,sh_price_u500,sh_relvol_o2&o=-change"
        )
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        rows = soup.select("table.screener-table tr.table-light-row-cp, "
                           "table.screener-table tr.table-dark-row-cp")
        console.print(f"[dim]  Finviz: found {len(rows)} raw rows[/dim]")
        for row in rows[:top_n * 2]:
            cols = row.find_all("td")
            if len(cols) < 10:
                continue
            try:
                ticker     = cols[1].get_text(strip=True)
                price      = float(cols[8].get_text(strip=True))
                change_pct = float(cols[9].get_text(strip=True).replace("%", ""))
                volume     = _parse_volume(cols[10].get_text(strip=True))
                tickers.append({
                    "ticker":     ticker,
                    "name":       ticker,
                    "price":      price,
                    "change_pct": change_pct,
                    "volume":     volume,
                    "source":     "Finviz",
                })
            except (ValueError, IndexError):
                continue

        if tickers:
            console.print(f"[green]✓[/green] Scraped {len(tickers)} gainers from Finviz")
            return tickers[:top_n]

    except Exception as e:
        console.print(f"[yellow]⚠ Finviz scrape failed: {e}[/yellow]")

    # ── Fallback 2: yfinance built-in screener ────────────────────────────────
    # FIX #5: The old fallback used a hardcoded seed list of mega-caps (NVDA,
    # AAPL, TSLA…) which never meet MIN_PRICE<=30 + MIN_GAP_PCT>=3% + MIN_RVOL>=2x
    # simultaneously, meaning evaluate_stock() always returned entry_valid=False
    # and no SMS was ever sent from this path.
    # Replaced with yfinance's built-in screener (requires yfinance >= 0.2.38).
    try:
        console.print("[cyan]  Trying yfinance built-in screener…[/cyan]")
        from yfinance import screen  # noqa: F401 — available in yfinance >= 0.2.38

        # day_gainers screen returns stocks sorted by % change, already filtered
        # for meaningful volume by Yahoo's backend.
        screener_result = yf.screen("day_gainers")
        quotes = screener_result.get("quotes", [])

        records = []
        for q in quotes:
            sym   = q.get("symbol", "")
            price = q.get("regularMarketPrice", 0) or 0
            chg   = q.get("regularMarketChangePercent", 0) or 0
            vol   = q.get("regularMarketVolume", 0) or 0
            name  = q.get("shortName", sym)
            if price > 0 and sym:
                records.append({
                    "ticker":     sym,
                    "name":       name,
                    "price":      round(price, 2),
                    "change_pct": round(chg, 2),
                    "volume":     int(vol),
                    "source":     "yfinance-screener",
                })

        # Sort by % change descending (screener may already do this)
        records.sort(key=lambda x: x["change_pct"], reverse=True)
        if records:
            console.print(f"[green]✓[/green] yfinance screener returned {len(records)} tickers")
            return records[:top_n]

    except Exception as e:
        console.print(f"[yellow]⚠ yfinance screener failed: {e}[/yellow]")

    console.print("[red]✗ All data sources failed. Check internet connection / API access.[/red]")
    return []


def _parse_volume(raw: str) -> int:
    """Convert strings like '1.23M', '500K', '123456' to integer."""
    raw = raw.strip().upper().replace(",", "")
    if raw.endswith("B"):
        return int(float(raw[:-1]) * 1_000_000_000)
    if raw.endswith("M"):
        return int(float(raw[:-1]) * 1_000_000)
    if raw.endswith("K"):
        return int(float(raw[:-1]) * 1_000)
    try:
        return int(float(raw))
    except ValueError:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA FETCHING — OHLCV + Indicators via yfinance
# ══════════════════════════════════════════════════════════════════════════════

def fetch_intraday_data(ticker: str) -> pd.DataFrame | None:
    """
    Download 5-minute OHLCV data for today.
    Returns a DataFrame with added indicator columns, or None on error.
    """
    try:
        df = yf.download(ticker, period="2d", interval="5m",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 20:
            return None

        # FIX #8: yfinance >=0.2.x returns a MultiIndex (field, ticker).
        # Flatten safely: take the first level (field name) for single-ticker downloads.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        # Rename any variant column names that yfinance uses
        df = df.rename(columns={"stock splits": "splits", "capital gains": "capgains"})

        # ── Indicators ──────────────────────────────────────────────────────
        df["ema9"]  = ta.trend.ema_indicator(df["close"], window=9)
        df["ema20"] = ta.trend.ema_indicator(df["close"], window=20)
        df["rsi"]   = ta.momentum.rsi(df["close"], window=14)

        # VWAP (session-level)
        df["vwap"] = _compute_vwap(df)

        # Relative candle volume (vs session avg so far)
        df["avg_vol"]  = df["volume"].expanding().mean()
        df["rel_cvol"] = df["volume"] / df["avg_vol"].shift(1).fillna(1)

        return df

    except Exception as e:
        console.print(f"  [red]✗ Data fetch failed for {ticker}: {e}[/red]")
        return None


def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Intraday VWAP reset each session day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    date_groups = df.index.date
    vwap = pd.Series(index=df.index, dtype=float)
    for day in set(date_groups):
        mask = date_groups == day
        tp_day  = tp[mask]
        vol_day = df["volume"][mask]
        cum_tpv = (tp_day * vol_day).cumsum()
        cum_vol = vol_day.cumsum()
        vwap[mask] = cum_tpv / cum_vol
    return vwap


def fetch_30d_avg_volume(ticker: str) -> float:
    """Return 30-day average daily volume for RVOL calculation."""
    try:
        hist = yf.Ticker(ticker).history(period="30d")
        if hist.empty:
            return 1.0
        return hist["Volume"].mean()
    except Exception:
        return 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. STRATEGY SCORING — Phase 1 & Phase 2 Filters
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_stock(stock: dict, df: pd.DataFrame, avg_30d_vol: float) -> dict:
    """
    Apply strategy rules and return a scored evaluation dict.
    score 0-100 represents confidence / alignment strength.
    """
    result = {
        "ticker":         stock["ticker"],
        "name":           stock.get("name", stock["ticker"]),
        "price":          stock["price"],
        "change_pct":     stock["change_pct"],
        "direction":      "LONG" if stock["change_pct"] > 0 else "SHORT",
        "score":          0,
        "signals":        [],
        "red_flags":      [],
        "entry_valid":    False,
        "stop_price":     None,
        "target_price":   None,
        "suggested_shares": None,
    }

    price    = stock["price"]
    gap_pct  = abs(stock["change_pct"])
    volume   = stock["volume"]
    is_long  = result["direction"] == "LONG"

    # ── Phase 1: Pre-Market Filters ──────────────────────────────────────────
    score = 0

    if gap_pct >= MIN_GAP_PCT:
        score += 20
        result["signals"].append(f"Gap {gap_pct:.1f}% ✓")
    else:
        result["red_flags"].append(f"Gap {gap_pct:.1f}% < {MIN_GAP_PCT}% ✗")

    if volume >= MIN_PREMARKET_VOL:
        score += 10
        result["signals"].append(f"Vol {_fmt_vol(volume)} ✓")
    else:
        result["red_flags"].append(f"Low pre-market vol ({_fmt_vol(volume)}) ✗")

    rvol = volume / avg_30d_vol if avg_30d_vol else 0
    result["rvol"] = rvol
    if rvol >= MIN_RVOL:
        score += 15
        result["signals"].append(f"RVOL {rvol:.1f}x ✓")
    else:
        result["red_flags"].append(f"RVOL {rvol:.1f}x < {MIN_RVOL}x ✗")

    if MIN_PRICE <= price <= MAX_PRICE:
        score += 10
        result["signals"].append(f"Price ${price:.2f} in range ✓")
    else:
        result["red_flags"].append(f"Price ${price:.2f} out of range ✗")

    # ── Phase 2: Indicator Signals (last completed candle) ───────────────────
    if df is None or len(df) < 2:
        result["score"] = score
        return result

    last = df.iloc[-1]
    prev = df.iloc[-2]  # noqa: F841  (available for future use)

    # VWAP
    above_vwap = last["close"] > last["vwap"]
    if (is_long and above_vwap) or (not is_long and not above_vwap):
        score += 15
        result["signals"].append(
            f"VWAP {'above' if above_vwap else 'below'} (${last['vwap']:.2f}) ✓"
        )
    else:
        result["red_flags"].append(
            f"Price {'below' if is_long else 'above'} VWAP ✗"
        )

    # EMA crossover
    ema_bullish = last["ema9"] > last["ema20"]
    if (is_long and ema_bullish) or (not is_long and not ema_bullish):
        score += 15
        result["signals"].append(
            f"9 EMA {'>' if ema_bullish else '<'} 20 EMA ✓"
        )
    else:
        result["red_flags"].append("EMA crossover not aligned ✗")

    # Price above/below both EMAs
    above_emas = last["close"] > last["ema9"] and last["close"] > last["ema20"]
    if (is_long and above_emas) or (not is_long and not above_emas):
        score += 5
        result["signals"].append("Price above both EMAs ✓" if is_long else "Price below both EMAs ✓")

    # RSI
    rsi = last["rsi"]
    result["rsi"] = round(rsi, 1) if not pd.isna(rsi) else None
    if is_long:
        if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
            score += 10
            result["signals"].append(f"RSI {rsi:.0f} in momentum zone ✓")
        elif rsi > 80:
            result["red_flags"].append(f"RSI {rsi:.0f} overbought — no chase ✗")
        else:
            result["red_flags"].append(f"RSI {rsi:.0f} not in 55-75 zone ✗")
    else:
        if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
            score += 10
            result["signals"].append(f"RSI {rsi:.0f} in short zone ✓")
        elif rsi < 20:
            result["red_flags"].append(f"RSI {rsi:.0f} oversold — no chase ✗")
        else:
            result["red_flags"].append(f"RSI {rsi:.0f} not in 25-45 zone ✗")

    # FIX #6: Guard against NaN rel_cvol before formatting and phase2 check.
    # On the first candle of a session, avg_vol.shift(1) is NaN → rel_cvol is NaN.
    # pd.isna(NaN) >= VOLUME_MULT evaluates False which silently fails the check
    # and f"{NaN:.1f}" prints "nan" in the red flag message, which is confusing.
    raw_rel_cvol = last["rel_cvol"]
    rel_cvol = float(raw_rel_cvol) if not pd.isna(raw_rel_cvol) else 0.0
    if rel_cvol >= VOLUME_MULT:
        score += 5
        result["signals"].append(f"Candle vol {rel_cvol:.1f}x avg ✓")
    else:
        result["red_flags"].append(
            f"Candle vol {rel_cvol:.1f}x avg (need {VOLUME_MULT}x) ✗"
        )

    # ── Entry Validity ───────────────────────────────────────────────────────
    # Entry is valid if 4 of 5 Phase 2 indicators align
    phase2_checks = [
        (is_long and above_vwap) or (not is_long and not above_vwap),
        (is_long and ema_bullish) or (not is_long and not ema_bullish),
        (is_long and above_emas) or (not is_long and not above_emas),
        (is_long and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX) or
        (not is_long and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX),
        rel_cvol >= VOLUME_MULT,   # uses the NaN-safe value
    ]
    result["entry_valid"] = sum(phase2_checks) >= 4

    # ── Stop & Target Calculation ────────────────────────────────────────────
    candle_low  = df["low"].iloc[-3:].min()
    candle_high = df["high"].iloc[-3:].max()

    if is_long:
        stop   = round(candle_low * 0.998, 2)
        stop   = min(stop, float(last["vwap"]) - 0.01)
        target = round(price + (price - stop) * REWARD_RISK_RATIO, 2)
    else:
        stop   = round(candle_high * 1.002, 2)
        target = round(price - (stop - price) * REWARD_RISK_RATIO, 2)

    result["stop_price"]   = stop
    result["target_price"] = target
    result["score"]        = min(score, 100)

    return result


def compute_position_size(price: float, stop: float,
                           account: float, risk_pct: float) -> dict:
    """Phase 3: Position sizing from strategy formula."""
    dollar_risk   = account * (risk_pct / 100)
    stop_distance = abs(price - stop)
    if stop_distance == 0:
        return {"shares": 0, "dollar_risk": dollar_risk, "position_value": 0,
                "stop_distance": 0}
    shares = int(dollar_risk / stop_distance)
    return {
        "shares":         shares,
        "dollar_risk":    round(dollar_risk, 2),
        "stop_distance":  round(stop_distance, 2),
        "position_value": round(shares * price, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. OUTPUT — Rich Terminal Display
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_vol(v: int | float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(int(v))


def print_header():
    console.print(Panel(
        "[bold cyan]Day Trading Scanner[/bold cyan]  •  "
        "Momentum Breakout + Multi-Indicator Confirmation\n"
        f"[dim]{datetime.now(ET).strftime('%A %B %d, %Y  %H:%M:%S')} ET[/dim]",
        box=box.DOUBLE_EDGE, expand=False
    ))


def print_market_timing():
    now = datetime.now(ET).time()
    premarket  = dtime(8, 0) <= now < dtime(9, 30)
    prime_open = dtime(9, 30) <= now < dtime(10, 30)
    lunch_chop = dtime(11, 30) <= now < dtime(13, 30)
    close_zone = dtime(15, 45) <= now

    if premarket:
        console.print("[bold green]⏰ PRE-MARKET PREP WINDOW (8:00–9:30)[/bold green] — Build your watchlist now")
    elif prime_open:
        console.print("[bold green]🔔 PRIME ENTRY WINDOW (9:30–10:30)[/bold green] — Best breakout opportunities")
    elif lunch_chop:
        console.print("[bold yellow]⚠  LUNCH CHOP ZONE (11:30–1:30)[/bold yellow] — Low volume, AVOID new entries")
    elif close_zone:
        console.print("[bold red]🛑 CLOSE ZONE (3:45+ PM)[/bold red] — CLOSE ALL POSITIONS, no new entries")
    else:
        console.print("[cyan]📈 Mid-session active[/cyan] — Monitor open positions")
    console.print()


def print_summary_table(results: list[dict]):
    table = Table(title="Top Gainers — Strategy Scorecard",
                  box=box.ROUNDED, show_lines=True)
    table.add_column("Rank",       style="dim",        width=4)
    table.add_column("Ticker",     style="bold white",  width=8)
    table.add_column("Price",      justify="right",     width=8)
    table.add_column("Gap %",      justify="right",     width=7)
    table.add_column("RVOL",       justify="right",     width=6)
    table.add_column("RSI",        justify="right",     width=5)
    table.add_column("Dir",        justify="center",    width=6)
    table.add_column("Score",      justify="center",    width=7)
    table.add_column("Entry?",     justify="center",    width=8)
    table.add_column("Stop",       justify="right",     width=8)
    table.add_column("Target",     justify="right",     width=8)

    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(sorted_results, 1):
        score = r["score"]
        score_color = (
            "bright_green" if score >= 75 else
            "yellow"        if score >= 50 else
            "red"
        )
        entry_str  = "[bright_green]✓ YES[/bright_green]" if r["entry_valid"] else "[red]✗ NO[/red]"
        dir_color  = "bright_green" if r["direction"] == "LONG" else "red"
        gap_color  = "bright_green" if abs(r["change_pct"]) >= MIN_GAP_PCT else "dim"

        table.add_row(
            str(i),
            r["ticker"],
            f"${r['price']:.2f}",
            f"[{gap_color}]{r['change_pct']:+.1f}%[/{gap_color}]",
            f"{r.get('rvol', 0):.1f}x",
            f"{r.get('rsi', '—')}",
            f"[{dir_color}]{r['direction']}[/{dir_color}]",
            f"[{score_color}]{score}[/{score_color}]",
            entry_str,
            f"${r['stop_price']}" if r["stop_price"] else "—",
            f"${r['target_price']}" if r["target_price"] else "—",
        )

    console.print(table)


def print_detailed_card(r: dict, account: float, risk_pct: float):
    direction_color = "bright_green" if r["direction"] == "LONG" else "red"
    score = r["score"]
    score_label = (
        "[bright_green]HIGH CONFIDENCE[/bright_green]" if score >= 75 else
        "[yellow]MODERATE[/yellow]"                    if score >= 50 else
        "[red]LOW / SKIP[/red]"
    )

    lines = [
        f"[bold]{r['ticker']}[/bold]  {r['name']}   "
        f"[{direction_color}]{r['direction']}[/{direction_color}]  "
        f"Score: [bold]{score}/100[/bold]  {score_label}",
        f"Price: [bold]${r['price']:.2f}[/bold]   "
        f"Gap: [bold]{r['change_pct']:+.1f}%[/bold]   "
        f"RVOL: [bold]{r.get('rvol', 0):.1f}x[/bold]   "
        f"RSI: [bold]{r.get('rsi', '—')}[/bold]",
        "",
    ]

    if r["signals"]:
        lines.append("[green]Signals:[/green]")
        for s in r["signals"]:
            lines.append(f"  [green]•[/green] {s}")

    if r["red_flags"]:
        lines.append("[red]Red Flags:[/red]")
        for fl in r["red_flags"]:
            lines.append(f"  [red]•[/red] {fl}")

    if r["entry_valid"] and r["stop_price"]:
        sizing = compute_position_size(
            r["price"], r["stop_price"], account, risk_pct
        )
        lines += [
            "",
            "[bold cyan]─── Position Sizing ─────────────────────────────[/bold cyan]",
            f"  Account: ${account:,.0f}   Risk: {risk_pct}% = ${sizing['dollar_risk']:,.0f}",
            f"  Stop distance: ${sizing['stop_distance']:.2f}",
            f"  [bold]Shares to buy: {sizing['shares']}[/bold]  "
            f"(Position value: ${sizing['position_value']:,.0f})",
            f"  Stop:  [red]${r['stop_price']}[/red]",
            f"  T1 (50% off at 2:1): [green]${r['target_price']}[/green]",
            f"  T2: Trail 9 EMA on 5-min chart after T1",
        ]

    border_color = "bright_green" if r["entry_valid"] else "dim"
    console.print(Panel(
        "\n".join(lines),
        border_style=border_color,
        expand=False
    ))


def print_kill_switches(account: float, losses_today: int, down_pct: float):
    console.print("\n[bold]Daily Kill Switches[/bold]")
    checks = [
        (losses_today >= 3,   f"Consecutive losses: {losses_today}/3 — STOP TRADING"),
        (down_pct   >= MAX_DAILY_LOSS_PCT,
                              f"Down {down_pct:.1f}% today (limit {MAX_DAILY_LOSS_PCT}%) — WALK AWAY"),
    ]
    for triggered, msg in checks:
        icon = "[bold red]✗ TRIGGERED[/bold red]" if triggered else "[green]✓ OK[/green]"
        console.print(f"  {icon}  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMS — Gmail → T-Mobile gateway
# ══════════════════════════════════════════════════════════════════════════════

def send_sms(message: str, dry_run: bool = False) -> bool:
    """
    Send an SMS via Gmail → T-Mobile email gateway (number@tmomail.net).
    Returns True on success. If dry_run=True, prints instead of sending.
    """
    import smtplib
    from email.message import EmailMessage

    if dry_run:
        console.print(Panel(
            f"[bold yellow]📱 DRY RUN — SMS not sent[/bold yellow]\n\n{message}",
            border_style="yellow", expand=False
        ))
        return True

    missing = [k for k, v in {
        "GMAIL_ADDRESS":      GMAIL_ADDRESS,
        "GMAIL_APP_PASSWORD": GMAIL_APP_PASSWORD,
        "TMOBILE_NUMBER":     TMOBILE_NUMBER,
    }.items() if not v]

    if missing:
        console.print(
            f"[red]✗ SMS skipped — missing env vars: {', '.join(missing)}[/red]\n"
            "[dim]  Set them as GitHub Secrets or pass --no-sms to suppress this warning.[/dim]"
        )
        return False

    number_clean = "".join(c for c in TMOBILE_NUMBER if c.isdigit())
    if len(number_clean) == 11 and number_clean.startswith("1"):
        number_clean = number_clean[1:]
    to_gateway = f"{number_clean}@tmomail.net"

    try:
        em = EmailMessage()
        em["From"]    = GMAIL_ADDRESS
        em["To"]      = to_gateway
        em["Subject"] = ""
        em.set_content(message)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(em)

        console.print(f"[green]✓ SMS sent → {to_gateway}[/green]")
        return True

    except smtplib.SMTPAuthenticationError:
        console.print(
            "[red]✗ Gmail authentication failed.[/red]\n"
            "[dim]  Make sure you're using a Gmail App Password, not your regular password.\n"
            "  Generate one at: myaccount.google.com/apppasswords[/dim]"
        )
        return False
    except Exception as e:
        console.print(f"[red]✗ SMS failed: {e}[/red]")
        return False


def build_sms(scan_num: int, top: dict | None,
              valid_entries: list[dict], account: float, risk_pct: float) -> str:
    """Build a concise SMS message for the top pick."""
    now_str = datetime.now(ET).strftime("%H:%M")

    if not top:
        return (
            f"📊 Trading Scanner [{now_str}] Scan #{scan_num}\n"
            "No valid entries found this scan.\n"
            "All 4 indicators not yet aligned — waiting."
        )

    sizing = compute_position_size(top["price"], top["stop_price"], account, risk_pct)
    others = [r["ticker"] for r in valid_entries[1:3]]
    others_str = f"\nAlso valid: {', '.join(others)}" if others else ""

    return (
        f"🎯 TOP PICK: {top['ticker']} [{now_str}] Scan #{scan_num}\n"
        f"Dir: {top['direction']}  Score: {top['score']}/100\n"
        f"Entry: ~${top['price']:.2f}\n"
        f"Stop:  ${top['stop_price']}\n"
        f"T1:    ${top['target_price']} (2:1 R:R)\n"
        f"Shares: {sizing['shares']}  Risk: ${sizing['dollar_risk']:.0f}\n"
        f"Gap: {top['change_pct']:+.1f}%  RSI: {top.get('rsi','—')}  "
        f"RVOL: {top.get('rvol',0):.1f}x"
        f"{others_str}\n"
        "⚠ Educational only. Not financial advice."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 6. SCHEDULER — 5-min loop, 9:30–10:30 AM ET
# ══════════════════════════════════════════════════════════════════════════════

def is_trading_window() -> bool:
    """Return True if current ET time is within 9:30–10:30 AM."""
    now = datetime.now(ET).time()
    return TRADING_START <= now < TRADING_END


def run_scheduler(account: float, risk_pct: float, top_n: int,
                  losses: int, down_pct: float, dry_run: bool):
    """
    Loop: scan every 5 minutes between 9:30 and 10:30 AM ET, then exit.
    Sends an SMS after each scan if a valid entry is found.

    FIX #1 + #2: Removed the pre-open wait sleep. Cron now fires exactly at
    9:30 AM ET so there is no gap between job start and window open. Removing
    the sleep also eliminates the race condition where a long sleep could push
    past 10:30 AM before the first scan ran.
    """
    if not is_trading_window():
        now_str = datetime.now(ET).strftime("%H:%M")
        msg = (
            f"⚠ Trading Scanner [{now_str}]\n"
            "Outside the 9:30–10:30 AM ET window — no scan run.\n"
            "Check cron schedule or use --run-once to test outside hours."
        )
        console.print(f"[yellow]{msg}[/yellow]")
        # FIX #3: was hardcoded dry_run=False, now correctly respects the flag
        if not dry_run:
            send_sms(msg, dry_run=False)
        return

    scan_num   = 0
    texted     = set()   # tickers already texted this session
    # FIX #7: track whether we've sent a "no pick" SMS this session.
    # We only send one at the start and one at the end to avoid SMS spam.
    no_pick_sms_sent = False

    console.print(Panel(
        f"[bold green]🟢 SCANNER STARTED[/bold green]\n"
        f"Running every {SCAN_INTERVAL_MINS} minutes  •  "
        f"Window: 9:30 – 10:30 AM ET\n"
        f"SMS {'DISABLED (dry run)' if dry_run else 'ENABLED → ' + (TMOBILE_NUMBER or 'number not set')}",
        border_style="green", expand=False
    ))
    console.print()

    # Startup heartbeat — confirms the job fired and SMS credentials work
    if not dry_run:
        now_str = datetime.now(ET).strftime("%H:%M")
        send_sms(
            f"🟢 Trading Scanner started [{now_str}]\n"
            f"Scanning every {SCAN_INTERVAL_MINS} min until 10:30 AM ET.\n"
            "Will text picks as found.",
            dry_run=False
        )

    while is_trading_window():
        scan_num += 1
        scan_start = datetime.now(ET)

        console.rule(
            f"[bold cyan]SCAN #{scan_num}  —  "
            f"{scan_start.strftime('%H:%M:%S')} ET[/bold cyan]"
        )

        gainers = scrape_top_gainers(top_n)
        results = []

        if gainers:
            for stock in gainers:
                ticker  = stock["ticker"]
                avg_vol = fetch_30d_avg_volume(ticker)
                df      = fetch_intraday_data(ticker)
                eval_r  = evaluate_stock(stock, df, avg_vol)
                results.append(eval_r)
                time.sleep(0.3)

            print_summary_table(results)
            console.print()

            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            valid_entries  = [r for r in sorted_results if r["entry_valid"]]
            top            = valid_entries[0] if valid_entries else None

            if top and top["ticker"] not in texted:
                sms_body = build_sms(scan_num, top, valid_entries, account, risk_pct)
                sent = send_sms(sms_body, dry_run=dry_run)
                if sent:
                    texted.add(top["ticker"])
                    console.print(
                        f"[green]📱 Texted alert for {top['ticker']}[/green]"
                    )
            elif top and top["ticker"] in texted:
                console.print(
                    f"[dim]⏭  {top['ticker']} already texted this session — skipping SMS[/dim]"
                )
            else:
                # FIX #7: No valid pick. Send one SMS at the start of the session
                # so you know the scanner is running but hasn't found anything yet.
                # After that, stay quiet until end-of-session summary to avoid spam.
                if not no_pick_sms_sent:
                    send_sms(
                        build_sms(scan_num, None, [], account, risk_pct),
                        dry_run=dry_run
                    )
                    no_pick_sms_sent = True
                else:
                    console.print(
                        f"[dim]  Scan #{scan_num}: no valid pick — holding SMS (already notified)[/dim]"
                    )

        else:
            console.print("[red]No gainers returned — all data sources may be blocked.[/red]")
            now_str = datetime.now(ET).strftime("%H:%M")
            send_sms(
                f"⚠ Trading Scanner [{now_str}] Scan #{scan_num}\n"
                "All data sources returned 0 gainers.\n"
                "Check GitHub Actions logs.",
                dry_run=dry_run
            )

        print_kill_switches(account, losses, down_pct)

        if not is_trading_window():
            break

        elapsed  = (datetime.now(ET) - scan_start).total_seconds()
        sleep_s  = max(0, SCAN_INTERVAL_MINS * 60 - elapsed)
        next_at  = datetime.now(ET) + timedelta(seconds=sleep_s)

        if next_at.time() >= TRADING_END:
            console.print(
                f"\n[dim]Next scan at {next_at.strftime('%H:%M')} ET "
                f"would be after 10:30 AM — stopping now.[/dim]"
            )
            break

        console.print(
            f"\n[dim]Next scan in {sleep_s/60:.1f} min  "
            f"({next_at.strftime('%H:%M:%S')} ET)[/dim]\n"
        )
        time.sleep(sleep_s)

    # ── Session complete ─────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold yellow]🏁 SCANNER STOPPED — First-hour window closed[/bold yellow]\n"
        f"Completed {scan_num} scan{'s' if scan_num != 1 else ''}  •  "
        f"Tickers texted: {', '.join(texted) if texted else 'none'}\n\n"
        "[dim]Strategy: avoid 11:30 AM–1:30 PM lunch chop. "
        "Close all positions by 3:45 PM.[/dim]",
        border_style="yellow"
    ))

    # Session-complete heartbeat
    if not dry_run:
        now_str = datetime.now(ET).strftime("%H:%M")
        send_sms(
            f"🏁 Scanner done [{now_str}] — {scan_num} scan{'s' if scan_num != 1 else ''} completed.\n"
            f"Picks texted: {', '.join(texted) if texted else 'none'}.",
            dry_run=False
        )


# ══════════════════════════════════════════════════════════════════════════════
# 7. CLI ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Day Trading Scanner — Momentum Breakout, 9:30–10:30 AM ET loop"
    )
    parser.add_argument("--account",  type=float, default=10_000,
                        help="Account size in USD (default: 10000)")
    parser.add_argument("--risk",     type=float, default=1.0,
                        help="Risk per trade as %% of account (default: 1.0)")
    parser.add_argument("--top",      type=int,   default=5,
                        help="Number of top gainers to scan (default: 5)")
    parser.add_argument("--losses",   type=int,   default=0,
                        help="Consecutive losses today (for kill switch)")
    parser.add_argument("--down",     type=float, default=0.0,
                        help="%% account drawdown today (for kill switch)")
    parser.add_argument("--no-sms",   action="store_true",
                        help="Disable SMS — print message to terminal instead")
    parser.add_argument("--run-once", action="store_true",
                        help="Run a single scan now (ignore time window check)")
    args = parser.parse_args()

    if args.run_once:
        # Single scan — useful for testing outside market hours
        console.print("[cyan]▶ Running single scan (--run-once)[/cyan]\n")
        gainers = scrape_top_gainers(args.top)
        results = []
        for stock in gainers:
            avg_vol = fetch_30d_avg_volume(stock["ticker"])
            df      = fetch_intraday_data(stock["ticker"])
            eval_r  = evaluate_stock(stock, df, avg_vol)
            results.append(eval_r)
            time.sleep(0.3)

        if results:
            print_summary_table(results)
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            for r in sorted_results:
                print_detailed_card(r, args.account, args.risk)
                console.print()

            valid  = [r for r in sorted_results if r["entry_valid"]]
            top    = valid[0] if valid else None
            sms    = build_sms(1, top, valid, args.account, args.risk)
            send_sms(sms, dry_run=args.no_sms)

        print_kill_switches(args.account, args.losses, args.down)
    else:
        run_scheduler(
            account  = args.account,
            risk_pct = args.risk,
            top_n    = args.top,
            losses   = args.losses,
            down_pct = args.down,
            dry_run  = args.no_sms,
        )


if __name__ == "__main__":
    main()
