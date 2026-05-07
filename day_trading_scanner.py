"""
Day Trading Scanner — Top Gainers with SMS Alert
Runs once at market open via GitHub Actions and texts you the top gainers.

Setup (GitHub Secrets — repo Settings → Secrets → Actions):
    GMAIL_ADDRESS       yourname@gmail.com
    GMAIL_APP_PASSWORD  abcdefghijklmnop   (16-char app password, no spaces)
    TMOBILE_NUMBER      2025551234          (10-digit T-Mobile number)

Usage:
    python day_trading_scanner.py                  # normal run
    python day_trading_scanner.py --top 10         # more stocks
    python day_trading_scanner.py --no-sms         # dry run, no texts sent
"""

import argparse
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from zoneinfo import ZoneInfo

import yfinance as yf

# ── Config from GitHub Secrets ─────────────────────────────────────────────────
GMAIL_ADDRESS      = os.environ.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
TMOBILE_NUMBER     = os.environ.get("TMOBILE_NUMBER", "")

ET = ZoneInfo("America/New_York")

# ── Strategy filters ───────────────────────────────────────────────────────────
MIN_PRICE     = 2.0    # minimum stock price
MAX_PRICE     = 50.0   # maximum stock price
MIN_CHANGE    = 3.0    # minimum % gain to be considered


def get_top_gainers(top_n: int) -> list[dict]:
    """Fetch top gainers from Yahoo Finance via yfinance screener."""
    print("Fetching top gainers...")
    try:
        result = yf.screen("day_gainers")
        quotes = result.get("quotes", [])
        if not quotes:
            print("WARNING: yfinance screener returned no results.")
            return []

        stocks = []
        for q in quotes:
            sym    = q.get("symbol", "")
            price  = q.get("regularMarketPrice") or 0
            change = q.get("regularMarketChangePercent") or 0
            volume = q.get("regularMarketVolume") or 0
            name   = q.get("shortName", sym)

            if not sym or price <= 0:
                continue
            if not (MIN_PRICE <= price <= MAX_PRICE):
                continue
            if change < MIN_CHANGE:
                continue

            stocks.append({
                "ticker":  sym,
                "name":    name,
                "price":   round(price, 2),
                "change":  round(change, 2),
                "volume":  volume,
            })

        stocks.sort(key=lambda x: x["change"], reverse=True)
        print(f"Found {len(stocks)} qualifying stocks (after filters).")
        return stocks[:top_n]

    except Exception as e:
        print(f"ERROR fetching gainers: {e}")
        return []


def fmt_volume(v: int) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return str(v)


def build_message(stocks: list[dict], top_n: int) -> str:
    """Build the SMS / email body."""
    now = datetime.now(ET).strftime("%I:%M %p ET")
    if not stocks:
        return f"Trading Scanner [{now}]\nNo stocks passed filters today."

    lines = [f"Trading Scanner [{now}] — Top {len(stocks)} Gainers"]
    for i, s in enumerate(stocks, 1):
        lines.append(
            f"{i}. {s['ticker']} +{s['change']:.1f}% "
            f"@ ${s['price']:.2f}  Vol:{fmt_volume(s['volume'])}"
        )
    lines.append("Educational only. Not financial advice.")
    return "\n".join(lines)


def send_sms(message: str, dry_run: bool = False) -> bool:
    """Send message as SMS via Gmail → T-Mobile email gateway."""
    if dry_run:
        print("\n--- DRY RUN (no SMS sent) ---")
        print(message)
        print("---")
        return True

    # Validate secrets
    missing = [k for k, v in {
        "GMAIL_ADDRESS": GMAIL_ADDRESS,
        "GMAIL_APP_PASSWORD": GMAIL_APP_PASSWORD,
        "TMOBILE_NUMBER": TMOBILE_NUMBER,
    }.items() if not v]
    if missing:
        print(f"ERROR: Missing secrets: {', '.join(missing)}")
        return False

    number = "".join(c for c in TMOBILE_NUMBER if c.isdigit())
    if number.startswith("1") and len(number) == 11:
        number = number[1:]
    if len(number) != 10:
        print(f"ERROR: TMOBILE_NUMBER must be 10 digits, got: {number!r}")
        return False

    to_addr = f"{number}@tmomail.net"
    try:
        em = EmailMessage()
        em["From"]    = GMAIL_ADDRESS
        em["To"]      = to_addr
        em["Subject"] = "Trading Scanner"
        em.set_content(message)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
            smtp.send_message(em)

        print(f"SMS sent to {to_addr}")
        return True

    except smtplib.SMTPAuthenticationError:
        print("ERROR: Gmail authentication failed. Check GMAIL_ADDRESS and GMAIL_APP_PASSWORD.")
        return False
    except Exception as e:
        print(f"ERROR sending SMS: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Day Trading Scanner")
    parser.add_argument("--top",    type=int,  default=5,     help="Number of top gainers (default: 5)")
    parser.add_argument("--no-sms", action="store_true",      help="Dry run — print SMS instead of sending")
    args = parser.parse_args()

    print(f"=== Trading Scanner | {datetime.now(ET).strftime('%Y-%m-%d %H:%M ET')} ===\n")

    stocks  = get_top_gainers(args.top)
    message = build_message(stocks, args.top)

    print("\n--- Message to send ---")
    print(message)
    print("-----------------------\n")

    send_sms(message, dry_run=args.no_sms)


if __name__ == "__main__":
    main()
