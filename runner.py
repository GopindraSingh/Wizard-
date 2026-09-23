"""
Single-shot live runner.

Called by GitHub Actions at approximately:
    09:46 IST
    04:16 UTC

The runner:
    1. verifies weekday
    2. verifies the intended execution window
    3. validates configuration
    4. runs scanner.run_live_scan() exactly once
    5. prints full traceback on failure
    6. exits

It does NOT loop.
"""

from __future__ import annotations

import os
import sys
import traceback

import pandas as pd

from scanner import (
    EXECUTION_TIME,
    MARKET_TZ,
    run_live_scan,
)


# ================================================================
# CONFIGURATION
# ================================================================

EXECUTION_WINDOW_END = "09:55:00"


# ================================================================
# MAIN
# ================================================================

def main() -> int:

    now = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    print(
        f"Runner time: "
        f"{now:%Y-%m-%d %H:%M:%S %Z}"
    )

    # ------------------------------------------------------------
    # Environment diagnostics
    # ------------------------------------------------------------

    print(
        "Python environment:"
    )

    print(
        f"TELEGRAM_BOT_TOKEN configured: "
        f"{bool(os.getenv('8984037851:AAGnc5Tm088pqdilp8kL-I5giUXylP8hRQQ'))}"
    )

    print(
        f"TELEGRAM_CHAT_ID configured: "
        f"{bool(os.getenv('1860594381'))}"
    )

    # ------------------------------------------------------------
    # Monday-Friday only.
    # ------------------------------------------------------------

    if now.weekday() >= 5:

        print(
            "Weekend. Scanner will not execute."
        )

        return 0

    # ------------------------------------------------------------
    # Intended execution window.
    #
    # GitHub Actions cron can be delayed.
    #
    # We therefore allow:
    #
    # 09:46:00
    # through
    # 09:55:00
    #
    # IST.
    # ------------------------------------------------------------

    date_string = (
        now.strftime(
            "%Y-%m-%d"
        )
    )

    execution_start = pd.Timestamp(
        f"{date_string} "
        f"{EXECUTION_TIME}:00",
        tz=MARKET_TZ,
    )

    execution_end = pd.Timestamp(
        f"{date_string} "
        f"{EXECUTION_WINDOW_END}",
        tz=MARKET_TZ,
    )

    print(
        f"Allowed execution window: "
        f"{execution_start:%H:%M:%S} - "
        f"{execution_end:%H:%M:%S} IST"
    )

    if not (
        execution_start
        <= now
        <= execution_end
    ):

        print(
            "Outside live execution window."
        )

        print(
            "Expected: "
            f"{EXECUTION_TIME}:00-"
            f"{EXECUTION_WINDOW_END} IST"
        )

        return 0

    # ------------------------------------------------------------
    # Execute scanner once.
    # ------------------------------------------------------------

    print(
        "Executing ONE live market scan..."
    )

    try:

        run_live_scan()

    except Exception as exc:

        print()
        print(
            "========================================"
        )
        print(
            "SCANNER FAILED"
        )
        print(
            "========================================"
        )

        print(
            f"Error type: {type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        print()
        print(
            "Full traceback:"
        )

        traceback.print_exc()

        print(
            "========================================"
        )

        return 1

    print(
        "Live scan completed successfully."
    )

    return 0


# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )
