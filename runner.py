"""
Single-shot live runner.

Called by GitHub Actions at approximately:
    09:46 IST
    04:16 UTC

The runner:
    1. verifies weekday
    2. verifies the intended execution window
    3. runs scanner.run_live_scan() exactly once
    4. exits

It does NOT loop.
"""

from __future__ import annotations

import sys

import pandas as pd

from scanner import (
    EXECUTION_TIME,
    MARKET_TZ,
    run_live_scan,
)


def main() -> int:

    now = pd.Timestamp.now(
        tz=MARKET_TZ
    )

    print(
        f"Runner time: "
        f"{now:%Y-%m-%d %H:%M:%S %Z}"
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
    # GitHub Actions cron can be delayed. We therefore allow
    # execution from 09:46 through 09:55 IST rather than blindly
    # executing at any arbitrary time.
    # ------------------------------------------------------------

    execution_start = pd.Timestamp(
        f"{now:%Y-%m-%d} 09:46:00",
        tz=MARKET_TZ,
    )

    execution_end = pd.Timestamp(
        f"{now:%Y-%m-%d} 09:55:00",
        tz=MARKET_TZ,
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
            "09:46-09:55 IST"
        )

        return 0

    print(
        "Executing ONE live market scan..."
    )

    try:

        run_live_scan()

    except Exception as exc:

        print(
            f"Scanner failed: {exc}"
        )

        return 1

    print(
        "Live scan completed."
    )

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
