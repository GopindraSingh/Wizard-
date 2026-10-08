# MORE12 Telegram Scanner — Upstox + GitHub Actions

This repository runs the **current-day opening 15-minute scan only**. It does not prompt for historical dates or run the rolling Mode 3 scan.

- Scan candle: 09:15–09:30 IST
- Scheduled run: 09:35 IST, Monday–Friday
- Data: Upstox V3 current-day intraday candles
- Delivery: Telegram Bot API
- Hosting: GitHub Actions; your computer does not need to stay on

GitHub's scheduler can start scheduled jobs a little late. The script also guards against running before 09:31 IST. The schedule is set to 09:35 IST to allow the opening candle to finish and be available via the API.

## 1. Create a Telegram bot

1. Open Telegram and start a chat with **@BotFather**.
2. Send `/newbot`, follow the prompts, and copy the bot token.
3. Open your new bot and send `/start`.
4. Get your chat ID by opening this URL in a browser, replacing `<BOT_TOKEN>` with your token:

   `https://api.telegram.org/bot<BOT_TOKEN>/getUpdates`

   Look for `message.chat.id`. Treat your bot token like a password. Do not commit it to the repository.

## 2. Create an Upstox Analytics Token

For an unattended market-data-only job, use an **Analytics Token** rather than a standard access token. Upstox documents Analytics Tokens as read-only, valid for one year, and supporting Historical Data APIs. This scan does not place orders.

1. Open the [Upstox Developer Apps](https://account.upstox.com/developer/apps) page.
2. Open the app's Analytics tab and generate an Analytics Token.
3. Copy the full token. Do not paste it into source code.

Documentation: [Upstox Analytics Token](https://upstox.com/developer/api-documentation/analytics-token/).

## 3. Create the GitHub repository and upload these files

Create a repository on GitHub and upload the contents of this folder, preserving this structure:

```text
.
├── more12_telegram.py
├── requirements.txt
├── README.md
├── .gitignore
└── .github/
    └── workflows/
        └── more12-telegram.yml
```

Make sure the workflow file is on your repository's **default branch**. Scheduled workflows run from the default branch.

## 4. Add GitHub Actions secrets

Go to **Repository → Settings → Secrets and variables → Actions → New repository secret** and create these three secrets:

| Secret name | Value |
|---|---|
| `UPSTOX_ACCESS_TOKEN` | Your Upstox Analytics Token |
| `TELEGRAM_BOT_TOKEN` | Token from @BotFather |
| `TELEGRAM_CHAT_ID` | Your Telegram numeric chat ID |

Never put actual credentials in `README.md`, Python files, or the workflow YAML.

## 5. Enable and test the workflow

1. Open the repository's **Actions** tab and enable Actions if GitHub asks.
2. Select **MORE12 Telegram Opening Scanner**.
3. Use **Run workflow** to test it after 09:31 IST on an NSE trading day.
4. On weekdays, GitHub schedules the run for **09:35 IST** automatically.

On NSE holidays, the job may find no current-day candles. It reports this separately from a genuine zero-signal scan. If a scheduled run does not occur, confirm the workflow is on the default branch and Actions are enabled.

## Local test (optional)

Install dependencies and set the three environment variables before running:

```bash
python -m pip install -r requirements.txt
export UPSTOX_ACCESS_TOKEN='your_analytics_token'
export TELEGRAM_BOT_TOKEN='your_bot_token'
export TELEGRAM_CHAT_ID='your_chat_id'
python more12_telegram.py
```

The script deliberately skips weekends and refuses to scan before 09:31 IST.

## Strategy

The existing MORE12 opening-candle thresholds remain unchanged: Tier 1 uses a 0.5% candle move and ₹4 Cr turnover; Tier 2 uses a 1.5% candle move and ₹15 Cr turnover, combined with the existing candle close-location filters. This is a scanner, not an order-execution bot.
