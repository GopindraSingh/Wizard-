import subprocess
import requests

TOKEN = "8984037851:AAGnc5Tm088pqdilp8kL-I5giUXylP8hRQQ"
CHAT_ID = "1860594381"


def run_scanner_and_notify():
  process = subprocess.run(
      ["python", "scanner.py"], capture_output=True, text=True, encoding="utf-8"
  )
  scan_output = process.stdout
  print(scan_output)

  if process.returncode != 0:
    scan_output = f"❌ Script failed:\n{process.stderr[:1000]}"

  if len(scan_output) > 4000:
    scan_output = scan_output[:3950] + "\n\n... [Output truncated]"

  message = f"```text\n{scan_output}\n```"
  url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
  payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}

  response = requests.post(url, json=payload)
  if response.status_code == 200:
    print("Telegram notification sent successfully!")
  else:
    print(f"Failed to send Telegram message: {response.text}")


if __name__ == "__main__":
  run_scanner_and_notify()
