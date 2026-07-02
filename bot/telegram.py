"""Minimal Telegram Bot API notifier.

Credentials come from TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID (see
.env.example) -- never hardcode them in source. When they're missing, or
dry_run is forced, no network call is made: the message is printed
instead. That's what lets this be exercised in sandboxes without outbound
access to api.telegram.org.
"""
import os

import requests

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramNotifier:
    def __init__(self, token: str = None, chat_id: str = None, dry_run: bool = False):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID")
        self.dry_run = dry_run or not (self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if self.dry_run:
            print("[telegram:dry-run] would send:\n" + text)
            return True
        url = TELEGRAM_API.format(token=self.token)
        resp = requests.post(
            url,
            data={"chat_id": self.chat_id, "text": text},
            timeout=10,
        )
        resp.raise_for_status()
        return bool(resp.json().get("ok"))
