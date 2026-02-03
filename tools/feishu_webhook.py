#!/usr/bin/env python3
import os, sys, json, urllib.request
import yaml
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
USERS_FILE = CONFIG_DIR / "users.yaml"

def load_user_webhook(user_id: str) -> str:
    """Load feishu webhook from user config."""
    if not user_id or not USERS_FILE.exists():
        return ""
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    user = config.get("users", {}).get(user_id)
    return user.get("feishu_webhook", "") if user else ""

def main():
    title = sys.argv[1] if len(sys.argv) > 1 else "Experiment finished"
    text  = sys.argv[2] if len(sys.argv) > 2 else ""
    user_id = sys.argv[3] if len(sys.argv) > 3 else None

    # Try user config first, then env var
    webhook = load_user_webhook(user_id) if user_id else ""
    if not webhook:
        webhook = os.getenv("FEISHU_WEBHOOK_URL", "")
    if not webhook:
        return 0

    payload = {
        "msg_type": "text",
        "content": {"text": f"{title}\n{text}"}
    }
    req = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
