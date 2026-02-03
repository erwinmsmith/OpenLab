#!/usr/bin/env python3
import os, sys, ssl, mimetypes, smtplib
import yaml
from email.message import EmailMessage
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
USERS_FILE = CONFIG_DIR / "users.yaml"

def getenv(name: str, default: str = None, required: bool = False) -> str:
    v = os.getenv(name, default)
    if required and (v is None or v.strip() == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def load_user_config(user_id: str):
    """Load user config from users.yaml."""
    if not USERS_FILE.exists():
        return None
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    user = config.get("users", {}).get(user_id)
    if user:
        user["_default_receiver"] = config.get("default_receiver", "openex@code-soul.com")
    return user

def attach_file(msg: EmailMessage, path: Path, max_bytes: int) -> bool:
    if not path.exists():
        return False
    size = path.stat().st_size
    if size > max_bytes:
        return False
    ctype, encoding = mimetypes.guess_type(str(path))
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)
    with open(path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype=maintype,
            subtype=subtype,
            filename=path.name
        )
    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: notify_email.py <run_dir> <subject> [user_id]", file=sys.stderr)
        sys.exit(2)

    run_dir = Path(sys.argv[1]).resolve()
    subject = sys.argv[2]
    user_id = sys.argv[3] if len(sys.argv) > 3 else None

    # Try to load user config first, fallback to env vars
    user_config = load_user_config(user_id) if user_id else None
    
    if user_config:
        smtp_cfg = user_config.get("smtp", {})
        smtp_host = smtp_cfg.get("host", "")
        smtp_port = int(smtp_cfg.get("port", 587))
        smtp_user = smtp_cfg.get("user", "")
        smtp_pass = smtp_cfg.get("pass", "")
        smtp_from = user_config.get("email", "")
        smtp_to = user_config.get("_default_receiver", "openex@code-soul.com")
        if not smtp_host or not smtp_from:
            print(f"Error: User '{user_id}' has incomplete SMTP config", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback to environment variables
        smtp_host = getenv("SMTP_HOST", required=True)
        smtp_port = int(getenv("SMTP_PORT", "587"))
        smtp_user = getenv("SMTP_USER", "")
        smtp_pass = getenv("SMTP_PASS", "")
        smtp_from = getenv("SMTP_FROM", required=True)
        smtp_to = getenv("SMTP_TO", "openex@code-soul.com")

    # Attachment policy
    max_attach_mb = int(getenv("MAX_ATTACH_MB", "20"))
    max_attach_bytes = max_attach_mb * 1024 * 1024

    # Optional: if attachments too big, include a link
    download_url = getenv("RUN_DOWNLOAD_URL", "")  # e.g. https://yourhost/runs/<run_id>/

    artifacts = run_dir / "artifacts"
    summary_md = artifacts / "summary.md"
    report_pdf = artifacts / "report.pdf"
    run_zip = artifacts / "run.zip"

    body_lines = []
    body_lines.append(f"Run: {run_dir.name}")
    body_lines.append("")
    if summary_md.exists():
        body_lines.append("=== Summary ===")
        body_lines.append(summary_md.read_text(encoding="utf-8", errors="ignore"))
        body_lines.append("")
    else:
        body_lines.append("(No summary.md found.)")
        body_lines.append("")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg.set_content("\n".join(body_lines))

    attached_any = False
    attached_pdf = attach_file(msg, report_pdf, max_attach_bytes)
    attached_zip = attach_file(msg, run_zip, max_attach_bytes)

    attached_any = attached_pdf or attached_zip

    if not attached_any:
        if download_url.strip():
            msg.set_content("\n".join(body_lines + [
                "",
                f"Attachments exceeded {max_attach_mb}MB or missing.",
                f"Download: {download_url.rstrip('/')}/{run_dir.name}/"
            ]))
        else:
            msg.set_content("\n".join(body_lines + [
                "",
                f"Attachments exceeded {max_attach_mb}MB or missing.",
                "Tip: set RUN_DOWNLOAD_URL to include a download link."
            ]))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as s:
        s.ehlo()
        # Try STARTTLS if port is 587-ish
        try:
            s.starttls(context=context)
            s.ehlo()
        except Exception:
            # Some SMTP servers might use 465 SSL or plain internal relay
            pass

        if smtp_user and smtp_pass:
            s.login(smtp_user, smtp_pass)

        s.send_message(msg)

if __name__ == "__main__":
    main()
