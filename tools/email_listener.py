#!/usr/bin/env python3
"""
Email listener for cc-exp-runner.
Polls inbox for experiment requests, processes them, and replies with results.

Flow:
1. Connect to IMAP inbox (openex@code-soul.com)
2. Fetch unread emails
3. For each email:
   - Extract sender email (reply-to address)
   - Extract experiment request from subject/body
   - Run experiment via subprocess
   - Reply to sender with results + attachments
4. Mark email as read
5. Sleep and repeat
"""
import os
import sys
import time
import email
import imaplib
import smtplib
import ssl
import subprocess
import json
from email.message import EmailMessage
from email.header import decode_header
from pathlib import Path
from datetime import datetime
import mimetypes

ROOT_DIR = Path(__file__).parent.parent.resolve()
PROCESSED_FILE = ROOT_DIR / "config" / "processed_emails.json"
ENV_FILE = ROOT_DIR / ".env"

def load_env():
    """Load environment variables from .env file."""
    if ENV_FILE.exists():
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())

# Load .env on import
load_env()

def load_config() -> dict:
    """Load server configuration from environment variables."""
    config = {
        "email": os.environ.get("EMAIL_ADDRESS", ""),
        "imap": {
            "host": os.environ.get("IMAP_HOST", ""),
            "port": int(os.environ.get("IMAP_PORT", "993")),
            "user": os.environ.get("IMAP_USER", ""),
            "pass": os.environ.get("IMAP_PASS", ""),
        },
        "smtp": {
            "host": os.environ.get("SMTP_HOST", ""),
            "port": int(os.environ.get("SMTP_PORT", "587")),
            "user": os.environ.get("SMTP_USER", ""),
            "pass": os.environ.get("SMTP_PASS", ""),
        },
        "poll_interval": int(os.environ.get("POLL_INTERVAL", "60")),
    }
    
    # Validate required fields
    if not config["email"] or not config["imap"]["host"] or not config["smtp"]["host"]:
        raise RuntimeError("Missing required environment variables. Copy .env.example to .env and fill in your values.")
    
    return config

def load_processed() -> set:
    """Load set of processed email IDs."""
    if not PROCESSED_FILE.exists():
        return set()
    with open(PROCESSED_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(data.get("processed", []))

def save_processed(processed: set):
    """Save processed email IDs."""
    PROCESSED_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_FILE, "w", encoding="utf-8") as f:
        json.dump({"processed": list(processed)[-1000:]}, f)  # Keep last 1000

def decode_mime_header(header_value: str) -> str:
    """Decode MIME encoded header."""
    if not header_value:
        return ""
    decoded_parts = decode_header(header_value)
    result = []
    for part, charset in decoded_parts:
        if isinstance(part, bytes):
            result.append(part.decode(charset or "utf-8", errors="ignore"))
        else:
            result.append(part)
    return "".join(result)

def get_email_body(msg) -> str:
    """Extract plain text body from email."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                charset = part.get_content_charset() or "utf-8"
                return payload.decode(charset, errors="ignore")
    else:
        payload = msg.get_payload(decode=True)
        charset = msg.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="ignore")
    return ""

def attach_file(msg: EmailMessage, path: Path, max_bytes: int = 20 * 1024 * 1024) -> bool:
    """Attach file to email if within size limit."""
    if not path.exists():
        return False
    if path.stat().st_size > max_bytes:
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

def sanitize_email_for_path(email_addr):
    """Convert email to safe directory name."""
    # Replace @ and . with underscores, remove other special chars
    safe = email_addr.lower().replace("@", "_at_").replace(".", "_")
    return "".join(c if c.isalnum() or c == "_" else "" for c in safe)

def run_experiment(request, sender_email):
    """
    Run experiment and return (run_dir, success, error_msg).
    Organizes runs by sender email.
    """
    # Create user directory based on sender email
    user_dir = sanitize_email_for_path(sender_email)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_{}".format(os.getpid())
    run_dir = ROOT_DIR / "runs" / user_dir / run_id
    art_dir = run_dir / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "figures").mkdir(exist_ok=True)
    (art_dir / "tables").mkdir(exist_ok=True)
    (art_dir / "logs").mkdir(exist_ok=True)

    # Step 1: Planning with Claude Code
    full_run_path = "{}/{}".format(user_dir, run_id)
    prompt = """User request: {request}

Please create the following under runs/{full_run_path}/:
1) spec.yaml: structured experiment design (data, model, hyperparams, seeds, metrics, statistical tests, artifact list)
2) run.sh: executable script (includes: prepare data -> train -> evaluate -> generate figures/tables -> write metrics.json & summary.md)
3) artifacts/summary.md: a concise summary (metrics, conclusions, next steps)
4) artifacts/report.qmd: based on templates/report.qmd, fill in this experiment's content, referencing artifacts/figures and artifacts/tables
Note: all outputs must be written to runs/{full_run_path}/; run.sh must write logs to artifacts/logs/.""".format(request=request, full_run_path=full_run_path)

    # Log file for Claude Code output (real-time)
    claude_log = art_dir / "logs" / "claude_output.log"
    
    try:
        # Run Claude Code for planning with REAL-TIME logging
        # Get claude path from environment or use default
        claude_path = os.environ.get("CLAUDE_PATH", "claude")
        
        print("[DEBUG] Running Claude Code...")
        print("[DEBUG] Log file: {}".format(claude_log))
        print("[DEBUG] Use 'tail -f {}' to watch in real-time".format(claude_log))
        
        # Write header to log file
        with open(claude_log, "w", encoding="utf-8") as log_file:
            log_file.write("=== CLAUDE CODE OUTPUT ===\n")
            log_file.write("Started at: {}\n".format(datetime.now().isoformat()))
            log_file.write("Prompt: {}\n".format(prompt[:500]))
            log_file.write("=" * 60 + "\n\n")
        
        # Run Claude Code with real-time output capture using threading
        import threading
        
        proc = subprocess.Popen(
            [claude_path, "-p", "--dangerously-skip-permissions", "--append-system-prompt-file", str(ROOT_DIR / "prompts" / "exp_rules.txt"), prompt],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        
        # Thread to read and write output in real-time
        def stream_output():
            with open(claude_log, "a", encoding="utf-8") as f:
                for line in iter(proc.stdout.readline, ''):
                    if line:
                        f.write(line)
                        f.flush()
                        print("[CLAUDE] {}".format(line.rstrip()))
                proc.stdout.close()
        
        output_thread = threading.Thread(target=stream_output)
        output_thread.start()
        
        # Wait with timeout
        try:
            return_code = proc.wait(timeout=600)
            output_thread.join(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            output_thread.join(timeout=5)
            with open(claude_log, "a", encoding="utf-8") as f:
                f.write("\n\n=== TIMEOUT (600s) ===\n")
            return run_dir, False, "Claude Code timeout after 600 seconds"
        
        with open(claude_log, "a", encoding="utf-8") as f:
            f.write("\n\n=== FINISHED ===\n")
            f.write("Return code: {}\n".format(return_code))
            f.write("Ended at: {}\n".format(datetime.now().isoformat()))
        
        print("[DEBUG] Claude finished with return code: {}".format(return_code))
        
        # Check required files
        required = [run_dir / "spec.yaml", run_dir / "run.sh", art_dir / "summary.md", art_dir / "report.qmd"]
        missing = [str(f) for f in required if not f.exists()]
        if missing:
            return run_dir, False, "Planning failed. Missing files: {}\nClaude output saved to: {}".format(missing, claude_log)

        # Step 2: Run experiment in conda environment
        (run_dir / "run.sh").chmod(0o755)
        conda_env = os.environ.get("CONDA_ENV", "openex")
        conda_base = os.environ.get("CONDA_BASE", os.path.expanduser("~/anaconda3"))
        with open(art_dir / "logs" / "run_stdout_stderr.log", "w") as log_file:
            # Use conda run to execute in specified environment
            exp_result = subprocess.run(
                ["bash", "-c", "source {}/etc/profile.d/conda.sh && conda activate {} && bash {}".format(conda_base, conda_env, run_dir / "run.sh")],
                cwd=str(ROOT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600  # 1 hour timeout for experiment
            )

        # Step 3: Render report (Python 3.6 compatible)
        subprocess.run(
            ["quarto", "render", str(art_dir / "report.qmd"), "--to", "pdf"],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )
        subprocess.run(
            ["quarto", "render", str(art_dir / "report.qmd"), "--to", "gfm"],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
        )

        # Step 4: Pack artifacts (Python 3.6 compatible)
        subprocess.run(
            ["python3", str(ROOT_DIR / "tools" / "pack_run.py"), str(run_dir)],
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )

        return run_dir, True, ""

    except subprocess.TimeoutExpired as e:
        return run_dir, False, f"Timeout: {e}"
    except Exception as e:
        return run_dir, False, f"Error: {e}"

def send_reply(config: dict, to_addr: str, subject: str, run_dir: Path, success: bool, error_msg: str):
    """Send reply email with results."""
    smtp_cfg = config.get("smtp", {})
    smtp_host = smtp_cfg.get("host")
    smtp_port = smtp_cfg.get("port", 587)
    smtp_user = smtp_cfg.get("user")
    smtp_pass = smtp_cfg.get("pass")
    smtp_from = config.get("email")

    art_dir = run_dir / "artifacts"
    summary_file = art_dir / "summary.md"
    report_pdf = art_dir / "report.pdf"
    run_zip = art_dir / "run.zip"

    # Build email body
    body_lines = [f"Experiment Run: {run_dir.name}", ""]
    
    if success:
        body_lines.append("Status: SUCCESS")
        body_lines.append("")
        if summary_file.exists():
            body_lines.append("=== Summary ===")
            body_lines.append(summary_file.read_text(encoding="utf-8", errors="ignore"))
        else:
            body_lines.append("(No summary generated)")
    else:
        body_lines.append("Status: FAILED")
        body_lines.append("")
        body_lines.append(f"Error: {error_msg}")

    msg = EmailMessage()
    msg["Subject"] = f"Re: {subject}"
    msg["From"] = smtp_from
    msg["To"] = to_addr
    msg.set_content("\n".join(body_lines))

    # Attach files if successful
    if success:
        attach_file(msg, report_pdf)
        attach_file(msg, run_zip)

    # Send
    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=60) as s:
        s.ehlo()
        try:
            s.starttls(context=context)
            s.ehlo()
        except Exception:
            pass
        if smtp_user and smtp_pass:
            s.login(smtp_user, smtp_pass)
        s.send_message(msg)

def poll_inbox(config: dict, processed: set) -> set:
    """Poll inbox for new emails and process them."""
    imap_cfg = config.get("imap", {})
    imap_host = imap_cfg.get("host")
    imap_port = imap_cfg.get("port", 993)
    imap_user = imap_cfg.get("user")
    imap_pass = imap_cfg.get("pass")

    context = ssl.create_default_context()
    
    with imaplib.IMAP4_SSL(imap_host, imap_port, ssl_context=context) as mail:
        mail.login(imap_user, imap_pass)
        mail.select("INBOX")

        # Search for unseen emails
        status, messages = mail.search(None, "UNSEEN")
        if status != "OK":
            return processed

        email_ids = messages[0].split()
        
        for email_id in email_ids:
            email_id_str = email_id.decode()
            if email_id_str in processed:
                continue

            # Fetch email
            status, msg_data = mail.fetch(email_id, "(RFC822)")
            if status != "OK":
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            # Extract info
            from_addr = decode_mime_header(msg.get("From", ""))
            # Extract actual email address from "Name <email>" format
            if "<" in from_addr and ">" in from_addr:
                reply_to = from_addr[from_addr.index("<")+1:from_addr.index(">")]
            else:
                reply_to = from_addr.strip()

            subject = decode_mime_header(msg.get("Subject", ""))
            body = get_email_body(msg)

            # Use subject as experiment request, fallback to body
            request = subject.strip()
            if not request or request.lower() in ["experiment", "run", "test"]:
                request = body.strip().split("\n")[0]  # First line of body

            if not request:
                print(f"[SKIP] Empty request from {reply_to}")
                processed.add(email_id_str)
                continue

            print(f"[PROCESS] From: {reply_to}, Request: {request[:80]}...")

            # Run experiment (pass sender email for per-user directory)
            run_dir, success, error_msg = run_experiment(request, reply_to)

            # Send reply
            try:
                send_reply(config, reply_to, subject, run_dir, success, error_msg)
                print(f"[DONE] Replied to {reply_to}, run_dir={run_dir.name}, success={success}")
            except Exception as e:
                print(f"[ERROR] Failed to send reply: {e}")

            processed.add(email_id_str)
            save_processed(processed)

            # Mark as read
            mail.store(email_id, "+FLAGS", "\\Seen")

    return processed

def main():
    print("=== Email Listener for cc-exp-runner ===")
    print(f"Config: .env")
    
    config = load_config()
    poll_interval = config.get("poll_interval", 60)  # seconds
    
    print(f"Email: {config.get('email')}")
    print(f"Poll interval: {poll_interval}s")
    print("Listening for experiment requests...")
    print("")

    processed = load_processed()

    while True:
        try:
            processed = poll_inbox(config, processed)
        except Exception as e:
            print(f"[ERROR] Poll failed: {e}")
        
        time.sleep(poll_interval)

if __name__ == "__main__":
    main()
