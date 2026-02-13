#!/usr/bin/env python3
"""
Email listener for cc-exp-runner.
Polls inbox for experiment requests, processes them, and replies with results.

This module handles ONLY email I/O:
1. Connect to IMAP inbox
2. Fetch unread emails
3. Parse request from subject/body/attachments
4. Delegate experiment execution to ExperimentRunner
5. Reply to sender with results
6. Mark email as read

Experiment execution logic lives in:
- tools/experiment_runner.py (middleware / orchestration)
- tools/cc_executor.py (Claude Code subprocess management)
"""
import os
import sys
import time
import email
import imaplib
import smtplib
import ssl
import json
import shutil
from email.message import EmailMessage
from email.header import decode_header
from pathlib import Path
from datetime import datetime
import mimetypes

# Ensure project root is in sys.path for both direct execution and module import
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from tools.experiment_runner import ExperimentRunner

# Optional imports for document parsing
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

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


def extract_pdf_text(file_path: Path) -> str:
    """Extract text content from PDF file."""
    text = ""
    
    # Try pdfplumber first (better for complex PDFs)
    if HAS_PDFPLUMBER:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"[WARN] pdfplumber failed: {e}")
    
    # Fallback to PyPDF2
    if HAS_PYPDF2:
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"[WARN] PyPDF2 failed: {e}")
    
    return text.strip()


def extract_markdown_text(file_path: Path) -> str:
    """Extract text content from Markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content.strip()
    except Exception as e:
        print(f"[WARN] Failed to read markdown file: {e}")
        return ""


def extract_attachments(msg, save_dir: Path) -> dict:
    """
    Extract attachments from email message.
    Returns dict with:
        - 'data_files': list of paths to CSV/Excel files
        - 'document_content': combined text from PDF/MD files
        - 'all_files': list of all saved attachment paths
    """
    result = {
        'data_files': [],
        'document_content': [],
        'all_files': []
    }
    
    if not msg.is_multipart():
        return result
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for part in msg.walk():
        # Skip non-attachment parts
        content_disposition = part.get("Content-Disposition", "")
        if "attachment" not in content_disposition.lower():
            # Also check for inline attachments with filename
            if not part.get_filename():
                continue
        
        filename = part.get_filename()
        if not filename:
            continue
        
        # Decode filename if needed
        filename = decode_mime_header(filename)
        
        # Sanitize filename
        safe_filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_filename:
            continue
        
        # Get file extension
        ext = Path(safe_filename).suffix.lower()
        
        # Save attachment
        file_path = save_dir / safe_filename
        try:
            payload = part.get_payload(decode=True)
            if payload:
                with open(file_path, 'wb') as f:
                    f.write(payload)
                result['all_files'].append(file_path)
                print(f"[ATTACH] Saved: {safe_filename}")
                
                # Categorize by type
                if ext in ['.csv', '.xlsx', '.xls', '.tsv', '.parquet']:
                    result['data_files'].append(file_path)
                    print(f"[DATA] Found data file: {safe_filename}")
                
                elif ext == '.pdf':
                    pdf_text = extract_pdf_text(file_path)
                    if pdf_text:
                        result['document_content'].append(f"=== Content from {filename} ===\n{pdf_text}")
                        print(f"[PDF] Extracted {len(pdf_text)} chars from {safe_filename}")
                
                elif ext in ['.md', '.markdown', '.txt']:
                    md_text = extract_markdown_text(file_path)
                    if md_text:
                        result['document_content'].append(f"=== Content from {filename} ===\n{md_text}")
                        print(f"[DOC] Extracted {len(md_text)} chars from {safe_filename}")
                        
        except Exception as e:
            print(f"[ERROR] Failed to save attachment {filename}: {e}")
    
    return result


def build_experiment_request(subject: str, body: str, attachments: dict) -> str:
    """
    Build experiment request from email subject, body, and attachments.
    Prioritizes: subject -> body -> attachment content
    """
    request_parts = []
    
    # Check subject first
    subject = subject.strip() if subject else ""
    generic_subjects = ["experiment", "run", "test", "openlab", "request", ""]
    
    if subject and subject.lower() not in generic_subjects:
        request_parts.append(f"Subject: {subject}")
    
    # Add body content (clean up email signatures and quoted text)
    if body:
        # Remove common email signatures and quoted replies
        body_lines = []
        for line in body.split('\n'):
            line_stripped = line.strip()
            # Stop at signature markers or quoted text
            if line_stripped.startswith('--') and len(line_stripped) <= 3:
                break
            if line_stripped.startswith('>'):
                continue
            if 'sent from my' in line_stripped.lower():
                continue
            if line_stripped:
                body_lines.append(line)
        
        clean_body = '\n'.join(body_lines).strip()
        if clean_body:
            request_parts.append(f"Request:\n{clean_body}")
    
    # Add document content from attachments
    if attachments.get('document_content'):
        doc_content = '\n\n'.join(attachments['document_content'])
        request_parts.append(f"Attached Document Content:\n{doc_content}")
    
    # Add data file information
    if attachments.get('data_files'):
        data_info = "Uploaded Data Files:\n"
        for f in attachments['data_files']:
            data_info += f"  - {f.name}\n"
        request_parts.append(data_info)
    
    # Combine all parts
    full_request = '\n\n'.join(request_parts)
    
    # If still empty, return None
    if not full_request.strip():
        return ""
    
    return full_request


def run_experiment(request, sender_email, data_files=None):
    """
    Run experiment and return (run_dir, success, error_msg).
    Delegates to ExperimentRunner middleware.
    
    Args:
        request: The experiment request text
        sender_email: Email address of the sender
        data_files: Optional list of Path objects to user-uploaded data files
    """
    runner = ExperimentRunner()
    result = runner.run(request, sender=sender_email, data_files=data_files)
    return result.run_dir, result.success, result.error_msg

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
            
            # Create temp directory for attachments
            temp_attach_dir = ROOT_DIR / "temp_attachments" / email_id_str
            
            # Extract attachments (PDF, MD, CSV, etc.)
            attachments = extract_attachments(msg, temp_attach_dir)
            
            # Build experiment request from subject, body, and attachments
            request = build_experiment_request(subject, body, attachments)

            if not request:
                print(f"[SKIP] Empty request from {reply_to}")
                processed.add(email_id_str)
                # Clean up temp attachments
                if temp_attach_dir.exists():
                    shutil.rmtree(temp_attach_dir, ignore_errors=True)
                continue

            print(f"[PROCESS] From: {reply_to}")
            print(f"[PROCESS] Subject: {subject[:50]}..." if subject else "[PROCESS] No subject")
            print(f"[PROCESS] Body length: {len(body)} chars")
            print(f"[PROCESS] Attachments: {len(attachments.get('all_files', []))} files")
            print(f"[PROCESS] Data files: {len(attachments.get('data_files', []))} files")
            print(f"[PROCESS] Request preview: {request[:100]}...")

            # Run experiment with data files if any
            data_files = attachments.get('data_files', [])
            run_dir, success, error_msg = run_experiment(request, reply_to, data_files=data_files if data_files else None)

            # Send reply
            try:
                send_reply(config, reply_to, subject, run_dir, success, error_msg)
                print(f"[DONE] Replied to {reply_to}, run_dir={run_dir.name}, success={success}")
            except Exception as e:
                print(f"[ERROR] Failed to send reply: {e}")

            # Clean up temp attachments
            if temp_attach_dir.exists():
                shutil.rmtree(temp_attach_dir, ignore_errors=True)

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
