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
import tempfile
import shutil
from email.message import EmailMessage
from email.header import decode_header
from pathlib import Path
from datetime import datetime
import mimetypes

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


def sanitize_email_for_path(email_addr):
    """Convert email to safe directory name."""
    # Replace @ and . with underscores, remove other special chars
    safe = email_addr.lower().replace("@", "_at_").replace(".", "_")
    return "".join(c if c.isalnum() or c == "_" else "" for c in safe)


def verify_experiment_results(run_dir: Path, art_dir: Path) -> dict:
    """
    Verify experiment results and detect any issues.
    Returns dict with:
        - 'success': bool
        - 'errors': list of error messages
        - 'warnings': list of warning messages
        - 'missing_artifacts': list of missing expected artifacts
    """
    result = {
        'success': True,
        'errors': [],
        'warnings': [],
        'missing_artifacts': [],
        'error_log_content': ''
    }
    
    # Check experiment log for errors
    exp_log = art_dir / "logs" / "experiment.log"
    run_log = art_dir / "logs" / "run_stdout_stderr.log"
    
    error_keywords = ['Error', 'Exception', 'Traceback', 'ModuleNotFoundError', 
                      'ImportError', 'AttributeError', 'TypeError', 'ValueError',
                      'KeyError', 'IndexError', 'NameError', 'SyntaxError']
    
    for log_file in [exp_log, run_log]:
        if log_file.exists():
            try:
                log_content = log_file.read_text(encoding='utf-8', errors='ignore')
                for keyword in error_keywords:
                    if keyword in log_content:
                        result['success'] = False
                        # Extract relevant error lines
                        lines = log_content.split('\n')
                        error_lines = []
                        capture = False
                        for i, line in enumerate(lines):
                            if 'Traceback' in line or any(kw in line for kw in error_keywords):
                                capture = True
                            if capture:
                                error_lines.append(line)
                                if len(error_lines) > 30:  # Limit error context
                                    break
                        if error_lines:
                            result['errors'].append(f"Error in {log_file.name}:\n" + '\n'.join(error_lines[-20:]))
                            result['error_log_content'] = '\n'.join(error_lines[-30:])
                        break
            except Exception as e:
                result['warnings'].append(f"Could not read {log_file.name}: {e}")
    
    # Check for expected artifacts
    figures_dir = art_dir / "figures"
    tables_dir = art_dir / "tables"
    
    # Check if figures were generated
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.pdf"))
        if not figures:
            result['missing_artifacts'].append("No figures generated in artifacts/figures/")
            result['success'] = False
    else:
        result['missing_artifacts'].append("artifacts/figures/ directory missing")
        result['success'] = False
    
    # Check if tables were generated
    if tables_dir.exists():
        tables = list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.xlsx"))
        if not tables:
            result['missing_artifacts'].append("No tables generated in artifacts/tables/")
            # This is a warning, not a failure
            result['warnings'].append("No CSV/Excel tables generated")
    
    # Check for metrics.json
    metrics_file = art_dir / "metrics.json"
    if not metrics_file.exists():
        result['missing_artifacts'].append("metrics.json not generated")
        result['warnings'].append("metrics.json missing")
    
    return result


def run_claude_code_with_retry(prompt: str, run_dir: Path, art_dir: Path, max_retries: int = 2) -> tuple:
    """
    Run Claude Code with automatic retry on errors.
    Returns (success, error_msg, retry_count)
    """
    claude_path = os.environ.get("CLAUDE_PATH", "claude")
    claude_log = art_dir / "logs" / "claude_output.log"
    
    for attempt in range(max_retries + 1):
        print(f"[DEBUG] Claude Code attempt {attempt + 1}/{max_retries + 1}", flush=True)
        
        # Prepare prompt (include error feedback for retries)
        current_prompt = prompt
        
        # Write header to log file
        with open(claude_log, "a" if attempt > 0 else "w", encoding="utf-8") as log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"=== CLAUDE CODE ATTEMPT {attempt + 1} ===\n")
            log_file.write(f"Started at: {datetime.now().isoformat()}\n")
            if attempt > 0:
                log_file.write(f"This is a RETRY attempt to fix previous errors\n")
            log_file.write("=" * 60 + "\n\n")
        
        # Run Claude Code using subprocess.run (more reliable than Popen+threading for nohup)
        # stdin=DEVNULL prevents EBADF error when running via nohup
        try:
            result = subprocess.run(
                [claude_path, "-p", "--dangerously-skip-permissions", 
                 "--append-system-prompt-file", str(ROOT_DIR / "prompts" / "exp_rules.txt"), 
                 current_prompt],
                cwd=str(ROOT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=600,
                universal_newlines=True
            )
            claude_output = result.stdout or ""
            return_code = result.returncode
        except subprocess.TimeoutExpired as e:
            claude_output = ""
            if hasattr(e, 'stdout') and e.stdout:
                claude_output = e.stdout if isinstance(e.stdout, str) else e.stdout.decode('utf-8', errors='ignore')
            with open(claude_log, "a", encoding="utf-8") as f:
                if claude_output:
                    f.write(claude_output)
                f.write("\n\n=== TIMEOUT (600s) ===\n")
            print(f"[DEBUG] Claude Code timeout on attempt {attempt + 1}", flush=True)
            return False, "Claude Code timeout after 600 seconds", attempt
        except Exception as e:
            with open(claude_log, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== ERROR: {e} ===\n")
            print(f"[ERROR] Claude Code failed on attempt {attempt + 1}: {e}", flush=True)
            if attempt < max_retries:
                continue
            return False, f"Claude Code error: {e}", attempt
        
        # Write output to log file
        with open(claude_log, "a", encoding="utf-8") as f:
            if claude_output:
                f.write(claude_output)
            f.write(f"\n\n=== ATTEMPT {attempt + 1} FINISHED ===\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Ended at: {datetime.now().isoformat()}\n")
        
        # Print summary to console
        output_lines = claude_output.strip().split('\n') if claude_output.strip() else []
        print(f"[DEBUG] Claude attempt {attempt + 1} finished, return_code={return_code}, output_lines={len(output_lines)}", flush=True)
        if output_lines:
            for line in output_lines[-5:]:
                print(f"[CLAUDE] {line.rstrip()}", flush=True)
        
        # Check if required files exist
        required = [
            run_dir / "spec.yaml",
            run_dir / "run.sh",
            run_dir / "main.py",
            art_dir / "summary.md",
            art_dir / "report.qmd"
        ]
        missing = [str(f) for f in required if not f.exists()]
        
        if missing:
            if attempt < max_retries:
                # Prepare retry prompt
                current_prompt = f"""RETRY REQUEST - Previous attempt failed to create required files.

Missing files: {missing}

Please create ALL required files for the experiment. The original request was:
{prompt}

IMPORTANT: Make sure to create ALL files including spec.yaml, run.sh, main.py, artifacts/summary.md, and artifacts/report.qmd"""
                print(f"[RETRY] Missing files, retrying... ({attempt + 1}/{max_retries})")
                continue
            else:
                return False, f"Planning failed after {max_retries + 1} attempts. Missing files: {missing}", attempt
        
        # Files exist, now run the experiment
        (run_dir / "run.sh").chmod(0o755)
        conda_env = os.environ.get("CONDA_ENV", "openex")
        conda_base = os.environ.get("CONDA_BASE", os.path.expanduser("~/anaconda3"))
        
        with open(art_dir / "logs" / "run_stdout_stderr.log", "w") as log_file:
            exp_result = subprocess.run(
                ["bash", "-c", f"source {conda_base}/etc/profile.d/conda.sh && conda activate {conda_env} && bash {run_dir / 'run.sh'}"],
                cwd=str(ROOT_DIR),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                timeout=3600
            )
        
        # Verify results
        verification = verify_experiment_results(run_dir, art_dir)
        
        if not verification['success']:
            if attempt < max_retries:
                # Prepare fix prompt with error details
                error_details = '\n'.join(verification['errors'])
                missing_artifacts = '\n'.join(verification['missing_artifacts'])
                
                fix_prompt = f"""FIX REQUEST - The experiment code has errors that need to be fixed.

ERRORS FOUND:
{error_details}

MISSING ARTIFACTS:
{missing_artifacts}

ERROR LOG CONTENT:
{verification['error_log_content']}

Please fix the code in runs/{run_dir.relative_to(ROOT_DIR)}/ to resolve these errors.
Common issues to check:
1. Import statements - only use installed packages (scikit-learn, torch, numpy, pandas, matplotlib, seaborn, scipy, statsmodels)
2. Function arguments - make sure types match (e.g., don't pass int where str is expected)
3. Variable names - ensure consistency throughout the code
4. Make sure figures are saved to artifacts/figures/ and tables to artifacts/tables/

The original request was:
{prompt}"""
                
                current_prompt = fix_prompt
                print(f"[RETRY] Errors detected, sending fix request... ({attempt + 1}/{max_retries})")
                continue
            else:
                error_summary = '; '.join(verification['errors'][:3])
                return False, f"Experiment failed after {max_retries + 1} attempts. Errors: {error_summary}", attempt
        
        # Success!
        return True, "", attempt
    
    return False, "Max retries exceeded", max_retries

def run_experiment(request, sender_email, data_files=None):
    """
    Run experiment and return (run_dir, success, error_msg).
    Organizes runs by sender email.
    
    Args:
        request: The experiment request text
        sender_email: Email address of the sender
        data_files: Optional list of Path objects to user-uploaded data files
    """
    # Create user directory based on sender email
    user_dir = sanitize_email_for_path(sender_email)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_{}".format(os.getpid())
    run_dir = ROOT_DIR / "runs" / user_dir / run_id
    art_dir = run_dir / "artifacts"
    data_dir = run_dir / "data" / "raw"
    art_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    (art_dir / "figures").mkdir(exist_ok=True)
    (art_dir / "tables").mkdir(exist_ok=True)
    (art_dir / "logs").mkdir(exist_ok=True)

    # Copy user data files to run directory
    data_file_info = ""
    if data_files:
        for src_file in data_files:
            if src_file.exists():
                dst_file = data_dir / src_file.name
                shutil.copy2(src_file, dst_file)
                print(f"[DATA] Copied {src_file.name} to {dst_file}")
                
                # Generate data file info for the prompt
                try:
                    import pandas as pd
                    if src_file.suffix.lower() == '.csv':
                        df = pd.read_csv(src_file, nrows=5)
                    elif src_file.suffix.lower() in ['.xlsx', '.xls']:
                        df = pd.read_excel(src_file, nrows=5)
                    else:
                        df = None
                    
                    if df is not None:
                        data_file_info += f"\n\nData file: {src_file.name}\n"
                        data_file_info += f"Shape: {df.shape[0]}+ rows x {df.shape[1]} columns\n"
                        data_file_info += f"Columns: {list(df.columns)}\n"
                        data_file_info += f"Data types:\n{df.dtypes.to_string()}\n"
                        data_file_info += f"Sample (first 5 rows):\n{df.head().to_string()}\n"
                except Exception as e:
                    print(f"[WARN] Could not preview data file {src_file.name}: {e}")
                    data_file_info += f"\n\nData file: {src_file.name} (preview unavailable)\n"

    # Step 1: Planning with Claude Code
    full_run_path = "{}/{}".format(user_dir, run_id)
    
    # Build prompt with data file information if available
    data_section = ""
    if data_files:
        data_section = f"""

User has uploaded data files to runs/{full_run_path}/data/raw/:
{data_file_info}

Please analyze this data and design an appropriate experiment/analysis pipeline.
The experiment should:
1. Load data from data/raw/ directory
2. Perform exploratory data analysis (EDA)
3. Design and implement appropriate models/analyses based on data characteristics
4. Generate comprehensive visualizations
5. Save all results to artifacts/
"""
    
    prompt = """User request: {request}
{data_section}
Please create an engineering-grade experiment under runs/{full_run_path}/ with the following structure:
1) spec.yaml: structured experiment design (data, model, hyperparams, seeds, metrics, statistical tests, artifact list)
2) main.py: main entry point that controls all hyperparameters
3) config/: configuration module with config.py and default_config.yaml
4) src/: modular source code (data/, models/, training/, utils/, visualization/)
5) run.sh: executable script that runs main.py
6) artifacts/summary.md: a concise summary (metrics, conclusions, next steps)
7) artifacts/report.qmd: based on templates/report.qmd, fill in this experiment's content

IMPORTANT: 
- All visualizations must be dynamically planned based on experiment type
- Generate figures to artifacts/figures/ and tables to artifacts/tables/
- All outputs must be written to runs/{full_run_path}/
- run.sh must write logs to artifacts/logs/""".format(request=request, data_section=data_section, full_run_path=full_run_path)

    claude_log = art_dir / "logs" / "claude_output.log"
    
    try:
        # Get max retries from environment (default 2)
        max_retries = int(os.environ.get("MAX_RETRIES", "2"))
        
        print("[DEBUG] Running Claude Code with auto-retry (max {} retries)...".format(max_retries))
        print("[DEBUG] Log file: {}".format(claude_log))
        
        # Run Claude Code with automatic retry on errors
        success, error_msg, retry_count = run_claude_code_with_retry(prompt, run_dir, art_dir, max_retries)
        
        if not success:
            return run_dir, False, error_msg
        
        if retry_count > 0:
            print(f"[INFO] Experiment succeeded after {retry_count + 1} attempts")

        # Render report
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

        # Pack artifacts
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
