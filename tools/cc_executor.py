#!/usr/bin/env python3
"""
Claude Code Executor - manages CC subprocess execution with retry, precheck, and verification.

This module is independent of any I/O channel (email, CLI, API).
It only knows how to:
1. Invoke Claude Code with a prompt
2. Pre-check generated code (syntax + import)
3. Run the experiment script
4. Verify results (figures, tables, metrics)
5. Retry on failure with error-specific fix prompts

Usage:
    from tools.cc_executor import CCExecutor
    executor = CCExecutor(run_dir, art_dir)
    result = executor.execute(prompt)
"""
import os
import re
import subprocess
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field


ROOT_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class ExecutionResult:
    """Result of a CC execution pipeline."""
    success: bool
    error_msg: str = ""
    attempt_count: int = 0
    run_dir: Path = None
    art_dir: Path = None
    # Intermediate results exposed for inspection
    claude_output: str = ""
    precheck_errors: str = ""
    verification: dict = field(default_factory=dict)


class CCExecutor:
    """
    Manages Claude Code subprocess execution with retry logic.
    
    Attributes:
        run_dir: Path to the experiment run directory
        art_dir: Path to the artifacts directory
        max_retries: Maximum number of retry attempts (default 2)
        claude_timeout: Timeout for Claude Code subprocess in seconds (default 1200)
        experiment_timeout: Timeout for experiment execution in seconds (default 3600)
        conda_env: Conda environment name (default from env or 'openex')
        conda_base: Conda base path (default from env or ~/anaconda3)
        claude_path: Path to claude binary (default from env or 'claude')
    """

    def __init__(
        self,
        run_dir: Path,
        art_dir: Path,
        max_retries: int = 2,
        claude_timeout: int = 1200,
        experiment_timeout: int = 3600,
        conda_env: str = None,
        conda_base: str = None,
        claude_path: str = None,
    ):
        self.run_dir = Path(run_dir)
        self.art_dir = Path(art_dir)
        self.max_retries = max_retries
        self.claude_timeout = claude_timeout
        self.experiment_timeout = experiment_timeout
        self.conda_env = conda_env or os.environ.get("CONDA_ENV", "openex")
        self.conda_base = conda_base or os.environ.get("CONDA_BASE", os.path.expanduser("~/anaconda3"))
        self.claude_path = claude_path or os.environ.get("CLAUDE_PATH", "claude")
        self.claude_log = self.art_dir / "logs" / "claude_output.log"
        self.run_log = self.art_dir / "logs" / "run_stdout_stderr.log"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, prompt: str) -> ExecutionResult:
        """
        Full pipeline: invoke CC -> precheck -> run experiment -> verify.
        Retries automatically on failure.
        
        Returns ExecutionResult with all intermediate data exposed.
        """
        current_prompt = prompt

        for attempt in range(self.max_retries + 1):
            self._log(f"Claude Code attempt {attempt + 1}/{self.max_retries + 1}")

            # --- Phase 1: Invoke Claude Code ---
            cc_ok, cc_err, claude_output, timed_out = self._invoke_claude(current_prompt, attempt)
            if not cc_ok and not timed_out:
                if attempt < self.max_retries:
                    continue
                return ExecutionResult(
                    success=False, error_msg=cc_err, attempt_count=attempt,
                    run_dir=self.run_dir, art_dir=self.art_dir, claude_output=claude_output,
                )

            # Handle timeout salvage
            if timed_out:
                if (self.run_dir / "main.py").exists():
                    self._log("main.py exists after timeout, auto-generating missing files...")
                    self._auto_generate_missing_files()
                else:
                    if attempt < self.max_retries:
                        continue
                    return ExecutionResult(
                        success=False,
                        error_msg="Claude Code timeout after {}s (no main.py generated)".format(self.claude_timeout),
                        attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                    )

            # --- Phase 2: Check required files ---
            missing = self._check_required_files()
            if missing:
                if attempt < self.max_retries:
                    current_prompt = self._build_missing_files_prompt(missing, prompt)
                    self._log(f"Missing files, retrying... ({attempt + 1}/{self.max_retries})")
                    continue
                return ExecutionResult(
                    success=False,
                    error_msg=f"Planning failed after {self.max_retries + 1} attempts. Missing: {missing}",
                    attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                )

            # --- Phase 3: Syntax pre-check (compile all .py) ---
            (self.run_dir / "run.sh").chmod(0o755)
            syntax_ok, syntax_err = self._precheck_syntax()
            if not syntax_ok:
                if attempt < self.max_retries:
                    current_prompt = self._build_syntax_fix_prompt(syntax_err)
                    self._log(f"Syntax pre-check failed, retrying... ({attempt + 1}/{self.max_retries})")
                    continue
                return ExecutionResult(
                    success=False,
                    error_msg=f"Syntax pre-check failed after {self.max_retries + 1} attempts: {syntax_err[:500]}",
                    attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                    precheck_errors=syntax_err,
                )

            # --- Phase 4: Import pre-check ---
            import_ok, import_err = self._precheck_import()
            if not import_ok:
                if attempt < self.max_retries:
                    current_prompt = self._build_import_fix_prompt(import_err)
                    self._log(f"Import pre-check failed, retrying... ({attempt + 1}/{self.max_retries})")
                    continue
                return ExecutionResult(
                    success=False,
                    error_msg=f"Import pre-check failed after {self.max_retries + 1} attempts: {import_err[:500]}",
                    attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                    precheck_errors=import_err,
                )

            self._log("Pre-check passed, running experiment...")

            # --- Phase 5: Run experiment ---
            run_ok, run_err = self._run_experiment()
            if not run_ok:
                if attempt < self.max_retries:
                    current_prompt = self._build_runtime_fix_prompt(run_err, prompt)
                    self._log(f"Experiment runtime error, retrying... ({attempt + 1}/{self.max_retries})")
                    continue
                return ExecutionResult(
                    success=False, error_msg=run_err,
                    attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                )

            # --- Phase 6: Verify results ---
            verification = self.verify_results()
            if not verification["success"]:
                if attempt < self.max_retries:
                    current_prompt = self._build_verification_fix_prompt(verification, prompt)
                    self._log(f"Verification failed, retrying... ({attempt + 1}/{self.max_retries})")
                    continue
                error_summary = "; ".join(verification["errors"][:3])
                return ExecutionResult(
                    success=False,
                    error_msg=f"Experiment failed after {self.max_retries + 1} attempts. Errors: {error_summary}",
                    attempt_count=attempt, run_dir=self.run_dir, art_dir=self.art_dir,
                    verification=verification,
                )

            # Success
            if attempt > 0:
                self._log(f"Experiment succeeded after {attempt + 1} attempts")
            return ExecutionResult(
                success=True, attempt_count=attempt,
                run_dir=self.run_dir, art_dir=self.art_dir,
                claude_output=claude_output, verification=verification,
            )

        return ExecutionResult(
            success=False, error_msg="Max retries exceeded",
            attempt_count=self.max_retries, run_dir=self.run_dir, art_dir=self.art_dir,
        )

    def verify_results(self) -> dict:
        """
        Verify experiment results and detect issues.
        Can be called independently for inspection.
        """
        result = {
            "success": True,
            "errors": [],
            "warnings": [],
            "missing_artifacts": [],
            "error_log_content": "",
        }

        # Check logs for errors
        error_keywords = [
            "Error", "Exception", "Traceback", "ModuleNotFoundError",
            "ImportError", "AttributeError", "TypeError", "ValueError",
            "KeyError", "IndexError", "NameError", "SyntaxError",
        ]

        for log_file in [self.art_dir / "logs" / "experiment.log", self.run_log]:
            if not log_file.exists():
                continue
            try:
                log_content = log_file.read_text(encoding="utf-8", errors="ignore")
                for keyword in error_keywords:
                    if keyword in log_content:
                        result["success"] = False
                        lines = log_content.split("\n")
                        error_lines = []
                        capture = False
                        for line in lines:
                            if "Traceback" in line or any(kw in line for kw in error_keywords):
                                capture = True
                            if capture:
                                error_lines.append(line)
                                if len(error_lines) > 30:
                                    break
                        if error_lines:
                            result["errors"].append(
                                f"Error in {log_file.name}:\n" + "\n".join(error_lines[-20:])
                            )
                            result["error_log_content"] = "\n".join(error_lines[-30:])
                        break
            except Exception as e:
                result["warnings"].append(f"Could not read {log_file.name}: {e}")

        # Check figures
        figures_dir = self.art_dir / "figures"
        if figures_dir.exists():
            figures = list(figures_dir.glob("*.png")) + list(figures_dir.glob("*.pdf"))
            if not figures:
                result["missing_artifacts"].append("No figures generated in artifacts/figures/")
                result["success"] = False
        else:
            result["missing_artifacts"].append("artifacts/figures/ directory missing")
            result["success"] = False

        # Check tables
        tables_dir = self.art_dir / "tables"
        if tables_dir.exists():
            tables = list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.xlsx"))
            if not tables:
                result["missing_artifacts"].append("No tables generated in artifacts/tables/")
                result["warnings"].append("No CSV/Excel tables generated")

        # Check metrics.json
        if not (self.art_dir / "metrics.json").exists():
            result["missing_artifacts"].append("metrics.json not generated")
            result["warnings"].append("metrics.json missing")

        return result

    # ------------------------------------------------------------------
    # Phase implementations (all exposed for testing)
    # ------------------------------------------------------------------

    def _invoke_claude(self, prompt: str, attempt: int) -> tuple:
        """
        Invoke Claude Code subprocess with real-time output streaming.
        Uses Popen + threading so output is visible immediately.
        Returns (success, error_msg, output, timed_out)
        """
        # Write log header
        mode = "a" if attempt > 0 else "w"
        with open(self.claude_log, mode, encoding="utf-8") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"=== CLAUDE CODE ATTEMPT {attempt + 1} ===\n")
            f.write(f"Started at: {datetime.now().isoformat()}\n")
            if attempt > 0:
                f.write("This is a RETRY attempt to fix previous errors\n")
            f.write("=" * 60 + "\n\n")

        self._log(f"Log file: {self.claude_log}")
        self._log(f"Use 'tail -f {self.claude_log}' to watch in real-time")

        claude_output_lines = []
        return_code = -1
        timed_out = False

        try:
            proc = subprocess.Popen(
                [
                    self.claude_path, "-p", "--dangerously-skip-permissions",
                    "--append-system-prompt-file", str(ROOT_DIR / "prompts" / "exp_rules.txt"),
                    prompt,
                ],
                cwd=str(ROOT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
            )

            # Thread to read and log output in real-time
            def stream_output():
                with open(self.claude_log, "a", encoding="utf-8") as f:
                    for line in iter(proc.stdout.readline, ""):
                        if line:
                            f.write(line)
                            f.flush()
                            claude_output_lines.append(line)
                            print(f"[CLAUDE] {line.rstrip()}", flush=True)
                    proc.stdout.close()

            output_thread = threading.Thread(target=stream_output, daemon=True)
            output_thread.start()

            # Wait with timeout
            try:
                return_code = proc.wait(timeout=self.claude_timeout)
                output_thread.join(timeout=10)
            except subprocess.TimeoutExpired:
                timed_out = True
                proc.kill()
                output_thread.join(timeout=10)
                self._log(f"Claude Code timeout on attempt {attempt + 1}")

        except Exception as e:
            with open(self.claude_log, "a", encoding="utf-8") as f:
                f.write(f"\n\n=== ERROR: {e} ===\n")
            self._log(f"Claude Code error on attempt {attempt + 1}: {e}")
            return False, f"Claude Code error: {e}", "", False

        claude_output = "".join(claude_output_lines)

        # Write footer to log
        with open(self.claude_log, "a", encoding="utf-8") as f:
            if timed_out:
                f.write(f"\n\n=== TIMEOUT ({self.claude_timeout}s) ===\n")
            f.write(f"\n\n=== ATTEMPT {attempt + 1} FINISHED ===\n")
            f.write(f"Return code: {return_code}\n")
            f.write(f"Ended at: {datetime.now().isoformat()}\n")

        # Print summary
        self._log(
            f"Claude attempt {attempt + 1} finished, return_code={return_code}, "
            f"output_lines={len(claude_output_lines)}, timed_out={timed_out}"
        )

        return True, "", claude_output, timed_out

    def _check_required_files(self) -> list:
        """Check if all required files exist. Returns list of missing file paths."""
        required = [
            self.run_dir / "spec.yaml",
            self.run_dir / "run.sh",
            self.run_dir / "main.py",
            self.art_dir / "summary.md",
            self.art_dir / "report.qmd",
        ]
        return [str(f) for f in required if not f.exists()]

    def _precheck_syntax(self) -> tuple:
        """
        Compile ALL .py files to catch syntax errors.
        Returns (success, error_message)
        """
        cmd = (
            f"source {self.conda_base}/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"cd {self.run_dir} && "
            f"find . -name '*.py' -exec python3 -m py_compile {{}} \\; 2>&1"
        )
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(self.run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            universal_newlines=True,
        )
        if result.returncode != 0:
            errors = result.stdout or "Unknown compile error"
            self._log(f"Syntax error: {errors[:200]}")
            with open(self.run_log, "w") as f:
                f.write(f"PRE-CHECK SYNTAX FAILED:\n{errors}\n")
            return False, errors
        return True, ""

    def _precheck_import(self) -> tuple:
        """
        Verify cross-module imports by importing main.py.
        Returns (success, error_message)
        """
        cmd = (
            f"source {self.conda_base}/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"cd {self.run_dir} && "
            f"python3 -c \"import sys; sys.path.insert(0,'.'); import main; print('Import OK')\" 2>&1"
        )
        result = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(self.run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            universal_newlines=True,
        )
        if result.returncode != 0:
            errors = result.stdout or "Unknown import error"
            self._log(f"Import error: {errors[:200]}")
            with open(self.run_log, "w") as f:
                f.write(f"PRE-CHECK IMPORT FAILED:\n{errors}\n")
            return False, errors
        return True, ""

    def _run_experiment(self) -> tuple:
        """
        Execute run.sh in conda environment.
        Returns (success, error_message)
        """
        try:
            cmd = (
                f"source {self.conda_base}/etc/profile.d/conda.sh && "
                f"conda activate {self.conda_env} && "
                f"bash {self.run_dir / 'run.sh'}"
            )
            with open(self.run_log, "w") as log_file:
                exp_result = subprocess.run(
                    ["bash", "-c", cmd],
                    cwd=str(ROOT_DIR),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    timeout=self.experiment_timeout,
                )
            if exp_result.returncode != 0:
                return False, f"Experiment exited with code {exp_result.returncode}"
            return True, ""
        except subprocess.TimeoutExpired:
            return False, f"Experiment timeout after {self.experiment_timeout}s"
        except Exception as e:
            return False, f"Experiment error: {e}"

    # ------------------------------------------------------------------
    # Auto-generation fallback
    # ------------------------------------------------------------------

    def _auto_generate_missing_files(self):
        """Auto-generate missing simple files when CC times out but main.py exists."""
        run_dir = self.run_dir
        art_dir = self.art_dir

        if not (run_dir / "run.sh").exists() and (run_dir / "main.py").exists():
            content = (
                "#!/bin/bash\n"
                "set -e\n"
                'echo "Starting experiment at $(date)"\n'
                'echo "Working directory: $(pwd)"\n\n'
                f"cd {run_dir}\n\n"
                "python3 main.py 2>&1 | tee artifacts/logs/experiment.log\n\n"
                'echo "Experiment completed at $(date)"\n'
            )
            (run_dir / "run.sh").write_text(content)
            (run_dir / "run.sh").chmod(0o755)
            self._log("Auto-generated run.sh")

        if not (run_dir / "spec.yaml").exists():
            content = (
                "experiment:\n"
                "  name: auto-generated\n"
                "  description: Experiment spec auto-generated after timeout\n"
                "  status: partial\n"
            )
            (run_dir / "spec.yaml").write_text(content)
            self._log("Auto-generated spec.yaml")

        if not (art_dir / "summary.md").exists():
            content = (
                "# Experiment Summary\n\n"
                "> Note: This summary was auto-generated because Claude Code timed out.\n\n"
                "## Status\n"
                "- Code generation: Partial (timeout)\n"
                "- Experiment execution: Pending\n"
            )
            (art_dir / "summary.md").write_text(content)
            self._log("Auto-generated summary.md")

        if not (art_dir / "report.qmd").exists():
            content = (
                '---\ntitle: "Experiment Report"\nformat:\n  pdf:\n    toc: true\n  gfm: default\n---\n\n'
                "# Experiment Report\n\n"
                "> Note: Auto-generated after timeout.\n\n"
                "## Results\n\nPlease refer to artifacts directory.\n"
            )
            (art_dir / "report.qmd").write_text(content)
            self._log("Auto-generated report.qmd")

    # ------------------------------------------------------------------
    # Fix prompt builders
    # ------------------------------------------------------------------

    def _build_missing_files_prompt(self, missing: list, original_prompt: str) -> str:
        return (
            f"RETRY REQUEST - Previous attempt failed to create required files.\n\n"
            f"Missing files: {missing}\n\n"
            f"Please create ALL required files for the experiment. The original request was:\n"
            f"{original_prompt}\n\n"
            f"IMPORTANT: Make sure to create ALL files including spec.yaml, run.sh, main.py, "
            f"artifacts/summary.md, and artifacts/report.qmd"
        )

    def _build_syntax_fix_prompt(self, compile_errors: str) -> str:
        # Extract file context around the error
        error_file_content = ""
        match = re.search(r'File "([^"]+)", line (\d+)', compile_errors)
        if match:
            err_file, err_line = match.group(1), int(match.group(2))
            try:
                with open(err_file, "r") as ef:
                    lines = ef.readlines()
                    start = max(0, err_line - 5)
                    end = min(len(lines), err_line + 5)
                    error_file_content = f"\nFile content around error ({err_file} lines {start+1}-{end}):\n"
                    for i in range(start, end):
                        marker = ">>> " if i == err_line - 1 else "    "
                        error_file_content += f"{marker}{i+1}: {lines[i]}"
            except Exception:
                pass

        return (
            f"FIX REQUEST - SYNTAX ERROR in generated code.\n\n"
            f"ERROR:\n{compile_errors[:2000]}\n"
            f"{error_file_content}\n\n"
            f"IMPORTANT: This is a SYNTAX error (truncated line, missing parenthesis, etc).\n"
            f"Read the file carefully and fix the broken syntax.\n"
            f"Also use built-in types (dict, list) instead of typing.Dict/List for Python 3.10+.\n\n"
            f"Fix the code in runs/{self.run_dir.relative_to(ROOT_DIR)}/"
        )

    def _build_import_fix_prompt(self, import_errors: str) -> str:
        return (
            f"FIX REQUEST - IMPORT ERROR in generated code.\n\n"
            f"ERROR:\n{import_errors[:2000]}\n\n"
            f"IMPORTANT: This is an import error. Common causes:\n"
            f"1. Class/function name in import doesn't match the actual name in the target file\n"
            f"2. Missing __init__.py re-exports\n"
            f"3. Circular imports between modules\n"
            f"4. Using typing.Dict/List instead of built-in dict/list\n\n"
            f"Fix the code in runs/{self.run_dir.relative_to(ROOT_DIR)}/"
        )

    def _build_runtime_fix_prompt(self, run_error: str, original_prompt: str) -> str:
        # Read run log for details
        log_content = ""
        if self.run_log.exists():
            try:
                log_content = self.run_log.read_text(encoding="utf-8", errors="ignore")[-3000:]
            except Exception:
                pass
        return (
            f"FIX REQUEST - RUNTIME ERROR during experiment execution.\n\n"
            f"ERROR: {run_error}\n\n"
            f"RUN LOG (last 3000 chars):\n{log_content}\n\n"
            f"Please fix the code in runs/{self.run_dir.relative_to(ROOT_DIR)}/\n\n"
            f"The original request was:\n{original_prompt}"
        )

    def _build_verification_fix_prompt(self, verification: dict, original_prompt: str) -> str:
        error_details = "\n".join(verification["errors"])
        missing_artifacts = "\n".join(verification["missing_artifacts"])
        return (
            f"FIX REQUEST - The experiment code has errors that need to be fixed.\n\n"
            f"ERRORS FOUND:\n{error_details}\n\n"
            f"MISSING ARTIFACTS:\n{missing_artifacts}\n\n"
            f"ERROR LOG CONTENT:\n{verification['error_log_content']}\n\n"
            f"Please fix the code in runs/{self.run_dir.relative_to(ROOT_DIR)}/ to resolve these errors.\n"
            f"Common issues to check:\n"
            f"1. Import statements - only use installed packages\n"
            f"2. Function arguments - make sure types match\n"
            f"3. Variable names - ensure consistency throughout the code\n"
            f"4. Make sure figures are saved to artifacts/figures/ and tables to artifacts/tables/\n\n"
            f"The original request was:\n{original_prompt}"
        )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        print(f"[CCExecutor] {msg}", flush=True)
