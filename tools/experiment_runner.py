#!/usr/bin/env python3
"""
Experiment Runner - middleware that orchestrates experiment lifecycle.

This module connects any input channel (email, CLI, API) to the CC executor.
It handles:
1. Run directory setup
2. User data file handling
3. Prompt construction
4. Invoking CCExecutor
5. Report rendering (Quarto)
6. Artifact packing (zip)

Usage:
    from tools.experiment_runner import ExperimentRunner
    runner = ExperimentRunner()
    result = runner.run("Compare SVM vs LSTM", sender="user@example.com")
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

from tools.cc_executor import CCExecutor, ExecutionResult


ROOT_DIR = Path(__file__).parent.parent.resolve()


@dataclass
class RunResult:
    """Final result of an experiment run, including all paths and status."""
    run_dir: Path = None
    art_dir: Path = None
    success: bool = False
    error_msg: str = ""
    # Pass-through from CCExecutor
    execution: ExecutionResult = None


def sanitize_email_for_path(email_addr: str) -> str:
    """Convert email to safe directory name."""
    safe = email_addr.lower().replace("@", "_at_").replace(".", "_")
    return "".join(c if c.isalnum() or c == "_" else "" for c in safe)


class ExperimentRunner:
    """
    Middleware that sets up the experiment environment and delegates
    execution to CCExecutor.
    
    Attributes:
        max_retries: Passed to CCExecutor (default from env or 2)
        claude_timeout: Passed to CCExecutor (default 1200)
        experiment_timeout: Passed to CCExecutor (default 3600)
    """

    def __init__(
        self,
        max_retries: int = None,
        claude_timeout: int = 1200,
        experiment_timeout: int = 3600,
    ):
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("MAX_RETRIES", "2"))
        self.claude_timeout = claude_timeout
        self.experiment_timeout = experiment_timeout

    def run(
        self,
        request: str,
        sender: str = "local_test",
        data_files: list = None,
    ) -> RunResult:
        """
        Run a full experiment from request text to packaged artifacts.
        
        Args:
            request: Natural language experiment request
            sender: Identifier for the requester (email or username)
            data_files: Optional list of Path objects to user-uploaded data files
            
        Returns:
            RunResult with run_dir, success, error_msg, and execution details
        """
        # --- Setup directories ---
        user_dir = sanitize_email_for_path(sender)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}"
        run_dir = ROOT_DIR / "runs" / user_dir / run_id
        art_dir = run_dir / "artifacts"
        data_dir = run_dir / "data" / "raw"

        art_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        (art_dir / "figures").mkdir(exist_ok=True)
        (art_dir / "tables").mkdir(exist_ok=True)
        (art_dir / "logs").mkdir(exist_ok=True)

        # --- Copy user data files ---
        data_file_info = self._copy_data_files(data_files, data_dir) if data_files else ""

        # --- Build prompt ---
        full_run_path = f"{user_dir}/{run_id}"
        prompt = self._build_prompt(request, full_run_path, data_file_info, data_files)

        self._log(f"Run directory: {run_dir}")
        self._log(f"Running Claude Code with auto-retry (max {self.max_retries} retries)...")

        # --- Execute via CCExecutor ---
        try:
            executor = CCExecutor(
                run_dir=run_dir,
                art_dir=art_dir,
                max_retries=self.max_retries,
                claude_timeout=self.claude_timeout,
                experiment_timeout=self.experiment_timeout,
            )
            execution = executor.execute(prompt)

            if not execution.success:
                return RunResult(
                    run_dir=run_dir, art_dir=art_dir,
                    success=False, error_msg=execution.error_msg,
                    execution=execution,
                )

            # --- Post-processing: render report + pack ---
            self._render_report(art_dir)
            self._pack_artifacts(run_dir)

            return RunResult(
                run_dir=run_dir, art_dir=art_dir,
                success=True, error_msg="",
                execution=execution,
            )

        except Exception as e:
            return RunResult(
                run_dir=run_dir, art_dir=art_dir,
                success=False, error_msg=f"Error: {e}",
            )

    # ------------------------------------------------------------------
    # Data file handling
    # ------------------------------------------------------------------

    def _copy_data_files(self, data_files: list, data_dir: Path) -> str:
        """Copy user data files to run directory and generate preview info."""
        data_file_info = ""
        for src_file in data_files:
            src_file = Path(src_file)
            if not src_file.exists():
                continue
            dst_file = data_dir / src_file.name
            shutil.copy2(src_file, dst_file)
            self._log(f"Copied {src_file.name} to {dst_file}")

            # Generate data preview
            try:
                import pandas as pd
                if src_file.suffix.lower() == ".csv":
                    df = pd.read_csv(src_file, nrows=5)
                elif src_file.suffix.lower() in [".xlsx", ".xls"]:
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
                self._log(f"Could not preview data file {src_file.name}: {e}")
                data_file_info += f"\n\nData file: {src_file.name} (preview unavailable)\n"

        return data_file_info

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self, request: str, full_run_path: str, data_file_info: str, data_files: list) -> str:
        """Build the prompt for Claude Code."""
        data_section = ""
        if data_files:
            data_section = (
                f"\n\nUser has uploaded data files to runs/{full_run_path}/data/raw/:\n"
                f"{data_file_info}\n\n"
                f"Please analyze this data and design an appropriate experiment/analysis pipeline.\n"
                f"The experiment should:\n"
                f"1. Load data from data/raw/ directory\n"
                f"2. Perform exploratory data analysis (EDA)\n"
                f"3. Design and implement appropriate models/analyses based on data characteristics\n"
                f"4. Generate comprehensive visualizations\n"
                f"5. Save all results to artifacts/\n"
            )

        prompt = (
            f"User request: {request}\n"
            f"{data_section}\n"
            f"Please create an engineering-grade experiment under runs/{full_run_path}/ with the following structure:\n"
            f"1) spec.yaml: structured experiment design (data, model, hyperparams, seeds, metrics, statistical tests, artifact list)\n"
            f"2) main.py: main entry point that controls all hyperparameters\n"
            f"3) config/: configuration module with config.py and default_config.yaml\n"
            f"4) src/: modular source code (data/, models/, training/, visualization/)\n"
            f"5) run.sh: executable script that runs main.py\n"
            f"6) artifacts/summary.md: a concise summary (metrics, conclusions, next steps)\n"
            f"7) artifacts/report.qmd: based on templates/report.qmd, fill in this experiment's content\n\n"
            f"IMPORTANT:\n"
            f"- All visualizations must be dynamically planned based on experiment type\n"
            f"- Generate figures to artifacts/figures/ and tables to artifacts/tables/\n"
            f"- All outputs must be written to runs/{full_run_path}/\n"
            f"- run.sh must write logs to artifacts/logs/"
        )
        return prompt

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _render_report(self, art_dir: Path):
        """Render Quarto report to PDF and GFM."""
        report_qmd = art_dir / "report.qmd"
        if not report_qmd.exists():
            return

        for fmt in ["pdf", "gfm"]:
            try:
                subprocess.run(
                    ["quarto", "render", str(report_qmd), "--to", fmt],
                    cwd=str(ROOT_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=300,
                )
            except Exception as e:
                self._log(f"Report render ({fmt}) failed: {e}")

    def _pack_artifacts(self, run_dir: Path):
        """Pack artifacts into a zip file."""
        try:
            subprocess.run(
                ["python3", str(ROOT_DIR / "tools" / "pack_run.py"), str(run_dir)],
                cwd=str(ROOT_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
            )
        except Exception as e:
            self._log(f"Pack failed: {e}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, msg: str):
        print(f"[ExperimentRunner] {msg}", flush=True)
