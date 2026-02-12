#!/usr/bin/env python3
"""
Local test script for cc-exp-runner.
Tests the ExperimentRunner middleware directly without email I/O.

Usage:
    python3 test_local.py "<experiment request>" [--email user] [--data file1.csv ...]
    
Examples:
    python3 test_local.py "Test SVM classifier"
    python3 test_local.py "Analyze my data" --email test@example.com --data data.csv
    python3 test_local.py "Build a CNN for image classification" --data train.csv
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Local test script for OpenLab experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 test_local.py "Test SVM classifier"
    python3 test_local.py "Analyze my data" --email test@example.com --data data.csv
    python3 test_local.py "Build a CNN" --data train.csv test.csv
    python3 test_local.py "Test SVM" --max-retries 1 --claude-timeout 600
        """
    )
    parser.add_argument("request", help="Experiment request (natural language description)")
    parser.add_argument("--email", "-e", default="local_test_at_localhost", 
                        help="Test sender identifier (default: local_test_at_localhost)")
    parser.add_argument("--data", "-d", nargs="+", default=[], 
                        help="Data files to upload (CSV, Excel, etc.)")
    parser.add_argument("--max-retries", type=int, default=None,
                        help="Max CC retry attempts (default: from env or 2)")
    parser.add_argument("--claude-timeout", type=int, default=1200,
                        help="Claude Code timeout in seconds (default: 1200)")
    parser.add_argument("--experiment-timeout", type=int, default=3600,
                        help="Experiment execution timeout in seconds (default: 3600)")
    
    args = parser.parse_args()
    
    # Validate data files
    data_files = []
    for f in args.data:
        path = Path(f)
        if not path.exists():
            print(f"[ERROR] Data file not found: {f}")
            sys.exit(1)
        data_files.append(path)
    
    print("=" * 60)
    print("Local Test for OpenLab (ExperimentRunner)")
    print("=" * 60)
    print(f"Request: {args.request}")
    print(f"Sender: {args.email}")
    print(f"Claude timeout: {args.claude_timeout}s")
    print(f"Experiment timeout: {args.experiment_timeout}s")
    if data_files:
        print(f"Data files: {[str(f) for f in data_files]}")
    print("")
    
    # Create runner with CLI-specified parameters
    runner = ExperimentRunner(
        max_retries=args.max_retries,
        claude_timeout=args.claude_timeout,
        experiment_timeout=args.experiment_timeout,
    )
    
    print("[RUN] Starting experiment via ExperimentRunner...")
    result = runner.run(
        args.request, 
        sender=args.email, 
        data_files=data_files if data_files else None,
    )
    
    print("")
    print("=" * 60)
    print("Result:")
    print(f"  Run directory: {result.run_dir}")
    print(f"  Success: {result.success}")
    if result.error_msg:
        print(f"  Error: {result.error_msg}")
    if result.execution:
        print(f"  Attempts: {result.execution.attempt_count + 1}")
    print("=" * 60)
    
    if result.success and result.run_dir:
        print("")
        print("Project structure generated:")
        for item in sorted(result.run_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(result.run_dir)
                size = item.stat().st_size
                print(f"  - {rel_path} ({size} bytes)")
        
        print("")
        print("Key artifacts:")
        art_dir = result.run_dir / "artifacts"
        for subdir in ["figures", "tables"]:
            subdir_path = art_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                if files:
                    print(f"  {subdir}/: {len(files)} files")
                    for f in files[:5]:
                        print(f"    - {f.name}")
                    if len(files) > 5:
                        print(f"    ... and {len(files) - 5} more")

if __name__ == "__main__":
    main()
