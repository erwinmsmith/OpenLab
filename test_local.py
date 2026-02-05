#!/usr/bin/env python3
"""
Local test script for cc-exp-runner.
Simulates an email request without actually sending email.

Usage:
    python3 test_local.py "<experiment request>" [test_email] [--data file1.csv file2.csv ...]
    
Examples:
    python3 test_local.py "Test SVM classifier"
    python3 test_local.py "Analyze my data" test@example.com --data data.csv
    python3 test_local.py "Build a CNN for image classification" --data train.csv
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.email_listener import run_experiment, load_config, send_reply

def main():
    parser = argparse.ArgumentParser(
        description="Local test script for OpenLab experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 test_local.py "Test SVM classifier"
    python3 test_local.py "Analyze my data" --email test@example.com --data data.csv
    python3 test_local.py "Build a CNN" --data train.csv test.csv
        """
    )
    parser.add_argument("request", help="Experiment request (natural language description)")
    parser.add_argument("--email", "-e", default="local_test_at_localhost", 
                        help="Test email address (default: local_test_at_localhost)")
    parser.add_argument("--data", "-d", nargs="+", default=[], 
                        help="Data files to upload (CSV, Excel, etc.)")
    
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
    print("Local Test for OpenLab")
    print("=" * 60)
    print(f"Request: {args.request}")
    print(f"Test email: {args.email}")
    if data_files:
        print(f"Data files: {[str(f) for f in data_files]}")
    print("")
    
    print("[1/2] Running experiment...")
    run_dir, success, error_msg = run_experiment(
        args.request, 
        args.email, 
        data_files=data_files if data_files else None
    )
    
    print("")
    print("=" * 60)
    print("Result:")
    print(f"  Run directory: {run_dir}")
    print(f"  Success: {success}")
    if error_msg:
        print(f"  Error: {error_msg}")
    print("=" * 60)
    
    if success:
        print("")
        print("Project structure generated:")
        # Show directory structure
        for item in sorted(run_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(run_dir)
                size = item.stat().st_size
                print(f"  - {rel_path} ({size} bytes)")
        
        print("")
        print("Key artifacts:")
        art_dir = run_dir / "artifacts"
        for subdir in ["figures", "tables"]:
            subdir_path = art_dir / subdir
            if subdir_path.exists():
                files = list(subdir_path.glob("*"))
                if files:
                    print(f"  {subdir}/: {len(files)} files")
                    for f in files[:5]:  # Show first 5
                        print(f"    - {f.name}")
                    if len(files) > 5:
                        print(f"    ... and {len(files) - 5} more")

if __name__ == "__main__":
    main()
