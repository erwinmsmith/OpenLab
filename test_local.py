#!/usr/bin/env python3
"""
Local test script for cc-exp-runner.
Simulates an email request without actually sending email.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tools.email_listener import run_experiment, load_config, send_reply

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 test_local.py \"<experiment request>\" [test_email]")
        print("Example: python3 test_local.py \"Test SVM classifier\" test@example.com")
        sys.exit(2)
    
    request = sys.argv[1]
    test_email = sys.argv[2] if len(sys.argv) > 2 else "local_test_at_localhost"
    
    print("=" * 60)
    print("Local Test for cc-exp-runner")
    print("=" * 60)
    print("Request: {}".format(request))
    print("Test email: {}".format(test_email))
    print("")
    
    print("[1/2] Running experiment...")
    run_dir, success, error_msg = run_experiment(request, test_email)
    
    print("")
    print("=" * 60)
    print("Result:")
    print("  Run directory: {}".format(run_dir))
    print("  Success: {}".format(success))
    if error_msg:
        print("  Error: {}".format(error_msg))
    print("=" * 60)
    
    if success:
        print("")
        print("Artifacts generated:")
        art_dir = run_dir / "artifacts"
        for f in art_dir.rglob("*"):
            if f.is_file():
                print("  - {}".format(f.relative_to(run_dir)))

if __name__ == "__main__":
    main()
