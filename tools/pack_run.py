#!/usr/bin/env python3
import sys
from pathlib import Path
import zipfile

def main():
    if len(sys.argv) < 2:
        print("Usage: pack_run.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    artifacts = run_dir / "artifacts"
    out_zip = artifacts / "run.zip"
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in artifacts.rglob("*"):
            if p.is_file() and p.name != "run.zip":
                z.write(p, arcname=str(p.relative_to(artifacts)))
        # Also include spec + run script for reproducibility
        for extra in ["spec.yaml", "run.sh"]:
            ep = run_dir / extra
            if ep.exists():
                z.write(ep, arcname=extra)

    print(str(out_zip))

if __name__ == "__main__":
    main()
