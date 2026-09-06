from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete A0-A10 ablation matrix over one or more seeds.")
    parser.add_argument("--config-dir", default="configs/ablations")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1337, 2026])
    parser.add_argument("--pattern", default="*.yaml")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    configs = sorted((root / args.config_dir).glob(args.pattern))
    if not configs:
        raise FileNotFoundError(f"No configs found in {args.config_dir} matching {args.pattern}")
    failures = []
    for config in configs:
        for seed in args.seeds:
            command = [
                sys.executable,
                str(root / "scripts" / "train.py"),
                "--config",
                str(config),
                "--set",
                f"training.seed={seed}",
            ]
            print("\n===", " ".join(command), "===")
            result = subprocess.run(command, cwd=root, check=False)
            if result.returncode != 0:
                failures.append((config.name, seed, result.returncode))
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
    if failures:
        print("Failed runs:")
        for failure in failures:
            print(failure)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
