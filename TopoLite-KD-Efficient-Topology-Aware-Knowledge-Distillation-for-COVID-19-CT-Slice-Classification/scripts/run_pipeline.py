from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from topokd.config import apply_overrides, load_config


def run(command: list[str]) -> None:
    print("\n===", " ".join(command), "===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute the reproducible manifest→topology→teacher→student pipeline.")
    parser.add_argument("--config", default="configs/topolite_kd.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--force-manifest", action="store_true")
    parser.add_argument("--skip-teacher-training", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = apply_overrides(load_config(config_path), args.set)
    common = ["--config", str(config_path)]
    for override in args.set:
        common.extend(["--set", override])

    manifest = Path(config["data"]["manifest"])
    if not manifest.exists() or args.force_manifest:
        command = [sys.executable, str(ROOT / "scripts" / "prepare_manifest.py"), *common]
        if args.force_manifest:
            command.append("--force")
        run(command)
    else:
        print(f"Using existing frozen manifest: {manifest.resolve()}")

    if bool(config["topology"].get("enabled", True)):
        run([sys.executable, str(ROOT / "scripts" / "build_tda_cache.py"), *common])

    kd_enabled = float(config["loss"].get("response_kd_weight", 0.0)) > 0 or float(config["loss"].get("feature_kd_weight", 0.0)) > 0
    if kd_enabled:
        teacher_checkpoint = Path(config["model"]["teacher"]["checkpoint"])
        if teacher_checkpoint.exists():
            print(f"Using existing teacher checkpoint: {teacher_checkpoint.resolve()}")
        elif args.skip_teacher_training:
            raise FileNotFoundError(
                f"KD is enabled and teacher checkpoint is missing: {teacher_checkpoint}. "
                "Remove --skip-teacher-training or attach the archived checkpoint."
            )
        else:
            teacher_command = [sys.executable, str(ROOT / "scripts" / "train_teacher.py"), *common]
            if args.device:
                teacher_command.extend(["--device", args.device])
            run(teacher_command)

    train_command = [sys.executable, str(ROOT / "scripts" / "train.py"), *common]
    if args.device:
        train_command.extend(["--device", args.device])
    run(train_command)


if __name__ == "__main__":
    main()
