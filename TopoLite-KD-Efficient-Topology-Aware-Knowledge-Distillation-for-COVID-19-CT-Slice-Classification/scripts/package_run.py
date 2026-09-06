from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a checksum-audited ZIP archive of one completed experiment run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    manifest_path = run_dir / "artifacts_manifest.sha256"
    lines = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path != manifest_path:
            lines.append(f"{sha256(path)}  {path.relative_to(run_dir)}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    output = Path(args.output).resolve() if args.output else run_dir.parent / f"{run_dir.parent.name}_{run_dir.name}"
    archive = shutil.make_archive(str(output), "zip", root_dir=run_dir.parent, base_dir=run_dir.name)
    print(f"Created: {archive}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
