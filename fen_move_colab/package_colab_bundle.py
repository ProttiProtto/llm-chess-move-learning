"""Create a clean Drive-uploadable copy of the Colab code bundle."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional


SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SOURCE_DIR.parent / "colab_drive_bundle"
EXCLUDED_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
EXCLUDED_FILES = {"one_legal_move_runtime.yaml", "all_legal_moves_runtime.yaml", "config_runtime.yaml"}


def _ignore(directory: str, names):
    ignored = set()
    for name in names:
        path = Path(directory) / name
        if name in EXCLUDED_DIRS or name in EXCLUDED_FILES or path.suffix == ".pyc":
            ignored.add(name)
    return ignored


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_bundle(output_dir: Path, overwrite: bool, raw_csv: Optional[Path] = None) -> Path:
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Bundle directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    shutil.copytree(SOURCE_DIR, output_dir, ignore=_ignore)
    if raw_csv is not None:
        raw_csv = raw_csv.resolve()
        if not raw_csv.is_file():
            raise FileNotFoundError(raw_csv)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(exist_ok=True)
        shutil.copy2(raw_csv, raw_dir / raw_csv.name)

    manifest = {
        "bundle_name": output_dir.name,
        "source_directory": str(SOURCE_DIR),
        "included_raw_csv": str(raw_csv) if raw_csv is not None else None,
        "files": {
            path.relative_to(output_dir).as_posix(): _sha256(path)
            for path in sorted(output_dir.rglob("*"))
            if path.is_file()
        },
    }
    with (output_dir / "BUNDLE_MANIFEST.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package the Colab source folder for Google Drive upload.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--raw-csv", type=Path, default=None, help="Optionally copy the Lichess CSV into bundle/raw/.")
    args = parser.parse_args()
    output_dir = package_bundle(args.output_dir.resolve(), args.overwrite, args.raw_csv)
    print(f"Created Drive-uploadable bundle: {output_dir}")


if __name__ == "__main__":
    main()
