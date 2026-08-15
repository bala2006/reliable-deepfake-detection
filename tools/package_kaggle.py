"""Package everything Kaggle needs into one self-contained zip.

The zip contains:
  * rlroinet/  (source code)
  * requirements.txt, pyproject.toml
  * outputs/models/best_model.pt, xception_best.pth  (frozen verdict backbones)
  * data/FaceForensics++/.rlroinet_index.json plus ONLY the face/mask/landmark
    files that manifest references (paths stay relative so cfg.resolve_paths
    and the manifest loader work unchanged).

Usage:
    python tools/package_kaggle.py [out.zip]

Upload the zip to Google Drive ("Anyone with the link"), then in Kaggle:
    !pip install --upgrade gdown
    !gdown https://drive.google.com/file/d/<ID>/view --fuzzy
    with zipfile.ZipFile('<out.zip>') as z: z.extractall('/kaggle/working/')
"""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FFPP = ROOT / "data" / "FaceForensics++"
MANIFEST = FFPP / ".rlroinet_index.json"


def main() -> int:
    out_name = sys.argv[1] if len(sys.argv) > 1 else "rlroinet_kaggle.zip"
    out_path = ROOT / out_name
    staging = ROOT / ".kaggle_staging"

    if staging.exists():
        shutil.rmtree(staging)

    def stage(rel: Path, kind: str):
        """Copy a single file into the staging tree keeping its repo-relative path."""
        src = ROOT / rel
        if not src.exists():
            print(f"WARN missing {kind}: {rel}")
            return
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    # code
    for p in sorted((ROOT / "rlroinet").rglob("*.py")):
        stage(p.relative_to(ROOT), "code")
    stage(Path("requirements.txt"), "code")
    stage(Path("pyproject.toml"), "code")

    # frozen verdict weights
    for w in ("best_model.pt", "xception_best.pth"):
        stage(Path("outputs/models") / w, "weights")

    # manifest + referenced data
    if not MANIFEST.exists():
        print("ERROR: manifest not found; refusing to package")
        return 1
    stage(Path("data/FaceForensics++/.rlroinet_index.json"), "manifest")
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    items = payload.get("items", [])
    n_files = 0
    for item in items:
        for key in ("face", "mask", "landmark"):
            rel = item.get(key)
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                stage(Path("data/FaceForensics++") / path, "data")
                n_files += 1
    print(f"staged {n_files} data files for {len(items)} manifest items")

    if out_path.exists():
        out_path.unlink()
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src in sorted(staging.rglob("*")):
            if src.is_file():
                zf.write(src, src.relative_to(staging))
    size_mb = out_path.stat().st_size / 1e6
    print(f"wrote {out_path.name} ({size_mb:.1f} MB)")
    shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
