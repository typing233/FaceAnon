"""Cross-platform build script for FaceAnon.

Usage:
    python packaging/build.py [--onedir]

Produces a standalone executable in dist/ using PyInstaller.
"""

import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_FILE = Path(__file__).resolve().parent / "faceanon.spec"


def main() -> int:
    onedir = "--onedir" in sys.argv

    if not (ROOT / "models" / "centerface.onnx").exists():
        print("Warning: models/centerface.onnx not found. The model will need to be")
        print("downloaded on first run unless you place it in the build output.")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath", str(ROOT / "dist"),
        "--workpath", str(ROOT / "build"),
    ]

    if onedir:
        cmd += [
            "--name", "faceanon",
            "--add-data", f"{ROOT / 'models' / 'centerface.onnx'}{os.pathsep}models",
            "--hidden-import", "onnxruntime",
            "--hidden-import", "scipy.optimize",
            "--hidden-import", "tqdm",
            "--onedir",
            str(ROOT / "faceanon" / "cli.py"),
        ]
    else:
        cmd += [str(SPEC_FILE)]

    print(f"Building for {platform.system()} ({platform.machine()})...")
    print(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print("Build failed!", file=sys.stderr)
        return 1

    output_path = ROOT / "dist" / "faceanon"
    if platform.system() == "Windows":
        output_path = output_path.with_suffix(".exe")

    if output_path.exists():
        print(f"\nBuild successful: {output_path}")
        print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    else:
        output_dir = ROOT / "dist" / "faceanon"
        if output_dir.is_dir():
            print(f"\nBuild successful (onedir): {output_dir}/")
        else:
            print(f"\nBuild output: {ROOT / 'dist'}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
