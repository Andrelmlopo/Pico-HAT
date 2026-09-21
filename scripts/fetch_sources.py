"""Fetch the source revisions used by the Pico-HAT adapters."""

import argparse
import subprocess
from pathlib import Path

SOURCES = {
    "PicoPose": (
        "https://github.com/foollh/PicoPose.git",
        "543f8fe0dc8fe014a602113977fc034bd9b4578d",
    ),
    "DROID-SLAM": (
        "https://github.com/princeton-vl/DROID-SLAM.git",
        "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99",
    ),
}


def fetch(root):
    root.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in SOURCES.items():
        target = root / name
        if target.exists():
            current = subprocess.check_output(
                ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
            ).strip()
            dirty = subprocess.check_output(
                ["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"],
                text=True,
            )
            if current != revision or dirty:
                raise RuntimeError(
                    f"{target} differs from the expected source. Use a fresh directory."
                )
        else:
            subprocess.run(["git", "init", str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
            subprocess.run(
                ["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], check=True
            )
            subprocess.run(
                ["git", "-C", str(target), "checkout", "--detach", "FETCH_HEAD"], check=True
            )
        if name == "DROID-SLAM":
            subprocess.run(
                ["git", "-C", str(target), "submodule", "update", "--init", "--recursive"],
                check=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("third_party"))
    fetch(parser.parse_args().out.resolve())
