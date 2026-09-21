import json
import os
import sys

import pytest

from pico_hat.runtime import read_runtime


def test_venv_interpreter_symlinks_keep_their_environment(tmp_path):
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    source = tmp_path / "source"
    source.mkdir()
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"fixture")
    data = {
        f"{name}_{key}": value
        for name in ("pico", "droid")
        for key, value in {
            "python": "venv/bin/python",
            "source": "source",
            "checkpoint": "weights.pth",
        }.items()
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(data))
    runtime = read_runtime(path)
    assert runtime["droid_python"] == os.path.abspath(interpreter)
    assert runtime["droid_python"] != str(interpreter.resolve())
    data["seed"] = -1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="seed"):
        read_runtime(path)
