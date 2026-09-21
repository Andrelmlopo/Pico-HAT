"""Separate native GPU environments connected through local process pipes."""

import json
import os
import subprocess
from multiprocessing import Pipe
from pathlib import Path


def read_runtime(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    required = {
        "pico_python",
        "pico_source",
        "pico_checkpoint",
        "droid_python",
        "droid_source",
        "droid_checkpoint",
    }
    if not required <= data.keys() or set(data) - required - {
        "droid_buffer",
        "seed",
        "timeout_seconds",
    }:
        raise ValueError("runtime.json requires pico/droid python, source and checkpoint paths")
    for key in required:
        if not isinstance(data[key], str) or not data[key]:
            raise ValueError(f"{key} must be a nonempty path string")
        # Resolving a venv interpreter symlink would select its base Python and
        # silently discard the environment. Keep its absolute lexical path.
        data[key] = os.path.abspath(path.parent / data[key])
        item = Path(data[key])
        if not item.exists():
            raise ValueError(f"{key} does not exist: {item}")
        if key.endswith("source") and not item.is_dir():
            raise ValueError(f"{key} must be a source directory: {item}")
        if not key.endswith("source") and not item.is_file():
            raise ValueError(f"{key} must be a file: {item}")
        if key.endswith("python") and not os.access(item, os.X_OK):
            raise ValueError(f"{key} is not executable: {item}")
    data.setdefault("seed", 0)
    data.setdefault("droid_buffer", 512)
    data.setdefault("timeout_seconds", 600)
    if type(data["droid_buffer"]) is not int or data["droid_buffer"] < 16:
        raise ValueError("droid_buffer must be an integer of at least 16")
    if type(data["seed"]) is not int or not 0 <= data["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if type(data["timeout_seconds"]) is not int or data["timeout_seconds"] <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    return data


class Worker:
    def __init__(self, kind, runtime, sequence, templates, output):
        if os.name != "posix":
            raise RuntimeError("The GPU runner requires Linux")
        self.kind, self.timeout = kind, runtime["timeout_seconds"]
        self.log_path = Path(output) / f"{kind}.log"
        self.log = self.log_path.open("w")
        self.connection, child = Pipe()
        self.process = None
        env = dict(os.environ)
        # Both subprocesses load this installed package, independent of their cwd.
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            env[name] = "1"
        try:
            self.process = subprocess.Popen(
                [runtime[f"{kind}_python"], "-m", "pico_hat.worker", str(child.fileno())],
                pass_fds=[child.fileno()],
                env=env,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            child.close()
            self.connection.send(
                {
                    "kind": kind,
                    "runtime": runtime,
                    "sequence": str(sequence),
                    "templates": str(templates),
                }
            )
            self.receive()
        except BaseException:
            child.close()
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(self.timeout):
            raise RuntimeError(f"{self.kind} timed out. See {self.log_path}")
        try:
            message = self.connection.recv()
        except EOFError as error:
            raise RuntimeError(f"{self.kind} exited unexpectedly. See {self.log_path}") from error
        if message["status"] == "error":
            raise RuntimeError(f"{self.kind}: {message['error']}. See {self.log_path}")
        return message.get("value")

    def call(self, frame):
        self.connection.send(frame)
        return self.receive()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.connection.send(None)
                    self.process.wait(timeout=10)
                except (BrokenPipeError, EOFError, OSError, subprocess.TimeoutExpired):
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
        self.connection.close()
        self.log.close()
