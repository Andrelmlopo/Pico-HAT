"""Private local-worker entry point used by the sequence runner."""

import random
import sys
import traceback
from multiprocessing.connection import Connection

import numpy as np

from .sequence import Sequence


def main():
    connection = Connection(int(sys.argv[1]))
    try:
        request = connection.recv()
        runtime, kind = request["runtime"], request["kind"]
        import cv2
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in this worker environment")
        random.seed(runtime["seed"])
        np.random.seed(runtime["seed"])
        torch.manual_seed(runtime["seed"])
        cv2.setRNGSeed(runtime["seed"] % (2**31))
        cv2.setNumThreads(1)
        sequence = Sequence(request["sequence"])
        if kind == "pico":
            from .pico import PicoPose

            model = PicoPose(
                runtime["pico_source"], runtime["pico_checkpoint"], request["templates"]
            )
        elif kind == "droid":
            from .droid import DroidMotion

            model = DroidMotion(
                runtime["droid_source"],
                runtime["droid_checkpoint"],
                sequence,
                runtime["droid_buffer"],
            )
        else:
            raise ValueError(f"Unknown worker {kind}")
        torch.cuda.synchronize()
        connection.send({"status": "ready"})
        while True:
            index = connection.recv()
            if index is None:
                break
            if kind == "pico":
                mask = sequence.localization(index)
                if mask is None:
                    raise ValueError("PicoPose was called without a detection")
                result = model.predict(sequence.rgb(index), mask, sequence.K)
            else:
                result = model.step(index)
            torch.cuda.synchronize()
            # NumPy 2 pickle payloads require NumPy 2 in the receiver. Lists keep
            # the protocol compatible with PicoPose's NumPy 1.26 environment.
            value = [x.tolist() for x in result] if kind == "pico" else result.tolist()
            connection.send({"status": "ok", "value": value})
    except BaseException as error:
        traceback.print_exc()
        try:
            connection.send({"status": "error", "error": f"{type(error).__name__}: {error}"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
