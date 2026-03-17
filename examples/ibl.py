from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lazyvideo import LazyVideo
from decord import gpu, cpu


class Video:
    def __init__(
        self,
        parent_path: Path,
        subject: str,
        session: str,
        camera: Literal["left", "right"],
        gpu_context: bool = True,
    ):
        vid_path = parent_path.joinpath(subject, session, "raw_video_data", f"_iblrig_{camera}Camera.raw.mp4")
        if gpu_context:
            ctx = gpu(0)
        else:
            ctx = cpu(0)

        self._array = LazyVideo(vid_path, ctx=ctx)

        timings_path = parent_path.joinpath(
            subject, session, "001", "alf", f"_ibl_{camera}Camera.times.npy"
        )
        self._timings = np.load(timings_path)

    @property
    def array(self) -> LazyVideo:
        return self._array

    @property
    def timings(self) -> np.ndarray:
        return self._timings
