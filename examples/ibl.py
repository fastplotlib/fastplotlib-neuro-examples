from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

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
        vid_path = parent_path.joinpath(
            subject, session, "raw_video_data", f"_iblrig_{camera}Camera.raw.mp4"
        )
        if gpu_context:
            ctx = gpu(0)
        else:
            ctx = cpu(0)

        self._array = LazyVideo(vid_path, ctx=ctx)

        timings_path = parent_path.joinpath(
            subject, session, "001", "alf", f"_ibl_{camera}Camera.times.npy"
        )
        self._timings = np.load(timings_path)

        self._tracks = pd.read_parquet(
            parent_path.joinpath(
                subject, session, "001", "alf", f"_ibl_{camera}Camera.lightningPose.pqt"
            )
        )

        df_paws = pd.read_parquet(
            parent_path.joinpath(
                subject,
                session,
                "001",
                "alf",
                "lightningaction",
                f"_ibl_{camera}Camera.pawstates.pqt",
            )
        )

        # only for Left paw:
        n_samples = 100
        L_cols = [
            c
            for c in df_paws.columns
            if c.startswith("paw_l") and not c.endswith("ens_var")
        ]

        R_cols = [
            c
            for c in df_paws.columns
            if c.startswith("paw_r") and not c.endswith("ens_var")
        ]

        L_probs = df_paws[L_cols].to_numpy()
        R_probs = df_paws[R_cols].to_numpy()
        L_probs = L_probs / L_probs.sum(axis=1, keepdims=True)
        R_probs = R_probs / R_probs.sum(axis=1, keepdims=True)

        n_rows = L_probs.shape[0]
        n_states = L_probs.shape[1]

        L_expanded = np.zeros((n_samples, n_rows), dtype=int)
        L_probs = np.nan_to_num(L_probs)
        L_probs = np.clip(L_probs, 0, None)
        L_probs = L_probs / L_probs.sum(axis=1, keepdims=True)

        for i in range(n_rows):

            probs = L_probs[i]

            counts = np.floor(probs * n_samples).astype(int)

            remainder = n_samples - counts.sum()

            # distribute remainder to largest fractional parts
            fractional = (probs * n_samples) - np.floor(probs * n_samples)
            order = np.argsort(fractional)[::-1]

            for j in range(remainder):
                counts[order[j]] += 1

            expanded = np.repeat(np.arange(n_states), counts)

            L_expanded[:, i] = expanded

        self._ethogram_prop = np.dstack(
            [
                np.broadcast_to(self.timings[None], (L_expanded.shape[0], L_expanded.shape[1])),
                L_expanded,
            ]
        )

    @property
    def array(self) -> LazyVideo:
        return self._array

    @property
    def timings(self) -> np.ndarray:
        return self._timings

    @property
    def tracks(self) -> pd.DataFrame:
        return self._tracks

    @property
    def ethogram_prop(self) -> np.ndarray:
        return self._ethogram_prop
