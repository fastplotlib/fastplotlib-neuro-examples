import numpy as np
from functools import lru_cache
from pathlib import Path
import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import NDPositions, ndp_extras, NDImage
import pandas as pd
from decord import VideoReader
from abc import ABC, abstractmethod
from typing import *
import os
from warnings import warn
import cmap
import pygfx


# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "128"

# Some stuff I copied from mesmerize-core for lazy-loading the video files using decord

from copy import copy

slice_or_int_or_range = Union[int, slice, range]


class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """

    def __array__(self, dtype=None, copy=None):
        if copy:
            return copy(self)

        return self

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        raise NotImplementedError

    def __array_function__(self, func, types, *args, **kwargs):
        raise NotImplementedError

    @property
    @abstractmethod
    def dtype(self) -> str:
        """
        str
            data type
        """
        pass

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """
        Tuple[int]
            (n_frames, dims_x, dims_y)
        """
        pass

    @property
    @abstractmethod
    def min(self) -> float:
        """
        float
            min value of the array if it were fully computed
        """
        pass

    @property
    @abstractmethod
    def max(self) -> float:
        """
        float
            max value of the array if it were fully computed
        """
        pass

    @property
    def ndim(self) -> int:
        """
        int
            Number of dimensions
        """
        return len(self.shape)

    @property
    def nbytes(self) -> int:
        """
        int
            number of bytes for the array if it were fully computed
        """
        return np.prod(self.shape + (np.dtype(self.dtype).itemsize,), dtype=np.int64)

    def __getitem__(self, indices: tuple[slice, ...]) -> np.ndarray:
        # indices can be a tuple of slice | Ellipsis
        # need to accoutn for Ellipsis as the last object in the tuple
        pass

    def __repr__(self):
        return (
            f"{self.__class__.__name__} @{hex(id(self))}\n"
            f"{self.__class__.__doc__}\n"
            f"Frames are computed only upon indexing\n"
            f"shape [frames, x, y]: {self.shape}\n"
        )


class LazyVideo(LazyArray):
    def __init__(
            self,
            path: Union[Path, str],
            min_max: Tuple[int, int] = None,
            **kwargs,
    ):
        """
        LazyVideo reader, basically just a wrapper for ``decord.VideoReader``.
        Should support opening anything that decord can open.

        **Important:** requires ``decord`` to be installed: https://github.com/dmlc/decord

        Parameters
        ----------
        path: Path or str
            path to video file

        min_max: Tuple[int, int], optional
            min and max vals of the entire video, uses min and max of 10th frame if not provided

        as_grayscale: bool, optional
            return grayscale frames upon slicing

        rgb_weights: Tuple[float, float, float], optional
            (r, g, b) weights used for grayscale conversion if ``as_graycale`` is ``True``.
            default is (0.299, 0.587, 0.114)

        kwargs
            passed to ``decord.VideoReader``

        Examples
        --------

        Lazy loading with CPU

        .. code-block:: python

            from mesmerize_core.arrays import LazyVideo

            vid = LazyVideo("path/to/video.mp4")

            # use fpl to visualize

            import fastplotlib as fpl

            iw = fpl.ImageWidget(vid)
            iw.show()


        Lazy loading with GPU, decord must be compiled with CUDA options to use this

        .. code-block:: python

            from decord import gpu
            from mesmerize_core.arrays import LazyVideo

            gpu_context = gpu(0)

            vid = LazyVideo("path/to/video.mp4", ctx=gpu_context)

        """
        self._video_reader = VideoReader(str(path), **kwargs)

        try:
            frame0 = self._video_reader[10].asnumpy()
            self._video_reader.seek(0)
        except IndexError:
            frame0 = self._video_reader[0].asnumpy()
            self._video_reader.seek(0)

        self._shape = (self._video_reader._num_frame, *frame0.shape)

        self._dtype = frame0.dtype

        if min_max is not None:
            self._min, self._max = min_max
        else:
            self._min = frame0.min()
            self._max = frame0.max()

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        """[n_frames, m, n, 3 | 4]"""
        return self._shape

    @property
    def min(self) -> float:
        warn("min not implemented for LazyTiff, returning min of 0th index")
        return self._min

    @property
    def max(self) -> float:
        warn("max not implemented for LazyTiff, returning min of 0th index")
        return self._max

    @lru_cache(maxsize=32)
    def __getitem__(self, indices: tuple[slice, ...]) -> np.ndarray:
        # indices can be a tuple of slice | Ellipsis

        # apply to frame index, then remaining dims
        a = self._video_reader[indices[0]].asnumpy()[indices[1:]]
        self._video_reader.seek(0)
        return a


# path with all the files for this session
parent_path = Path("/home/kushal/data/kcenia/")

# load video file as a lazy array
vid = LazyVideo(parent_path.joinpath("mouse1.mp4"))

vid_indexing = np.load(parent_path.joinpath("camera_times.npy"))

df_tracks = pd.read_csv(parent_path.joinpath("video_data.csv"))

keypoints = [
    "nose_tip",
    "pupil_top_r",
    "pupil_bottom_r",
    "pupil_right_r",
    "pupil_left_r",
    "paw_l",
    "paw_r",
    "tongue_end_l",
    "tongue_end_r",
]

keypoints_cols = np.array([(f"{k}_x", f"{k}_y", f"{k}_likelihood") for k in keypoints])
likelihood_cols = keypoints_cols[:, -1]
for k in keypoints_cols.ravel():
    if k not in df_tracks.columns:
        print(k)


df_paws = pd.read_parquet(parent_path.joinpath("pawstates_LA.pqt"))
df_paws

df_paws["times"] = df_tracks.times

#only for Left paw:
n_samples = 100
L_cols = [c for c in df_paws.columns
          if c.startswith("paw_l") and not c.endswith("ens_var")]

R_cols = [c for c in df_paws.columns
          if c.startswith("paw_r") and not c.endswith("ens_var")]

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
paws = sorted(set("_".join(c.split("_")[:2]) for c in df_paws.columns))
paws = paws[0:2]

states = sorted(
    set(
        "_".join(c.split("_")[2:])
        for c in df_paws.columns
        if not c.endswith("_ens_var")
    )
)

states = states[1:6]
paw_cols = np.array([
    (
        f"{paw}_{state}",
        f"{paw}_{state}_ens_var"
    )
    for paw in paws
    for state in states
])
paw_state_cols = paw_cols[:, 0]
paw_ensvar_cols = paw_cols[:, 1]
xs = df_tracks["times"].values

ethogram_prop = np.dstack([np.broadcast_to(xs[None], (L_expanded.shape[0], L_expanded.shape[1])), L_expanded])

kp_colors = cmap.Colormap("tab10").lut(keypoints_cols.shape[0])[:keypoints_cols.shape[0]]
kp_colors[:, None, :].repeat(100, axis=1).shape

ll_l_data = df_tracks["tongue_end_l_likelihood"].values
ll_r_data = df_tracks["tongue_end_r_likelihood"].values

def alpha_using_likelihood(data, dw_slice: slice):
    # number of datapoints
    p = dw_slice.stop - dw_slice.start
    # [l, p, 4] array of colors
    new_colors = kp_colors[:, None, :].repeat(p, axis=1)
    # set alpha using likelihood of tongue
    new_colors[-2, :, -1] = ll_l_data[dw_slice]
    new_colors[-1, :, -1] = ll_r_data[dw_slice]
    return new_colors

# start, stop, step range for time
reference_range = {"time": (vid_indexing[0], vid_indexing[-1], 0.025)}

# Create an ND Widget with the reference dimensions
ndw = fpl.NDWidget(ref_ranges=reference_range, shape=(2, 3), size=(1000, 1000))

# add video as an nd image
ndw[0, 0].add_nd_image(
    vid,
    dims=("time", "m", "n", "rgb"),
    spatial_dims=("m", "n", "rgb"),
    rgb_dim=("rgb"),
    slider_dim_transforms={"time": vid_indexing},
    compute_histogram=False,
    name="vid",
)
ndw[0, 1].add_nd_image(
    vid,
    dims=("time", "m", "n", "rgb"),
    spatial_dims=("m", "n", "rgb"),
    rgb_dim=("rgb"),
    slider_dim_transforms={"time": vid_indexing},
    compute_histogram=False,
    name="vid",
)

ndw[0, 2].add_nd_image(
    vid,
    dims=("time", "m", "n", "rgb"),
    spatial_dims=("m", "n", "rgb"),
    rgb_dim=("rgb"),
    slider_dim_transforms={"time": vid_indexing},
    compute_histogram=False,
    name="vid",
)

# add behavior data as an nd scatter
nd_scatter = ndw[0, 0].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    slider_dim_transforms={"time": df_tracks["times"].values},
    name="keypoints",
    colors=alpha_using_likelihood,
    # datapoints_window_func = (np.mean, "xy", 3.0),
)

# # add behavior data as an nd scatter
nd_scatter2 = ndw[0, 1].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    slider_dim_transforms={"time": df_tracks["times"].values},
    name="keypoints",
)

# # add behavior data as an nd scatter
nd_scatter3 = ndw[0, 2].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    slider_dim_transforms={"time": df_tracks["times"].values},
    name="keypoints",
)

# add any other time-series data (likelihood, lighting-pose estimates, ethograms, etc.)
nd_ll = ndw[1, 0].add_nd_timeseries(
    df_tracks,
    ("l", "time", "d"),
    [("times", c) for c in likelihood_cols],  # must provide the "times" x-valuess,
    processor=ndp_extras.NDPP_Pandas,
    display_window=5,
    slider_dim_transforms={"time": df_tracks["times"].values},
    graphic_type=fpl.ImageGraphic,
    name="likelihood",
)

nd_eth = ndw[1, 1].add_nd_timeseries(
    ethogram_prop,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    display_window=5,
    slider_dim_transforms={"time": df_tracks["times"].values},
    graphic_type=fpl.ImageGraphic,
    name="ethogram",
)

c = pygfx.cm.create_colormap(cmap.Colormap(["white", "green", "orange", "red"]).lut(4), n=4)
nd_eth.graphic._material.map = c
nd_eth.graphic.vmin, nd_eth.graphic.vmax = 1, 4


def prob_tooltip(pick_info):
    # row col position of the cursor
    col, row = pick_info["index"]
    # current displayed image data value at this row, col position
    val = round(nd_eth.graphic.data[row, col])
    return {1: "still", 2: "lick", 3: "move", 4: "groom"}.get(val, "undefined")


nd_eth.graphic.tooltip_format = prob_tooltip

for subplot in [ndw.figure[1, 0], ndw.figure[1, 1]]:
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

cursor = fpl.Cursor()

cursor.add_subplot(ndw.figure[0, 0])
cursor.add_subplot(ndw.figure[0, 1])
cursor.add_subplot(ndw.figure[0, 2])

# a cmap for the scatter collection, each keypoint will get its own color
# There is a lot of fine-tuning you can do for scatter colors
for ng in [nd_scatter, nd_scatter2, nd_scatter3]:
    ng.graphic.cmap = "tab10"

    # change some properties of the scatter
    for g in ng.graphic:
        g.sizes = 7
        g.visible = True
        g.edge_width = 0.1

ndw.figure.show()
fpl.loop.run()
