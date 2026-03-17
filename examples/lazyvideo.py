import os

# decord uses up all the RAM otherwise
os.environ["DECORD_EOF_RETRY_MAX"] = "20480"

import numpy as np
from pathlib import Path

from decord import VideoReader
from decord import gpu as gpu_context
from abc import ABC, abstractmethod
from typing import *
from warnings import warn

# Some stuff I copied from mesmerize-core for lazy-loading the video files using decord

from copy import copy as copyf

slice_or_int_or_range = Union[int, slice, range]


class LazyArray(ABC):
    """
    Base class for arrays that exhibit lazy computation upon indexing
    """

    def __array__(self, dtype=None, copy=None):
        if copy:
            return copyf(self)

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

    # @lru_cache(maxsize=32)
    def __getitem__(self, indices: tuple[slice, ...]) -> np.ndarray:
        # indices can be a tuple of slice | Ellipsis

        # apply to frame index, then remaining dims
        a = self._video_reader[indices[0]].asnumpy()[indices[1:]]
        # self._video_reader.seek(0)
        return a