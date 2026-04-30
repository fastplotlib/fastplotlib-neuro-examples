import numpy as np
from spikeinterface import BaseRecording
from fastplotlib.widgets.nd_widget import NDPositionsProcessor


class NDSpikeInterfaceProcessor(NDPositionsProcessor):
    def __init__(
            self,
            data: BaseRecording,
            dims,  # just [l, time, d]
            # also [l, time, d], the middle "time" dim is both spatial & non-spatial since it's the n_datapoints dim
            spatial_dims,
            **kwargs,
    ):
        super().__init__(
            data=data,
            dims=dims,
            spatial_dims=spatial_dims,
            **kwargs,
        )

    @property
    def data(self) -> BaseRecording:
        return self._data

    @data.setter
    def data(self, recording: BaseRecording):
        self._data = recording

    def _validate_data(self, data: BaseRecording):
        if not isinstance(data, BaseRecording):
            raise TypeError

        return data

    @property
    def shape(self) -> dict[str, int]:
        return {
            self.spatial_dims[0]: self.data.get_num_channels(),  # `l` dim, number of channels
            self.spatial_dims[1]: self.data.get_num_samples(),  # `p` dim, number of datapoints/samplaines
            self.spatial_dims[2]: 2  # `d` dim, spatial dim, xy data values
        }

    def get(self, indices: tuple[float | int, ...]) -> dict[str, np.ndarray]:
        # assume no additional slider dims, only time slider dim
        s = self._get_dw_slice(indices)

        # slice xs
        xs = self.data.get_times()[s]

        start, stop = s.start, s.stop

        ys = self.data.get_traces(0, start, stop)[::s.step]

        return {"data": np.stack([np.broadcast_to(xs[:, None], (xs.shape[0], ys.shape[1])), ys]).T}
