import numpy as np
import spikeinterface.full as si
import dartsort
from ephys_utils import NDSpikeInterfaceProcessor
import fastplotlib as fpl

rec_path = "/home/kushal/data/charlie/ppx_sub-CSH-ZAD-026"
rec = si.read_binary_folder(rec_path)

# sorting
sorting = dartsort.threshold(output_dir="/home/kushal/data/charlie/ppx_sub-CSH-ZAD-026_spikes", recording=rec)
spike_xs = sorting.times_seconds
spike_ys = sorting.point_source_localizations[:, 2]
amps = sorting.denoised_ptp_amplitudes

# [n_datapoints, xy] -> [1, n_datapoints, xy]
ty_data = np.column_stack([spike_xs, spike_ys])[None]


# just a scaled ty so it fits onto the recording heatmap, not sure if this is entirely right??
ty_data_scaled = np.column_stack([spike_xs, spike_ys / 10])[None]
ampy_data = np.c_[amps, spike_ys][None]

# generate some random colors with the same number of datapoints as ampy_data and ty_data
data_colors = np.random.rand(ampy_data.shape[1], 4)
data_colors[:, -1] = 8  # set alpha value to 0.8

# define the reference ranges, you will probably just have time
# other examples of reference spaces are depth
ref_ranges = {"time": (rec.get_start_time(), rec.get_end_time(), 0.001)}

# create an ndwidget, give it the reference range
ndw_rec = fpl.NDWidget(
    ref_ranges=ref_ranges,
    size=(800, 300)
)

# add the recording as an nd timeseries
ndg_rec = ndw_rec[0, 0].add_nd_timeseries(
    rec,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    graphic_type=fpl.ImageGraphic,  # view as an image, can change to a linestack on the fly
    cmap="seismic",
    display_window=0.05,  # window of data to plot in reference space, i.e. seconds here
    x_range_mode="fixed",
    processor=NDSpikeInterfaceProcessor,
    slider_dim_transforms={"time": lambda t: rec.time_to_sample_index(t)},
)

ndw_rec[0, 0].zoom = 1.3
# set vmin, vmax
ndg_rec.graphic.vmin, ndg_rec.graphic.vmax = -5, 5

# overlay nd scatter on recording heatmap
ndg_ty_overlay = ndw_rec[0, 0].add_nd_timeseries(
    ty_data_scaled,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    slider_dim_transforms={"time": lambda t: spike_xs.searchsorted(t)},
    display_window=0.05,
    x_range_mode="fixed",
    graphic_type=fpl.ScatterCollection,
    max_display_datapoints=1_000_000,
)
ndg_ty_overlay.graphic.sizes = 4
# same color all dots so we can specify it like this
ndg_ty_overlay.graphic.colors = "y"

# create an NDWidget for the spikes, share the same reference range and reference index
# you can also just use 1 NDWidget and have these as subplots if you want
ndw_spikes = fpl.NDWidget(
    ref_ranges=ref_ranges,
    ref_index=ndw_rec.indices,  # share the reference index
    extents = {  # the 3 subplots names and their portions of the canvas for this figure
        "xy": (0, 0.25, 0, 1),
        "ampy": (0.25, 0.5, 0, 1),
        "ty": (0.5, 1, 0, 1)
    },
    size=(1000, 700),
)

# add ampy data as an nd scatter
ndg_ampy = ndw_spikes["ampy"].add_nd_scatter(
    ampy_data,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    slider_dim_transforms={"time": lambda t: spike_xs.searchsorted(t)},
    display_window=0.1,
    max_display_datapoints=1_000_000,
    colors=data_colors,  # if you have per-point colors supply them here
    sizes=10,  # scatter sizes, you can also provide an array of per-point sizes
)


ndw_spikes.figure["ampy"].camera.maintain_aspect = False

# add ty data as an nd timeseries
# an nd timeseries provides a linear selector along the x-axis and auto-pans along the x axis
# other than these the linear selector & auto-pan it's almost identical to an nd scatter or nd lines
ndg_ty = ndw_spikes["ty"].add_nd_timeseries(
    ty_data,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    slider_dim_transforms={"time": lambda t: spike_xs.searchsorted(t)},
    display_window=0.1,
    x_range_mode="fixed",
    graphic_type=fpl.ScatterCollection,
    max_display_datapoints=1_000_000,
    colors=data_colors,  # if you have per-point colors supply them here
    sizes=10,
)

ndw_rec.show()
ndw_spikes.show()

# in a notebook you can use ipywidget layouting
# ex: import ipywidgets
# VBox([ndw_rec.show(), ndw_spikes.show(), ndw_beh.show()])

fpl.loop.run()
