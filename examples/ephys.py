from lazyvideo import LazyVideo
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


vid_path_left = "/home/kushal/data/charlie/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_VideoRightCamera.mp4"
vid_path_right = "/home/kushal/data/charlie/sub-CSH-ZAD-026_ses-15763234-d21e-491f-a01b-1238eb96d389_VideoLeftCamera.mp4"

vid_left = LazyVideo(vid_path_left)
vid_right = LazyVideo(vid_path_right)

timings_left = np.load("/home/kushal/data/charlie/timings_left.npy")
timings_right = np.load("/home/kushal/data/charlie/timings_right.npy")

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
    slider_dim_transforms={"time": rec.time_to_sample_index},
)

# set vmin, vmax
ndg_rec.graphic.vmin, ndg_rec.graphic.vmax = -5, 5

# overlay nd scatter on recording heatmap
ndg_ty_overlay = ndw_rec[0, 0].add_nd_timeseries(
    ty_data_scaled,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    slider_dim_transforms={"time": spike_xs.searchsorted},
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
    slider_dim_transforms={"time": spike_xs.searchsorted},
    display_window=1,
    max_display_datapoints=1_000_000,
    # colors=colors,  # if you have per-point colors supply them here
)

ndw_spikes.figure["ampy"].camera.maintain_aspect = False

# add ty data as an nd timeseries
# an nd timeseries provides a linear selector along the x-axis and auto-pans along the x axis
# other than these the linear selector & auto-pan it's almost identical to an nd scatter or nd lines
ndg_ty = ndw_spikes["ty"].add_nd_timeseries(
    ty_data,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    slider_dim_transforms={"time": spike_xs.searchsorted},
    display_window=1,
    x_range_mode="auto",
    graphic_type=fpl.ScatterCollection,
    max_display_datapoints=1_000_000,
    # colors=colors,
)

# better dot sizes and stuff for the scatters
for ndg in [ndg_ampy, ndg_ty]:
    ndg.graphic.sizes = 3
    ndg.graphic.edge_width = 0

# another window for the behavior
ndw_beh = fpl.NDWidget(
    ref_ranges=ref_ranges,
    ref_index=ndw_rec.indices,
    names=["left", "right"],
    shape=(1, 2)
)

# a video is just an nd image
ndw_beh["left"].add_nd_image(
    vid_left,
    dims=("time", "m", "n", "c"),  # mp4 vids are usually RGB, so we have an extra "c" color dim
    spatial_dims=tuple("mnc"),
    rgb_dim="c",  # need to specify color rgb dim if it exists, otherwise it will assume the data is 3D volumes
    slider_dim_transforms={"time": timings_left},
    compute_histogram=False,  # for mp4 vids make sure this is False, else it takes a long time to compute histograms
)

# a video is just an nd image
ndw_beh["right"].add_nd_image(
    vid_right,
    dims=("time", "m", "n", "c"),  # mp4 vids are usually RGB, so we have an extra "c" color dim
    spatial_dims=tuple("mnc"),
    rgb_dim="c",  # need to specify color rgb dim if it exists, otherwise it will assume the data is 3D volumes
    slider_dim_transforms={"time": timings_right},
    compute_histogram=False,
)


ndw_rec.show()
ndw_spikes.show()
ndw_beh.show()

# in a notebook you can use ipywidget layouting
# ex: import ipywidgets
# VBox([ndw_rec.show(), ndw_spikes.show(), ndw_beh.show()])

fpl.loop.run()
