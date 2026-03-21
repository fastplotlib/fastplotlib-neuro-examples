import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import ndp_extras
import numpy as np
import json
from lazyvideo import LazyVideo
import cmap
from itertools import cycle
import pandas as pd

# Just data loading stuff
vid_path = "/home/kushal/Downloads/painted_tinted_video.mp4"
vid = LazyVideo(vid_path)
vid_shape = vid.shape

masks_path = "/home/kushal/Downloads/masks.npz"
masks_meta = "/home/kushal/Downloads/masks_meta.json"

masks = np.load(masks_path)
meta = json.load(open(masks_meta, "r"))

metrics_df = pd.read_csv("/home/kushal/Downloads/example_traces.csv")
metrics_df

# for now I'm assuming the frames are ordered in the dataframe
# if they're not let me know, NDWidget can map to the right frame on the fly too
cols = metrics_df.columns.drop("frame")

# convert to [n_lines, n_timepoints, xy]
# NDWidget can also use a pandas dataframe directly if xy data are provided, ex: pose tracking with full xy data

# I'm creating a dict of arrays for each of these metrics since they have very different y-ranges
metrics_data: dict[str, np.ndarray] = dict()
n_timepoints = metrics_df.index.size
for col in cols:
    a = np.zeros((n_timepoints, 2))
    a[:, 0] = np.arange(0, n_timepoints) # x vals
    a[:, 1] = metrics_df[col]
    metrics_data[col] = a[None] # broadcast to [1, n_timepoints, xy]

# Set colors for each tracked object
_, obj_ids = zip(*[k.split("_") for k in list(np.load(masks_path).keys())])
obj_ids = np.unique(obj_ids)
# make a dict obj_id -> color
# you also use your own colors of course!
obj_colors = dict()
for obj_id, color in zip(obj_ids, cycle(cmap.Colormap("tab10").iter_colors())):
    obj_colors[obj_id.item()] = np.asarray(color)
    obj_colors[obj_id.item()][-1] = 0.3 # use an alpha of 0.3
obj_colors

# function to create an overlay on the fly for each frame
def get_overlay(index):
    # creates an RGBA overlay array at the given frame index
    # there are many diff options to do this, this is just one way
    global vid_shape

    i = int(index["frame_index"])

    # RGBA overlay
    overlay = np.zeros((*vid_shape[1:-1], 4), dtype=np.float32)

    for obj_id in obj_ids:
        k = f"f{i}_{obj_id}"
        if k not in meta.keys():
            continue

        mask_shape = meta[k]["shape"]
        tbbox = meta[k]["tbbox"]

        col_start, row_start, *_ = tbbox

        # get the mask indices
        mask_sub = np.unpackbits(masks[k])[:np.prod(mask_shape)].reshape(mask_shape)
        mask_ixs = np.argwhere(mask_sub) + (row_start, col_start)

        # add color indicating this object
        overlay[mask_ixs[:, 0], mask_ixs[:, 1]] =+ obj_colors[obj_id]

    return overlay

# NDWidget code below

# set the reference ranges, this is a range, usually in scientific units relevant to the experiment
# it is usually time in seconds, but if you're only loooking at 1 modality (such as just behavior)
# and all the data arrays you're looking at were sampled at the same time you can just use the array
# min and maxes for the reference range
ref_ranges = {"frame_index": (0, vid.shape[0], 1)} # (start, stop, step)

# subplot locations
# mapping subplot names -> (xmin, xmax, ymin, ymax)
extents = {
    "vid": (0, 0.4, 0, 1),
    "metrics-1": (0.4, 1, 0, 0.33),
    "metrics-2": (0.4, 1, 0.33, 0.66),
    "sentinel": (0.4, 1, 0.66, 1),
}
# create the ndwidget window
ndw = fpl.NDWidget(
    ref_ranges=ref_ranges,
    extents=extents,
    size=(1300, 800), # fig window size in pixels, you can resize it afterwards this is just the initial size
    # shape=(2, 3) # you can alternatively give a shape if a simple grid of subplots is sufficient for your usecase
    # names = ["plot1", "plot2", ...] # if you provide a grid shape you can also give them names
)

# add the video as an NDImage
vid_ndg = ndw["vid"].add_nd_image(
    data=vid,
    # dims names, for convenience, but any "slider" dims like 'time' must be defined in the ref_range
    dims=("frame_index", "r", "c", "rgb"),
    spatial_dims=("r", "c", "rgb"), # the spatial dims you want to look at
    rgb_dim="rgb", # the name of the rgb dim, if present
    compute_histogram=False, # computing a histogram for a large mp4 video can be very slow
)

# add the overlay as an image, we will update it by reading from the masks data on the fly
# you don't have to do this, there are many diff ways to do this, you could precompute
# everything beforehand too but this is probably the most scalable if real datasets will
# have hundreds or thousands of masks
overlay_graphic = ndw.figure["vid"].add_image(get_overlay(ndw.indices))

def update_overlay(new_indices):
    overlay_graphic.data = get_overlay(new_indices)

ndw.indices.add_event_handler(update_overlay, "indices")

metrics = ndw["metrics-1"].add_nd_timeseries(
    metrics_data[cols[0]],
    dims=("metric", "frame_index", "d"),
    spatial_dims=("metric", "frame_index", "d"),
    display_window=None, # window of data to show, in reference units, in this case it's just number of frames to show, if None then it just shows everything
)

metrics = ndw["metrics-2"].add_nd_timeseries(
    metrics_data[cols[1]],
    dims=("metric", "frame_index", "d"),
    spatial_dims=("metric", "frame_index", "d"),
    display_window=None,
)

# reduce some clutter
for subplot in ndw.figure:
    subplot.toolbar = False

ndw.show()

fpl.loop.run()
