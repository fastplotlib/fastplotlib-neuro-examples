import numpy as np
from pathlib import Path
import pandas as pd
import cmap

from asyncvideo import AsyncVideoReader

import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import ndp_extras


# path with all the files for this session
parent_path = Path("/home/kushal/data/kcenia/")

# decodes ahead on a background thread, YUV planes go straight to the GPU
vid = AsyncVideoReader(parent_path.joinpath("mouse1.mp4"), buffer_size=512)

# recorded camera frame times, this is what maps the reference index onto video frames
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

ethogram_prop = np.dstack(
    [np.broadcast_to(xs[None], (L_expanded.shape[0], L_expanded.shape[1])), L_expanded]
).astype(np.float32)

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
# fractions of the canvas, (xmin, xmax, ymin, ymax). The keys become the subplot names.
# three video views across the top, then two full-width rows of time series below them
extents = {
    "vid-view-0": (0, 1 / 3, 0, 0.5),
    "vid-view-1": (1 / 3, 2 / 3, 0, 0.5),
    "vid-view-2": (2 / 3, 1, 0, 0.5),
    "ethogram": (0, 1, 0.5, 0.75),
    "likelihood": (0, 1, 0.75, 1),
}

ndw = fpl.NDWidget(ranges=reference_range, extents=extents, size=(1000, 1000))

for name in extents:
    ndw.figure[name].title = name

# add the video to each of the three views
for i in range(3):
    ndw[f"vid-view-{i}"].add_video(
        vid,
        dims=("time", "m", "n"),
        display_dims=("m", "n"),
        slider_maps={"time": vid_indexing},
        compute_histogram=False,
        name="vid",
    )

# add behavior data as an nd scatter, overlaid on the video
# a DataFrame has no dims of its own to name, so `dims` and `display_dims` are the same 3 names
# and the next positional arg is the columns, one (x, y) pair per keypoint
nd_scatter = ndw["vid-view-0"].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    slicer=ndp_extras.Pandas,
    display_window=5.0,
    slider_maps={"time": df_tracks["times"].values},
    name="keypoints",
    colors=alpha_using_likelihood,
    # datapoints_window_func = (np.mean, "xy", 3.0),
)

nd_scatter2 = ndw["vid-view-1"].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    slicer=ndp_extras.Pandas,
    display_window=5.0,
    slider_maps={"time": df_tracks["times"].values},
    name="keypoints",
)

nd_scatter3 = ndw["vid-view-2"].add_nd_scatter(
    df_tracks,
    ("l", "time", "d"),
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    slicer=ndp_extras.Pandas,
    display_window=5.0,
    slider_maps={"time": df_tracks["times"].values},
    name="keypoints",
)

# add any other time-series data (likelihood, lighting-pose estimates, ethograms, etc.)
nd_ll = ndw["likelihood"].add_nd_timeseries(
    df_tracks,
    ("l", "time", "d"),
    ("l", "time", "d"),
    [("times", c) for c in likelihood_cols],  # must provide the "times" x-values
    slicer=ndp_extras.Pandas,
    display_window=5,
    slider_maps={"time": df_tracks["times"].values},
    graphic_type=fpl.ImageGraphic,
    name="likelihood",
)

# a discrete colormap with an explicit vmin, vmax so state k is always color k
nd_eth = ndw["ethogram"].add_nd_timeseries(
    ethogram_prop,
    dims=("l", "time", "d"),
    display_dims=("l", "time", "d"),
    display_window=5,
    slider_maps={"time": df_tracks["times"].values},
    graphic_type=fpl.ImageGraphic,
    graphic_kwargs={
        "cmap": cmap.Colormap(["white", "green", "orange", "red"]),
        "vmin": 1,
        "vmax": 4,
    },
    name="ethogram",
)


def prob_tooltip(pick_info):
    # row col position of the cursor
    col, row = pick_info["index"]
    # current displayed image data value at this row, col position
    val = round(nd_eth.graphic.data[row, col])
    return {1: "still", 2: "lick", 3: "move", 4: "groom"}.get(val, "undefined")


nd_eth.graphic.tooltip_format = prob_tooltip

for name in ("ethogram", "likelihood"):
    subplot = ndw.figure[name]
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

cursor = fpl.Cursor()

for i in range(3):
    cursor.add_subplot(ndw.figure[f"vid-view-{i}"])

# a cmap for the scatter collection, each keypoint will get its own color
# There is a lot of fine-tuning you can do for scatter colors
for ng in [nd_scatter, nd_scatter2, nd_scatter3]:
    ng.graphic.cmap = "tab10"

    # set across every graphic in the collection at once
    ng.graphic.sizes = 7
    ng.graphic.visibles = True
    ng.graphic.edge_width = 0.1

ndw.show()
fpl.loop.run()
