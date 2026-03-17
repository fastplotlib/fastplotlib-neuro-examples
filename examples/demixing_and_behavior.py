from pathlib import Path

import masknmf
import pygfx
import cmap
import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import ndp_extras
import numpy as np
from ibl import Video
from masknmf_utils import ContoursManager


parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-18"

session_path = parent_path.joinpath(subject, session)

vid_left = Video(parent_path, subject, session, "left", gpu_context=True)
vid_right = Video(parent_path, subject, session, "right", gpu_context=True)

dmr_path = session_path.joinpath(f"demix.hdf5")
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)
dmr.to("cuda")

fov_index = "00"

imaging_timings_path = session_path.joinpath(
    "001", "alf", f"FOV_{fov_index}", "mpci.times.npy"
)
dmr.timings = np.load(imaging_timings_path)

start_time = np.min(
    [np.min(a) for a in [vid_left.timings, vid_right.timings, dmr.timings]]
)
stop_time = np.min(
    [np.max(a) for a in [vid_left.timings, vid_right.timings, dmr.timings]]
)
# 25 ms step size
step = 25 / 1000

ref_range = {"time": (start_time, stop_time, step)}

extents = [
    (0, 0.33, 0.0, 0.25),  # pmd
    (0.33, 0.66, 0.0, 0.25),  # ac
    (0.66, 1, 0.0, 0.5),  # beh left
    (0, 0.33, 0.25, 0.5),  # res
    (0.33, 0.66, 0.25, 0.5),  # bg
    (0.66, 1, 0.5, 1.0),  # beh right
    (0, 0.66, 0.5, 0.75),  # traces
    (0, 0.66, 0.75, 1),  # lightning pose ethogram
]

ndw = fpl.NDWidget(
    ref_range,
    extents=extents,
    names=[
        "pmd",
        "ac",
        "behavior-left",
        "residual",
        "bg",
        "behavior-right",
        "traces",
        "ethogram",
    ],
    controller_ids=[
        ("pmd", "ac", "residual", "bg"),
        ("traces",),
        ("behavior-right",),
        ("behavior-left",),
    ],
    size=(1200, 1200),
)

# dmr.pmd_array
# dmr.residual_array
# dmr.ac_array
# dmr.fluctuating_background_array
# dmr.colorful_ac_array
#
# dmr.bkgd_corr_img_mean

calcium_dims = ["time", "m", "n"]
calcium_spatial_dims = ["m", "n"]
calcium_index_mapping = {"time": dmr.timings}

vid_dims = ["time", "m", "n", "c"]
vid_spatial_dims = ["m", "n", "c"]

dmr.pmd_array.to("cuda")

ndg_pmd = ndw["pmd"].add_nd_image(
    dmr.pmd_array,
    calcium_dims,
    calcium_spatial_dims,
    index_mappings=calcium_index_mapping.copy(),
    name="pmd movie",
)

ndg_ac = ndw["ac"].add_nd_image(
    dmr.ac_array,
    calcium_dims,
    calcium_spatial_dims,
    index_mappings=calcium_index_mapping.copy(),
    name="ac movie",
)

ndg_res = ndw["residual"].add_nd_image(
    dmr.residual_array,
    calcium_dims,
    calcium_spatial_dims,
    index_mappings=calcium_index_mapping.copy(),
    name="pmd movie",
)

ndg_bg = ndw["bg"].add_nd_image(
    dmr.fluctuating_background_array,
    calcium_dims,
    calcium_spatial_dims,
    index_mappings=calcium_index_mapping.copy(),
    name="bg movie",
)

ndg_beh_left = ndw["behavior-left"].add_nd_image(
    vid_left.array,
    vid_dims,
    vid_spatial_dims,
    rgb_dim="c",
    index_mappings={"time": vid_left.timings},
    compute_histogram=False,
    name="beh movie left",
)

ndg_beh_right = ndw["behavior-right"].add_nd_image(
    vid_right.array,
    vid_dims,
    vid_spatial_dims,
    rgb_dim="c",
    index_mappings={"time": vid_right.timings},
    compute_histogram=False,
    name="beh movie right",
)

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


# add behavior data as an nd scatter
nd_scatter_left = ndw["behavior-left"].add_nd_scatter(
    vid_left.tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    index_mappings={"time": vid_left.timings},
    name="keypoints",
)

# add behavior data as an nd scatter
nd_scatter_right = ndw["behavior-right"].add_nd_scatter(
    vid_right.tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    index_mappings={"time": vid_right.timings},
    name="keypoints",
)

for ng in [nd_scatter_left, nd_scatter_right]:
    ng.graphic.cmap = "tab10"

    # change some properties of the scatter
    for g in ng.graphic:
        g.sizes = 7
        g.visible = True
        g.edge_width = 0.1


# add lightning pose ethogram
nd_eth = ndw["ethogram"].add_nd_timeseries(
    vid_left.ethogram_prop,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    display_window=50.0,
    index_mappings={"time": vid_left.timings},
    graphic=fpl.ImageGraphic,
    name="ethogram",
    x_range_mode="view-range",
)

c = pygfx.cm.create_colormap(
    cmap.Colormap(["white", "green", "orange", "red"]).lut(4), n=4
)
nd_eth.graphic._material.map = c
nd_eth.graphic.vmin, nd_eth.graphic.vmax = 1, 4


def prob_tooltip(pick_info):
    # row col position of the cursor
    col, row = pick_info["index"]
    # current displayed image data value at this row, col position
    val = round(nd_eth.graphic.data[row, col])
    return {1: "still", 2: "lick", 3: "move", 4: "groom"}.get(val, "undefined")


nd_eth.graphic.tooltip_format = prob_tooltip


# convert C to [l, p, 2]
arr = np.zeros((dmr.c.shape[0], dmr.c.shape[1], 2), dtype=np.float32)

# get c in shape [k, t]
c = dmr.c.T.cpu().numpy()

# shape to [k, t, xy], i.e. [k, t, 2]
traces = np.dstack([np.broadcast_to(dmr.timings[None], (c.shape[0], c.shape[1])), c])

ndg_c = ndw["traces"].add_nd_timeseries(
    traces[:10],
    ("l", "time", "d"),
    ("l", "time", "d"),
    index_mappings=calcium_index_mapping.copy(),
    x_range_mode="view-range",
    display_window=50.0,
    name="traces",
)

for subplot in [ndw.figure["traces"], ndw.figure["ethogram"]]:
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})


contours_manager = ContoursManager(
    demixing_results=dmr,
    subplots=[ndw.figure[name] for name in ["pmd", "ac", "residual", "bg"]],
)


def update_traces(selection: list[tuple[masknmf.DemixingResults, int]]):
    if len(selection) < 1:
        return

    # the new selected components
    indices = [s[1] for s in selection]

    c_subset = c[np.asarray(indices)]

    new_data = np.dstack(
        [
            np.broadcast_to(dmr.timings[None], (c_subset.shape[0], c_subset.shape[1])),
            c_subset,
        ]
    )

    ndg_c.data = new_data

    for i, g in enumerate(ndg_c.graphic.graphics):
        g.colors = contours_manager._colors[i]

    ndw.figure["traces"].auto_scale()


contours_manager.add_event_handler(update_traces)

cursor = fpl.Cursor()

for subplot in ndw.figure:
    subplot.toolbar = False

for n in ["pmd", "ac", "bg", "residual"]:
    cursor.add_subplot(ndw.figure[n])

ndw.show()
fpl.loop.run()
