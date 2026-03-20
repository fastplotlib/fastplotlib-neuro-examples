from pathlib import Path

import masknmf
import pygfx
import cmap
import fastplotlib as fpl
from fastplotlib.widgets.nd_widget import ndp_extras
import numpy as np
from ibl import Video
from masknmf_utils import ContoursManager

import pyinstrument

adapter = fpl.enumerate_adapters()[0]
print(adapter.info)
fpl.select_adapter(adapter)

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

ndw_fov = fpl.NDWidget(
    ref_range,
    names=["pmd", "ac", "residual", "bg"],
    controller_ids=[
        ("pmd", "ac", "residual", "bg"),
    ],
    shape=(2, 2),
    size=(800, 800),
    canvas_kwargs={"max_fps": 999},
)

ndw_beh = fpl.NDWidget(
    ref_ranges=ndw_fov.indices.ref_ranges,
    ref_index=ndw_fov.indices,
    names=["left", "right"],
    shape=(1, 2),
    size=(800, 400),
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

ndg_pmd = ndw_fov["pmd"].add_nd_image(
    dmr.pmd_array,
    calcium_dims,
    calcium_spatial_dims,
    slider_dim_transforms=calcium_index_mapping.copy(),
    name="pmd movie",
)

ndg_ac = ndw_fov["ac"].add_nd_image(
    dmr.ac_array,
    calcium_dims,
    calcium_spatial_dims,
    slider_dim_transforms=calcium_index_mapping.copy(),
    name="ac movie",
)

ndg_res = ndw_fov["residual"].add_nd_image(
    dmr.residual_array,
    calcium_dims,
    calcium_spatial_dims,
    slider_dim_transforms=calcium_index_mapping.copy(),
    name="pmd movie",
)

ndg_bg = ndw_fov["bg"].add_nd_image(
    dmr.fluctuating_background_array,
    calcium_dims,
    calcium_spatial_dims,
    slider_dim_transforms=calcium_index_mapping.copy(),
    name="bg movie",
)

ndg_beh_left = ndw_beh["left"].add_nd_image(
    vid_left.array,
    vid_dims,
    vid_spatial_dims,
    rgb_dim="c",
    slider_dim_transforms={"time": vid_left.timings},
    compute_histogram=False,
    name="beh movie left",
)

ndg_beh_right = ndw_beh["right"].add_nd_image(
    vid_right.array,
    vid_dims,
    vid_spatial_dims,
    rgb_dim="c",
    slider_dim_transforms={"time": vid_right.timings},
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
nd_scatter_left = ndw_beh["left"].add_nd_scatter(
    vid_left.tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    slider_dim_transforms={"time": vid_left.timings},
    name="keypoints",
)

# add behavior data as an nd scatter
nd_scatter_right = ndw_beh["right"].add_nd_scatter(
    vid_right.tracks,
    ("l", "time", "d"),
    keypoints_cols[:, :-1],
    processor=ndp_extras.NDPP_Pandas,
    display_window=5.0,
    slider_dim_transforms={"time": vid_right.timings},
    name="keypoints",
)

for ng in [nd_scatter_left, nd_scatter_right]:
    ng.graphic.cmap = "tab10"

    # change some properties of the scatter
    for g in ng.graphic:
        g.sizes = 7
        g.visible = True
        g.edge_width = 0.1


cursor = fpl.Cursor()
for n in ["pmd", "ac", "bg", "residual"]:
    cursor.add_subplot(ndw_fov.figure[n])

for subplot in ndw_fov.figure:
    subplot.toolbar = False


extents_zoom = {
    "ac-zoom": (0, 0.25, 0, 1),

    "C": (0.25, 1, 0, 0.5),
    "behavior": (0.25, 1, 0.5, 1),
}

ndw_zoom = fpl.NDWidget(
    ref_ranges=ndw_fov.indices.ref_ranges,
    ref_index=ndw_fov.indices,
    extents=extents_zoom,
    size=(800, 400),
    canvas_kwargs={"max_fps": 999}
)

ndw_zoom.figure["C"].controller.add_camera(
    ndw_zoom.figure["C"].camera, include_state={"x", "width"}
)
ndw_zoom.figure["behavior"].controller.add_camera(
    ndw_zoom.figure["behavior"].camera, include_state={"x", "width"}
)

calcium_dims = ["time", "m", "n"]
calcium_spatial_dims = ["m", "n"]
calcium_index_mapping = {"time": dmr.timings}

vid_dims = ["time", "m", "n", "c"]
vid_spatial_dims = ["m", "n", "c"]

dmr.pmd_array.to("cuda")

ac_zoom = ndw_zoom["ac-zoom"].add_nd_image(
    np.zeros((2, 2)),
    dims=("m", "n"),
    spatial_dims=("m", "n"),
)

# add lightning pose ethogram
nd_eth = ndw_zoom["behavior"].add_nd_timeseries(
    vid_left.ethogram_prop,
    dims=("l", "time", "d"),
    spatial_dims=("l", "time", "d"),
    display_window=50.0,
    slider_dim_transforms={"time": vid_left.timings},
    graphic_type=fpl.ImageGraphic,
    name="ethogram",
    x_range_mode="auto",
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

# get c in shape [k, t]
c = dmr.c.T.cpu().numpy()

# [k, t] -> [k, t, 2]
traces = fpl.utils.heatmap_to_positions(c, dmr.timings)

ndg_c = ndw_zoom["C"].add_nd_timeseries(
    None,
    ("l", "time", "d"),
    ("l", "time", "d"),
    slider_dim_transforms=calcium_index_mapping.copy(),
    x_range_mode="auto",
    display_window=50.0,
    name="traces",
)

cursor.add_subplot(ndw_zoom.figure["ac-zoom"])

for subplot in ndw_zoom.figure:
    subplot.toolbar = False


ndw_heatmap = fpl.NDWidget(
    ref_ranges=ndw_fov.indices.ref_ranges,
    ref_index=ndw_fov.indices,
    names=["heatmap"],
    size=(500, 1000),
)

ndg_heatmap = ndw_heatmap[0, 0].add_nd_timeseries(
    traces,
    ("l", "time", "d"),
    ("l", "time", "d"),
    slider_dim_transforms=calcium_index_mapping.copy(),
    x_range_mode=None,
    display_window=None,
    graphic_type=fpl.ImageGraphic,
    name="traces",
)

heatmap_comp_selector = ndg_heatmap.graphic.add_linear_selector(axis="y")

contours_manager = ContoursManager(
    demixing_results=dmr,
    subplots=[ndw_fov.figure[name] for name in ["pmd", "ac", "residual", "bg"]],
)


def update_traces(selection: list[tuple[masknmf.DemixingResults, int]]):
    if len(selection) == 0:
        ndg_c.data = None
        return

    # the new selected components
    indices = [s[1] for s in selection]
    ndg_c.data = traces[np.asarray(indices)]

    # for i, g in enumerate(ndg_c.graphic.graphics):
    #     g.colors = contours_manager._colors[i]

    ndw_zoom.figure["C"].auto_scale()

rect_selectors: list[fpl.RectangleSelector] = list()
for ndi in ndw_fov.ndgraphics:
    rs = ndi.graphic.add_rectangle_selector(
        edge_color="r",
        edge_thickness=1.0,
        vertex_size=4.0,
        vertex_color="w",
        fill_color=(0, 0, 0, 0), # no fill color, otherwise it blocks picking of the image below, which blocks tooltips
    )

    rect_selectors.append(rs)


@heatmap_comp_selector.add_event_handler("selection")
def heatmap_selector_handler(ev: fpl.GraphicFeatureEvent):
    index = ev.get_selected_index()
    select_component(index)


def select_component(index: int | None):
    contours_manager.clear_selection(ndw_fov.figure["ac"], dmr)

    if index is None:
        ndg_c.data = None
        return

    contour = contours_manager._contours[dmr][index]
    # contours are [row_ixs, col_ixs], NOT [x, y].
    # this is because they are used for directly indexing into the image for representing contours graphically
    ymin, xmin, ymax, xmax = (*contour.min(axis=0), *contour.max(axis=0))

    for rs in rect_selectors:
        # pad by a few pixels
        rs.selection = (xmin - 5, xmax + 5, ymin - 5, ymax + 5)

    update_traces([(dmr, index)])
    heatmap_comp_selector.selection = index
    contours_manager.select_component(dmr, index)


contours_manager.add_event_handler(lambda new: select_component(new[-1][1]) if len(new) > 0 else None)


@rect_selectors[1].add_event_handler("selection")
def update_zoom(ev: fpl.GraphicFeatureEvent):
    zoom_data = ev.get_selected_data()
    # this provides a VIEW of the buffer in the full FOV graphic
    # so we do not need to have extra update logic for the zoomed graphic
    # when the full FOV graphic buffer updates, since the zoomed one uses
    # has a view of the same buffer it will also visually update
    ac_zoom.data = zoom_data
    xmin, xmax, ymin, ymax = ev.info["value"]
    # set the position of the zoomed view in the x-y location as the full FOV
    # so the crosshair tool aligns
    ac_zoom.graphic.offset = (xmin, ymin, 0)
    ndw_zoom.figure["ac-zoom"].auto_scale()

    # update the histogram using the current histogram in the ac full FOV subplot
    ac_zoom.histogram_widget.histogram = ndg_ac.processor.histogram
    # likewise set vmin, vmax
    ac_zoom.graphic.vmin, ac_zoom.graphic.vmax = ndg_ac.graphic.vmin, ndg_ac.graphic.vmax


ndw_fov.show()
ndw_beh.show()
ndw_heatmap.show()
ndw_zoom.show()
ndw_heatmap.show()


ndw_zoom.figure.renderer.pixel_ratio = 1.0
ndw_zoom.figure.imgui_show_fps = True


run_profile = False

if run_profile:
    ndw_zoom.indices["time"] = 1000
    ndw_zoom._sliders_ui._playing["time"] = True

    with pyinstrument.Profiler(async_mode="enabled") as profiler:
        fpl.loop.run()

    profiler.print()
    profiler.open_in_browser()

else:
    fpl.loop.run()
