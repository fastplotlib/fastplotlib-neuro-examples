from pathlib import Path
import glfw
import masknmf
import pygfx
import fastplotlib as fpl
import numpy as np
import time
# from ibl import Video
import torch
import cmap as cmap_lib


import pyinstrument

adapter = fpl.enumerate_adapters()[1]
print(adapter.info)
fpl.select_adapter(adapter)

parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-18"

session_path = parent_path.joinpath(subject, session)

dmr_path = session_path.joinpath(f"demix.hdf5")
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)
dmr.to("cuda")

fov_index = "00"

imaging_timings_path = session_path.joinpath(
    "001", "alf", f"FOV_{fov_index}", "mpci.times.npy"
)

dmr.timings = np.load(imaging_timings_path)

ref_range = {"time": (0, dmr.ac_array.shape[0], 1)}

ndw = fpl.NDWidget(
    ref_range,
    shape=(2, 2),
    names=[
        "pmd",
        "ac",
        "residual",
        "bg",
    ],
    controller_ids="sync",
    size=(1200, 1200),
    canvas_kwargs={"max_fps": 999, "vsync": False}
)

calcium_dims = ["time", "m", "n"]
calcium_spatial_dims = ["m", "n"]
compute_histogram = False

dmr.pmd_array.to("cuda")

ndg_pmd = ndw["pmd"].add_nd_image(
    dmr.pmd_array,
    calcium_dims,
    calcium_spatial_dims,
    name="pmd movie",
    compute_histogram=compute_histogram,
)

ndg_ac = ndw["ac"].add_nd_image(
    dmr.ac_array,
    calcium_dims,
    calcium_spatial_dims,
    name="ac movie",
    compute_histogram=compute_histogram,
)

ndg_res = ndw["residual"].add_nd_image(
    dmr.residual_array,
    calcium_dims,
    calcium_spatial_dims,
    name="pmd movie",
    compute_histogram=compute_histogram,
)

ndg_bg = ndw["bg"].add_nd_image(
    dmr.fluctuating_background_array,
    calcium_dims,
    calcium_spatial_dims,
    name="bg movie",
    compute_histogram=compute_histogram,
)

ndw_hm = fpl.NDWidget(
    ref_range,
    ref_index=ndw.indices,
    names=["heatmap"],
    size=(500, 1300),
    canvas_kwargs={"max_fps": 999, "vsync": False},
)

# get c in shape [k, t]
c = dmr.c.T.cpu().numpy()
# shape to [k, t, xy], i.e. [k, t, 2]
traces = fpl.utils.heatmap_to_positions(c, xvals=np.arange(c.shape[1]))#dmr.timings)
traces_dims = ("k", "time", "d")
traces_spatial_dims = traces_dims

ndg_hm_all = ndw_hm["heatmap"].add_nd_timeseries(
    traces,
    dims=traces_dims,
    spatial_dims=traces_spatial_dims,
    graphic_type=fpl.ImageGraphic,
    display_window=None,
    # display_window=30.0, # 30 seconds
)

ndw_traces = fpl.NDWidget(
    ref_range,
    ref_index=ndw.indices,
    names=["heatmap", "traces"],
    shape=(2, 1),
    size=(1000, 500),
    canvas_kwargs={"max_fps": 999, "vsync": False},
)

ndg_traces_selected = ndw_traces["traces"].add_nd_timeseries(
    traces,
    dims=traces_dims,
    spatial_dims=traces_spatial_dims,
    display_window=500.0,
    x_range_mode="auto",
)

ndg_hm_selected = ndw_traces["heatmap"].add_nd_timeseries(
    traces,
    dims=traces_dims,
    spatial_dims=traces_spatial_dims,
    display_window=500.0,
    x_range_mode="auto",
    graphic_type=fpl.ImageGraphic,
)

def get_coors(col_indices: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    rows = col_indices // shape[1]
    cols = col_indices % shape[1]
    return torch.dstack([rows, cols]).squeeze()


contours_ = dmr.ac_array.contours
centers = np.empty(shape=(contours_.shape[1], 2), dtype=np.float32)
contours = list()
for i in range(contours_.shape[1]):
    coors = get_coors(contours_[:, i].coalesce().indices(), dmr.fov_shape).cpu().numpy()
    contours.append(coors)
    centers[i] = coors.mean(axis=0)

tab10_cmap = cmap_lib.Colormap("tab10").lut(10)


# create selectors
contour_selector = fpl.ImageHighlightSelector(
    lut=tab10_cmap,
    lut_wrap="repeat",
    selection_options={"pixels": contours},
    options_color="w",
    options_alpha=0.1,
)

# highlights selected rows on the full heatmap
hm_highlighter = fpl.ImageHighlightSelector(
    lut=tab10_cmap,
    lut_wrap="repeat",
    options_alpha=0.0,
    selection_options={"rows": np.arange(c.shape[0]).tolist()}
)

hm_highlighter.add_graphic(ndg_hm_all.graphic)

# changes which traces are visible in the linestack and heatmap above it
traces_visible_selector = fpl.VisibilitySelector(ndg_traces_selected.graphic, lut=tab10_cmap, lut_wrap="repeat")
traces_hm_visible_selector = fpl.ImageVisibilitySelector(ndg_hm_selected.graphic)

sv = fpl.SelectionVector()

# add all selectors to the selection vector
# contours are pre-loaded in the selection options so we actually don't need to specify a mapping
sv.add_selector(contour_selector)
sv.add_selector(traces_visible_selector)
sv.add_selector(traces_hm_visible_selector)
sv.add_selector(hm_highlighter)


# image click changes the selection, can change the selection vector in any other way too
def image_clicked(ev):
    col, row = ev.pick_info["index"]
    comp_index = np.argmin(np.linalg.norm(centers - np.array([row, col]), axis=1))

    global sv

    if "Shift" in ev.modifiers:
        sv.append(comp_index)
    else:
        sv.selection = [comp_index]

    for subplot in ndw_traces.figure:
        subplot.auto_scale()

# set the contour selectors on the images
for ndg in [ndg_pmd, ndg_ac, ndg_res, ndg_bg]:
    contour_selector.add_graphic(ndg.graphic)
    ndg.graphic.add_event_handler(image_clicked, "double_click")

for subplot in ndw_traces.figure:
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

ndw.show(axes_visible=False)
ndw_hm.show()
ndw_traces.show()
ndw.figure.imgui_show_fps = True

run_profile = False

if run_profile:
    ndw.indices["time"] = 1000
    ndw._sliders_ui._playing["time"] = True

    with pyinstrument.Profiler(async_mode="enabled") as profiler:
        fpl.loop.run()

    profiler.print()
    profiler.open_in_browser()

else:
    fpl.loop.run()
