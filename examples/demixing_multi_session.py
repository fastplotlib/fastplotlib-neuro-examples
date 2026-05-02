from functools import partial
from itertools import chain
from pathlib import Path
from typing import Sequence

import masknmf
import fastplotlib as fpl
import numpy as np
import pandas as pd
import torch

session_df = pd.read_csv("./multisession-matching.csv")


def get_coors(col_indices: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    rows = col_indices // shape[1]
    cols = col_indices % shape[1]
    return torch.dstack([rows, cols]).squeeze()


def master_to_local(session_id: int, index: Sequence[int]) -> Sequence[int | None]:
    indices = list()
    for i in index:
        local_id = session_df.iloc[i][str(session_id)]

        if np.isnan(local_id):
            return None

        indices.append(int(local_id))

    return indices


def image_clicked(session_id: int, centers: np.ndarray, ev):
    col, row = ev.pick_info["index"]

    local_index = np.argmin(
        np.linalg.norm(centers - np.array([row, col]), axis=1)
    )
    # inverse map
    index = np.argwhere(session_df[str(session_id)].values == local_index)
    if index.size == 0:
        print(f"index not in multi-session matching: {local_index}")
        return

    print(index)
    index = index.item()

    if "Shift" in ev.modifiers:
        sv.append(index)
    else:
        sv.selection = [index]

    for subplot in ndw_traces.figure:
        subplot.auto_scale()


parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"

dmrs = list()
ref_ranges = dict()
for session_index, session in enumerate(["2024-07-18", "2024-07-19", "2024-07-23"]):
    session_path = parent_path.joinpath(subject, session)

    dmr_path = session_path.joinpath(f"demix.hdf5")
    dmr = masknmf.DemixingResults.from_hdf5(dmr_path)
    dmr.to("cuda")

    fov_index = "00"

    imaging_timings_path = session_path.joinpath(
        "001", "alf", f"FOV_{fov_index}", "mpci.times.npy"
    )
    dmr.timings = np.load(imaging_timings_path)
    dmr.pmd_array.to("cuda")

    dmrs.append(dmr)

    # 25 ms step size
    step = 25 / 1000

    ref_ranges.update(
        {
            f"time-{session_index}": (dmr.timings[0], dmr.timings[-1], step),
        }
    )

subplot_names = [f"ac-{i}" for i in range(len(dmrs))]


ndw_images = fpl.NDWidget(
    ref_ranges,
    names=subplot_names,
    controller_ids="sync",
    shape=(1, len(dmrs)),
    size=(1800, 800),
)

ndw_traces = fpl.NDWidget(
    ref_ranges,
    ref_index=ndw_images.indices,
    names=[f"traces-{i}" for i in range(len(dmrs))],
    shape=(len(dmrs), 1),
    size=(1800, 500),
)

sv = fpl.SelectionVector()
ndgs = dict()
# go through each session
for session_index, dmr in enumerate(dmrs):
    # add ac image
    ndi = ndw_images[f"ac-{session_index}"].add_nd_image(
        dmr.ac_array,
        dims=(f"time-{session_index}", "m", "n"),
        spatial_dims=("m", "n"),
        slider_dim_transforms={f"time-{session_index}": dmr.timings},
        name=f"ac-{session_index}",
    )

    # add traces
    ndt = ndw_traces[f"traces-{session_index}"].add_nd_timeseries(
        fpl.utils.heatmap_to_positions(dmr.c.T.cpu().numpy(), dmr.timings),
        dims=("l", f"time-{session_index}", "d"),
        spatial_dims=("l", f"time-{session_index}", "d"),
        x_range_mode="fixed",
        display_window=500,
    )

    contours = dmr.ac_array.contours
    coors = list()
    centers = np.empty((contours.shape[1], 2))

    for k in range(contours.shape[1]):
        c = get_coors(contours[:, k].coalesce().indices(), dmr.fov_shape).cpu().numpy()
        coors.append(c)
        centers[k] = c.mean(axis=0)

    # add selectors
    image_selector = fpl.ImageHighlightSelector(
        lut="tab10",
        lut_wrap="repeat",
        selection_options={"pixels": coors},
        options_color="w",
        options_alpha=0.1,
        alpha=0.7,
    )
    image_selector.add_graphic(ndi.graphic)
    traces_visible_selector = fpl.VisibilitySelector(
        ndt.graphic, lut="tab10", lut_wrap="repeat",
    )
    smap = partial(master_to_local, session_index)
    sv.add_selector((image_selector, smap))
    sv.add_selector((traces_visible_selector, smap))

    ndi.graphic.add_event_handler(partial(image_clicked, session_index, centers), "double_click")

cursor = fpl.Cursor()

for subplot in ndw_images.figure:
    subplot.toolbar = False
    cursor.add_subplot(subplot)

ndw_images.show()
ndw_traces.show()

fpl.loop.run()
