from itertools import chain
from pathlib import Path

import masknmf
import fastplotlib as fpl
import numpy as np

# from ibl import Video
from masknmf_utils import ContoursManager


parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"

dmrs = list()
ref_ranges = dict()
for i, session in enumerate(["2024-07-18", "2024-07-19", "2024-07-23"]):
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
            f"time-{i}": (dmr.timings[0], dmr.timings[-1], step),
        }
    )

extents = [
    (0.00, 0.25, 0.75, 1.00),
    (0.25, 0.50, 0.75, 1.00),
    (0.50, 0.75, 0.75, 1.00),
    (0.75, 1.00, 0.75, 1.00),

    (0.00, 0.25, 0.50, 0.75),
    (0.25, 0.50, 0.50, 0.75),
    (0.50, 0.75, 0.50, 0.75),
    (0.75, 1.00, 0.50, 0.75),

    (0.00, 0.25, 0.25, 0.50),
    (0.25, 0.50, 0.25, 0.50),
    (0.50, 0.75, 0.25, 0.50),
    (0.75, 1.00, 0.25, 0.50),

    # --- bottom full-width panel ---
    (0.00, 1.00, 0.00, 0.25),
]

subplot_names = list(chain.from_iterable([f"pmd-{i}", f"ac-{i}", f"residual-{i}", f"fluctuating_background-{i}"] for i in range(len(dmrs))))
subplot_names.extend(["traces"])

ndw = fpl.NDWidget(
    ref_ranges,
    extents=extents,
    names=subplot_names,
    controller_ids="sync",
    shape=(3, 4),
    size=(1200, 1200),
)

ndgs = dict()
for i, dmr in enumerate(dmrs):
    for name in ["pmd", "ac", "residual", "fluctuating_background"]:
        ndg = ndw[f"{name}-{i}"].add_nd_image(
            getattr(dmr, f"{name}_array"),
            (f"time-{i}", "m", "n"),
            ("m", "n"),
            index_mappings={f"time-{i}": dmr.timings},
            name=f"{name}-{i}",
        )

contours_manager = ContoursManager(
    demixing_results=list(chain.from_iterable([dmr] * 4 for dmr in dmrs)),
    subplots=[
        ndw.figure[subplot_name]
        for subplot_name in list(
            chain.from_iterable(
                [f"pmd-{i}", f"ac-{i}", f"residual-{i}", f"fluctuating_background-{i}"]
                for i in range(len(dmrs))
            )
        )
    ],
)


cursor = fpl.Cursor()

for subplot in ndw.figure:
    subplot.toolbar = False
    cursor.add_subplot(subplot)

ndw.show()
fpl.loop.run()
