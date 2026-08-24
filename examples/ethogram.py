from pathlib import Path

import cmap
import fastplotlib as fpl
from imgui_bundle import imgui
import numpy as np
import pandas as pd
import pygfx
from rendercanvas.auto import RenderCanvas
import wgpu
from wgpu.utils.imgui import ImguiRenderer

from fastplotlib.ui import ImguiWindow
from fastplotlib.widgets.nd_widget import NDTimeseries


SOCIAL = ("solo", "pair", "group")
ANIMALS = ("undefined", "mom", "dad", "pup", "multi")
INITIATOR = ("self", "other", "undefined")
BEHAVIORS = (
    "approach",
    "nose-to-nose",
    "fight",
    "play-fight",
    "dig",
    "eat",
    "drink",
    "competition",
    "huddle",
    "groom",
    "chase",
    "alarm-one-call",
    "alarm-bout",
    "heard",
    "produced",
)

# one row per entry in the csv, `row` and `col` are the double click position on a video
COLUMNS = {
    "behavior": "string",
    "social": "string",
    "animal_id": "string",
    "partner_id": "string",
    "initiator": "string",
    "start": "float64",
    "end": "float64",
    "row": "Int64",
    "col": "Int64",
}

# a heatmap cell is 0 where there is no entry, else 1 + the behavior index, so the colormap needs
# one more color than there are behaviors
_colors = np.zeros((len(BEHAVIORS) + 1, 4), dtype=np.float32)
_colors[0] = (0.1, 0.1, 0.1, 1)
_colors[1:] = cmap.Colormap("glasbey").lut(len(BEHAVIORS))

CMAP = pygfx.cm.create_colormap(_colors, n=_colors.shape[0])
VMIN, VMAX = 0, len(BEHAVIORS)


class Ethogram:
    def __init__(self, path: Path | str, times: np.ndarray):
        """
        Behavior entries in a dataframe that is mirrored to a csv, and the heatmap array rendered from them.

        Parameters
        ----------
        path: Path | str
            csv file, loaded if it exists and re-written on every :meth:`store`

        times: np.ndarray
            timestamps for the heatmap columns

        """
        self._path = Path(path)
        self._times = np.asarray(times, dtype=np.float32)

        self._data = np.zeros((len(BEHAVIORS), self._times.size, 2), dtype=np.float32)
        self._data[..., 0] = self._times

        if self._path.exists():
            self._df = pd.read_csv(self._path, dtype=COLUMNS)
        else:
            self._df = pd.DataFrame({c: pd.Series(dtype=d) for c, d in COLUMNS.items()})

        self._rasterize()

    @property
    def data(self) -> np.ndarray:
        """[n_behaviors, n_times, xy] array displayed by the NDTimeseries"""
        return self._data

    @property
    def df(self) -> pd.DataFrame:
        """one row per entry, the same contents as the csv"""
        return self._df

    @property
    def times(self) -> np.ndarray:
        """timestamps of the heatmap columns"""
        return self._times

    def store(self, entry: dict, index: int = None) -> int:
        """append ``entry`` as a new row, or overwrite the row at ``index``, and re-write the csv"""
        if index is None:
            index = self._df.index.size

        self._df.loc[index] = entry
        self._df.to_csv(self._path, index=False)
        self._rasterize()

        return index

    def find(self, behavior: int, t: float) -> int | None:
        """df index of the entry drawn at this behavior row and time, ``None`` if there is no entry"""
        col = self._times.searchsorted(t)
        found = None

        for e in self._df[self._df["behavior"] == BEHAVIORS[behavior]].itertuples():
            span = self._span(e.start, e.end)
            if span.start <= col < span.stop:
                # last match wins, it is the entry that _rasterize() painted on top
                found = e.Index

        return found

    def _span(self, start: float, end: float) -> slice:
        """columns covered by an entry, one column when start == end"""
        a, b = self._times.searchsorted([start, end])
        return slice(a, b + 1)

    def _rasterize(self):
        self._data[..., 1] = 0

        for e in self._df.itertuples():
            row = BEHAVIORS.index(e.behavior)
            self._data[row, self._span(e.start, e.end), 1] = row + 1


def _format_click(entry) -> str:
    if pd.isna(entry["row"]):
        return "Undefined"

    return f"({entry['row']}, {entry['col']})"


class EthogramManager:
    def __init__(self, nd_subplot, ethogram: Ethogram, dim: str = "time", **kwargs):
        """
        Manages the ethogram heatmap, one row per behavior, colored by behavior.

        Hovering a cell shows the entry it belongs to. Once an :class:`EthogramEditor` has been
        added, a double click on an entry loads it into the editor.

        Parameters
        ----------
        nd_subplot: NDWSubplot
            subplot that the heatmap is added to, ex: ``ndw["ethogram"]``

        ethogram: Ethogram
            entries displayed in the heatmap

        dim: str
            reference dim of the timestamps

        kwargs
            passed to ``NDWSubplot.add_nd_timeseries``

        """
        self._ethogram = ethogram
        self._dim = dim
        self._editor = None

        self._nd_graphic = nd_subplot.add_nd_timeseries(
            ethogram.data,
            ("l", dim, "d"),
            ("l", dim, "d"),
            graphic_type=fpl.ImageGraphic,
            slider_dim_transforms={dim: ethogram.times},
            **kwargs,
        )

        self._subplot = nd_subplot

        nd_subplot.subplot.axes.y.tick_format = self._tick_format
        self._connect_graphic()

        # the span of the in-progress entry, `center` is only applied with a parent graphic so the
        # offset is set directly, and a parent would also dangle when refresh() replaces the graphic
        self._selector = fpl.LinearRegionSelector(
            (0, 0),
            limits=(float(ethogram.times[0]), float(ethogram.times[-1])),
            size=len(BEHAVIORS),
            center=0,
            resizable=False,
            edge_color="cyan",
            name="__entry_in_progress",
        )
        self._selector.offset = (0, (len(BEHAVIORS) - 1) / 2, 0)
        self._selector.visible = False

        # display only, the fill blocks picking on the heatmap underneath
        for wo in self._selector.world_object.children:
            wo.material.pick_write = False

        nd_subplot.subplot.add_graphic(self._selector)
        nd_subplot.subplot.add_animations(self._update_selector)

    @property
    def nd_graphic(self) -> NDTimeseries:
        """the NDTimeseries displaying the ethogram"""
        return self._nd_graphic

    @property
    def ref_index(self) -> float:
        """current index of the reference dim"""
        # numpy scalar when the reference range or the x_range_mode poller supplies one
        return float(self._nd_graphic.indices[self._dim])

    def add_editor(self, editor):
        """a double click on an entry loads it into this editor"""
        self._editor = editor

    def refresh(self):
        """re-render from the ethogram array, ex: after an entry has been stored"""
        # setting NDGraphic.data replaces the ImageGraphic, so it is connected again
        cam_state = self._subplot.subplot.camera.get_state()
        self._nd_graphic.data = self._ethogram.data.astype(np.float32)
        self._subplot.subplot.camera.set_state(cam_state)

        self._connect_graphic()

    def tooltip_format(self, pick_info: dict) -> str:
        col, row = pick_info["index"]
        index = self._ethogram.find(row, self._pick_time(col))

        if index is None:
            return BEHAVIORS[row]

        e = self._ethogram.df.loc[index]
        return (
            f"{e['behavior']}\n"
            f"social: {e['social']}\n"
            f"id: {e['animal_id']}\n"
            f"partner: {e['partner_id']}\n"
            f"initiator: {e['initiator']}\n"
            f"{e['start']:.3f} → {e['end']:.3f}\n"
            f"click: {_format_click(e)}"
        )

    def _connect_graphic(self):
        graphic = self._nd_graphic.graphic

        graphic._material.map = CMAP
        graphic.vmin, graphic.vmax = VMIN, VMAX

        graphic.tooltip_format = self.tooltip_format
        graphic.add_event_handler(self._double_click, "double_click")

    def _tick_format(self, value, min_value, max_value) -> str:
        return BEHAVIORS[min(max(round(value), 0), len(BEHAVIORS) - 1)]

    def _update_selector(self):
        if self._editor is None or not self._editor.in_progress:
            self._selector.visible = False
            return

        entry = self._editor.entry
        self._selector.visible = True
        # sorted, the selection feature ignores a reversed range
        self._selector.selection = sorted((entry["start"], entry["end"]))

    def _pick_time(self, col: int) -> float:
        """time at a picked heatmap column"""
        return float(self._nd_graphic.graphic.map_model_to_world((col, 0, 0))[0])

    def _double_click(self, ev):
        if self._editor is None:
            return

        col, row = ev.pick_info["index"]
        index = self._ethogram.find(row, self._pick_time(col))

        if index is not None:
            self._editor.edit(index)


class EthogramEditor(ImguiWindow):
    def __init__(
        self,
        ethogram: Ethogram,
        eth_manager: EthogramManager,
        device: wgpu.GPUDevice,
        size: tuple[int, int] = (560, 620),
    ):
        """
        imgui UI for creating and editing ethogram entries, drawn in its own window.

        Parameters
        ----------
        ethogram: Ethogram
            entries edited by this UI

        eth_manager: EthogramManager
            provides the current reference index, and is re-rendered when an entry is stored

        device: wgpu.GPUDevice
            device for the imgui renderer, ex: ``ndw.figure.renderer.device``

        size: (int, int)
            window size in pixels

        """
        super().__init__()

        self._ethogram = ethogram
        self._eth_manager = eth_manager

        # the working entry, and the df row it overwrites on store, None for a new entry
        self._entry: dict = None
        self._index: int = None
        self.new_entry()

        self._device = device
        self._canvas = RenderCanvas(
            size=size, title="ethogram entry", max_fps=60, update_mode="continuous"
        )
        self._imgui_renderer = ImguiRenderer(device, self._canvas)
        self._imgui_renderer.set_gui(self.draw)
        self._canvas.request_draw(self._draw_frame)

        eth_manager.add_editor(self)

    @property
    def entry(self) -> dict:
        """the entry currently being edited"""
        return self._entry

    @property
    def in_progress(self) -> bool:
        """whether the entry being edited has not been stored yet"""
        return self._index is None

    def new_entry(self, row=pd.NA, col=pd.NA):
        """discard the entry being edited and start a new one at the current reference index"""
        ref_index = self._eth_manager.ref_index

        self._entry = {
            "behavior": "",
            "social": "",
            "animal_id": "undefined",
            "partner_id": "undefined",
            "initiator": "undefined",
            "start": ref_index,
            "end": ref_index,
            "row": row,
            "col": col,
        }
        self._index = None

    def edit(self, index: int):
        """load the entry at this df index into the UI"""
        self._entry = self._ethogram.df.loc[index].to_dict()
        self._index = index

    def update(self):
        entry = self._entry
        ref_index = self._eth_manager.ref_index

        if self.in_progress:
            # the end follows the reference index until the entry is stored
            entry["end"] = ref_index

        if self._index is None:
            imgui.text("new entry")
        else:
            imgui.text(f"editing entry {self._index}")

        imgui.same_line()
        imgui.text(f"click: {_format_click(entry)}")

        imgui.separator_text("Times")
        imgui.text(f"{entry['start']:.3f} - {entry['end']:.3f}")
        imgui.same_line()
        imgui.text(f"index: {ref_index:.3f}")

        if imgui.button("reset start"):
            entry["start"] = ref_index

        child_flags = imgui.ChildFlags_.auto_resize_y | imgui.ChildFlags_.always_auto_resize

        imgui.begin_child("groups", imgui.ImVec2(270, 0), child_flags)
        self._radio("Social", "social", SOCIAL)
        self._radio("ID of clicked animal", "animal_id", ANIMALS)
        self._radio("ID of partner", "partner_id", ANIMALS)
        self._radio("Initiator", "initiator", INITIATOR)
        imgui.end_child()

        imgui.same_line()

        imgui.begin_child("behaviors", imgui.ImVec2(0, 0), child_flags)
        # the spacers group the alarm calls, and whether a call was heard or produced
        self._radio("Behavior", "behavior", BEHAVIORS, spacers=(11, 13))
        imgui.end_child()

        imgui.separator()
        width = imgui.get_content_region_avail().x

        radios = (entry["behavior"], entry["social"], entry["animal_id"],
                  entry["partner_id"], entry["initiator"])

        imgui.begin_disabled(not all(radios) or entry["end"] < entry["start"])
        if imgui.button("Store", imgui.ImVec2(width, 0)):
            self._index = self._ethogram.store(entry, self._index)
            self._eth_manager.refresh()
        imgui.end_disabled()

        if imgui.button("New Entry", imgui.ImVec2(width, 0)):
            self.new_entry()

    def draw(self):
        imgui.set_next_window_size(self._canvas.get_logical_size())
        imgui.set_next_window_pos((0, 0))
        imgui.begin(f"##{self._id_counter}", p_open=None, flags=self._window_flags)

        imgui.push_id(self._id_counter)
        for update_call in self._update_calls:
            update_call()
        imgui.pop_id()

        imgui.end()

    def _radio(
        self,
        label: str,
        field: str,
        options: tuple[str, ...],
        spacers: tuple[int, ...] = (),
    ):
        imgui.separator_text(label)
        # "undefined" is an option in several groups, push the field so the labels stay unique
        imgui.push_id(field)

        for i, option in enumerate(options):
            if i in spacers:
                imgui.spacing()
            if imgui.radio_button(option, bool(self._entry[field] == option)):
                self._entry[field] = option

        imgui.pop_id()

    def _draw_frame(self):
        # the imgui renderer loads the canvas instead of clearing it
        self._clear()
        self._imgui_renderer.render()

    def _clear(self):
        context = self._canvas.get_wgpu_context()
        encoder = self._device.create_command_encoder()

        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": context.get_current_texture().create_view(),
                    "resolve_target": None,
                    "clear_value": (0, 0, 0, 1),
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                }
            ],
        )
        render_pass.end()
        self._device.queue.submit([encoder.finish()])
