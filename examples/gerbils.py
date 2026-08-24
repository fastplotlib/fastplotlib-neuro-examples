import glfw
import numpy as np
from functools import partial
import soundfile as sf
from scipy.signal import spectrogram
import fastplotlib as fpl
import os
from ethogram import Ethogram, EthogramManager, EthogramEditor
from typing import *
from pathlib import Path
from warnings import warn
from asyncvideo import AsyncVideoReader
import pynapple as nap
from imgui_bundle import imgui


# ==== Choose ====
animal_name = "dad"
exp_num = 492
file_num = 11
channel_numbers = [5, 3, 1, 0]  # Choose 4 channels   [2,0,4,5]

spec_height = 300
spec_width = 1500
vid_height = 600

# === spectrogram variables ===
n_fft = 512
n_samples_bin = 512
n_samples_overlap = 256
window_width_sec = 5  # choose


# Functions
def make_specgram(audio, fps=125000):
    f, t, spec = spectrogram(
        audio,
        fs=fps,
        nfft=n_fft,
        nperseg=n_samples_bin,
        noverlap=n_samples_overlap,
        return_onesided=True,
    )
    # Remove the 0-frequency bin and flip the frequency axis so that high frequencies are at the top.
    f = f[1:][::-1]
    spec = np.flip(spec[1:], axis=0)
    spec = np.log(np.abs(spec) + 1e-12)
    spec32 = np.zeros(spec.shape, dtype=np.float32)
    spec32[:] = spec.astype(np.float32)
    return t, f, spec32


# === Video file naming rule based on channel number ===
video_prefix_map = {
    2: "video_center_",
    3: "video_center_",
    4: "video_gily_center_",  # change back
    1: "video_gily_center_",  # change back
    0: "video_nest_top_",  # "video_nest_top_", video_nest_side_
    5: "video_burrow_side_",  # "video_burrow_top_" video_burrow_side_
}

# maps int -> location
name_mapping = {
    # 0: "center-1",
    1: "center-1",  # 1
    2: "center-2",
    3: "center-2",
    0: "nest",
    5: "burrow",
}

burrow_extras = ["video_burrow_top_"]
nest_extras = ["video_nest_other_side_", "video_nest_side_"]

location_order = ["center-2", "center-1", "burrow", "nest"]

base_path = fr"/home/kushal/data/gerbils/{animal_name}_exp{exp_num}/A_V_file_{file_num:03}"

# === Collect paths ===
# maps location str -> path str
video_paths: dict[str, str] = dict()
audio_paths: dict[str, str] = dict()

file_num_str = f"{file_num:03d}"
print(file_num_str)
for ch in channel_numbers:
    video_prefix = video_prefix_map[ch]
    if video_prefix is None:
        print(f"Warning: Invalid channel number {ch}, skipping.")
        continue

    # try padded ("060") first, then unpadded ("60")
    video_path_padded = os.path.join(
        base_path, f"{video_prefix}{file_num_str}.mp4"
    )
    video_path_unpadded = os.path.join(
        base_path, f"{video_prefix}{file_num}.mp4"
    )

    if os.path.exists(video_path_padded):
        video_path = video_path_padded
    elif os.path.exists(video_path_unpadded):
        video_path = video_path_unpadded
    else:
        print(f"Warning: no video found for ch {ch} (tried {video_path_padded} and {video_path_unpadded})")
        continue

    print(video_path)
    audio_path = os.path.join(
        base_path, f"channel_{ch:02d}_file_{file_num_str}.wav"
    )

    video_paths[name_mapping[ch]] = video_path
    audio_paths[name_mapping[ch]] = audio_path

for extra in burrow_extras + nest_extras:
    video_paths[extra] = video_path_padded = os.path.join(
        base_path, f"{extra}{file_num_str}.mp4"
    )

## Load video and audio ##
# # maps location name -> LazyVideo
movies: dict[str, AsyncVideoReader] = dict()

# # maps location name -> full spectrogram
specs: dict[str, np.ndarray] = dict()


# --- Video loading ---
for location, path in video_paths.items():
    # --- check video file lengths and sizes before loading ---
    print("\n=== Checking video file info ===")
    try:
        vid = AsyncVideoReader(path, buffer_size=512)
        n_frames, height, width = vid.shape
        movies[location] = vid
        print(
            f"{os.path.basename(path)}: {n_frames} frames, {width}x{height}"
        )
    except Exception as e:
        print(f"❌ Could not open {path}: {e}")

print("=== Done checking video file info ===\n")


for location, path in audio_paths.items():
    audio_data, fps_audio = sf.read(path, dtype="float32")
    print(f"loaded: {path}")

    # Unpack the outputs of make_specgram
    t_spec, f_spec, spec_data = make_specgram(audio_data, fps_audio)

    spec_reshaped = np.dstack([np.broadcast_to(t_spec[None, :], spec_data.shape), spec_data])

    specs[location] = (spec_reshaped, t_spec, f_spec)

# reference range start, stop step from one of the audio data
ref_range = {"time": (0, t_spec[-1], 1/30)}
extents = [
    # spectrogram subplot locations
    (0, 0.25, 0, 0.35),
    (0.25, 0.5, 0, 0.35),
    (0.5, 0.75, 0, 0.35),
    (0.75, 1, 0, 0.35),
    # behavior vid subplot locations
    (0, 0.25, 0.35, 1),
    (0.25, 0.5, 0.35, 1),
    (0.5, 0.75, 0.35, 1),
    (0.75, 1, 0.35, 1),
]

# %%
spec_names = list()
beh_names = list()
# maps location name -> behavior video NDGraphic
videos = dict()
for loc in location_order:
    spec_names.append(f"spec-{loc}")
    beh_names.append(f"beh-{loc}")

ndw_main = fpl.NDWidget(
    ref_ranges=ref_range,
    extents=extents,
    names=[*spec_names, *beh_names],
    size=(1500, 800),
    controller_ids = None,#[[0, 0, 0, 0], [1, 2, 3, 4]]
)

cursor = fpl.Cursor()

def spec_tooltip_format(pick_info):
    col, row = pick_info["index"]
    return f"freq: {f_spec[row] / 1_000:.1f} kHz"


for loc in location_order:
    spec_loc, t_spec, f_spec = specs[loc]
    spec_name = f"spec-{loc}"

    ng = ndw_main[spec_name].add_nd_timeseries(
        spec_loc,
        ("l", "time", "d"),
        ("l", "time", "d"),
        display_window=5.0,
        slider_dim_transforms={"time": t_spec},
        graphic_type=fpl.ImageGraphic,
        name=spec_name,
        x_range_mode="auto",
        graphic_kwargs = {"metadata": {"f_spec": f_spec}}
    )

    ng.graphic.cmap = "viridis"
    ng.graphic.tooltip_format = spec_tooltip_format
    subplot = ndw_main.figure[spec_name]

    fmin, fmax = f_spec.min(), f_spec.max()
    subplot.axes.y.tick_format = lambda v, fmin, fmax: f"{round(f_spec[min(max(round(v), 0), f_spec.size - 1)] / 1_000)} kHz"
    subplot.axes.y.text.font_size = 16
    subplot.axes.y.text.material.weight_offset = 50
    subplot.axes.y.text.material.outline_thickness = 0.1

    subplot.camera.maintain_aspect = False
    subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

    cursor.add_subplot(subplot)

    beh_name = f"beh-{loc}"
    vid = movies[loc]

    ndg_vid = ndw_main[beh_name].add_video(
        vid,
        dims=("time", "m", "n"),
        spatial_dims=("m", "n"),
        slider_dim_transforms={"time": vid.time},
        compute_histogram=False,
        name="video",
    )
    ndw_main[beh_name].subplot.tooltip.enabled = False

    videos[loc] = ndg_vid

CURRENT_BURROW_SELECTION = "burrow"
# burrow vid options
@ndw_main["beh-burrow"].subplot.add_imgui_window(location="top", size=36, title=None)
def burrow_vid_selection(subplot):
    global CURRENT_BURROW_SELECTION
    for i, option in enumerate(burrow_extras + ["burrow"]):
        if imgui.radio_button(option, CURRENT_BURROW_SELECTION == option):
            if option != CURRENT_BURROW_SELECTION:
                ndw_main["beh-burrow"]["video"].data = movies[option]
                CURRENT_BURROW_SELECTION = option
                subplot.camera.zoom = 1.25
        imgui.same_line()

CURRENT_NEST_SELECTION = "nest"
# nest vid options
@ndw_main["beh-nest"].subplot.add_imgui_window(location="top", size=36, title=None)
def nest_vid_selection():
    global CURRENT_NEST_SELECTION
    for i, option in enumerate(nest_extras + ["nest"]):
        if imgui.radio_button(option, CURRENT_NEST_SELECTION == option):
            if option != CURRENT_NEST_SELECTION:
                ndw_main["beh-nest"]["video"].data = movies[option]
                CURRENT_NEST_SELECTION = option
                subplot.camera.zoom = 1.25
        imgui.same_line()

ndw_main.show()

ndw_ephys = fpl.NDWidget(
    ref_index=ndw_main.indices,
    names=["spikes"],
    size=(1500, 400),
)

spike_times_path = "/home/kushal/data/gerbils/dad_exp492/KS_spike_times.npy"
spike_times = np.load(spike_times_path, allow_pickle=True)

spikes_data = {i: nap.Ts(s / 25_000) for i, s in enumerate(spike_times)}
spikes = nap.TsGroup(spikes_data, time_units="s")

offset = 100
duration = 400
ep = nap.IntervalSet(start=offset, end=offset + duration)

counts = spikes.count(bin_size=0.1, time_units="s", ep=ep)
spike_counts = fpl.utils.heatmap_to_positions(counts.values.T, xvals=counts.t - offset)

spikes_ndg = ndw_ephys["spikes"].add_nd_timeseries(
    spike_counts,
    ("l", "time", "d"),
    ("l", "time", "d"),
    graphic_type=fpl.ImageGraphic,
    slider_dim_transforms={"time": counts.t - offset},
    x_range_mode="auto",
)
spikes_ndg.graphic.cmap = "gray_r"
subplot = ndw_ephys["spikes"].subplot
subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

CURRENT_BIN_SIZE = 100
@ndw_ephys["spikes"].subplot.add_imgui_window(location="top", size=36, title=None)
def bin_size_ui(subplot):
    global CURRENT_BIN_SIZE
    changed, bin_size = imgui.input_int("bin size (ms)", v=CURRENT_BIN_SIZE, step=10, step_fast=50)
    if changed:
        if bin_size < 5:
            bin_size = 5

        cam_state = subplot.camera.get_state()

        counts = spikes.count(bin_size=bin_size / 1_000, time_units="s", ep=ep)
        spike_counts = fpl.utils.heatmap_to_positions(counts.values.T, xvals=counts.t - offset)
        spikes_ndg.data = spike_counts
        spikes_ndg.slider_dim_transforms = {"time": counts.t - offset}
        spikes_ndg.cmap = "gray_r"
        CURRENT_BIN_SIZE = bin_size

        subplot.camera.set_state(cam_state)


def spikes_tooltip_format(pick_info):
    col, row = pick_info["index"]
    n_spikes = int(spikes_ndg.graphic.format_pick_info(pick_info))

    return f"neuron: {row}\nspikes:{round(n_spikes)}"

spikes_ndg.graphic.tooltip_format = spikes_tooltip_format

ndw_ephys.show()


# ethogram, one row per behavior, columns are the behavior video frames
eth_times = movies[location_order[0]].time
ethogram = Ethogram(os.path.join(base_path, "ethogram.csv"), eth_times)

ndw_eth = fpl.NDWidget(
    ref_index=ndw_main.indices,
    names=["ethogram"],
    size=(1500, 400),
)

eth_manager = EthogramManager(
    ndw_eth["ethogram"],
    ethogram,
    display_window=10.0,
    x_range_mode="auto",
    name="ethogram",
)

subplot = ndw_eth["ethogram"].subplot
subplot.controller.add_camera(subplot.camera, include_state={"x", "width"})

ndw_eth.show()

# the entry UI is drawn in its own window
editor = EthogramEditor(ethogram, eth_manager, device=ndw_main.figure.renderer.device)


def video_double_click(ev):
    col, row = ev.pick_info["index"]
    editor.new_entry(row=int(row), col=int(col))


for ndg_vid in videos.values():
    ndg_vid.graphic.add_event_handler(video_double_click, "double_click")


for subplot in ndw_main.figure:
    subplot.toolbar = False
    if "beh" in subplot.name:
        subplot.axes.visible = False
        subplot.camera.zoom = 1.25

fpl.loop.run()
