from collections.abc import Callable
from functools import partial

import cv2
import numpy as np
import fastplotlib as fpl
import pygfx
import masknmf
from tqdm import tqdm
import cmap


def mask_to_contour_points(mask: np.ndarray, outline_mode) -> np.ndarray:
    # make contour outlines
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    if outline_mode == "all":
        points = np.fliplr(np.vstack(contours).squeeze())
    elif outline_mode == "top":
        sizes = [c.shape[0] for c in contours]
        if len(sizes) < 1:
            points = []
        else:
            biggest_ix = np.argmax(sizes)
            contour_biggest = contours[biggest_ix].squeeze()
            if contour_biggest.ndim < 2:  # single point
                # force to be 2d
                contour_biggest = contour_biggest[None]
            points = np.fliplr(contour_biggest)
    else:
        raise ValueError("`outline_mode` must be one of: 'top' | 'all'")

    return points


def pixel_crop_stack(array, p1, p2):
    if array.shape[0] == 1:
        raise ValueError("Need more than 1 frame in data")
    if np.amin(p1) == np.amax(p1):
        term1 = slice(np.amin(p1), np.amin(p1) + 1)
        dim1_flag = True
    else:
        term1 = slice(np.amin(p1), np.amax(p1) + 1)
        dim1_flag = False

    if np.amin(p2) == np.amax(p2):
        term2 = slice(np.amin(p2), np.amin(p2) + 1)
        dim2_flag = True
    else:
        term2 = slice(np.amin(p2), np.amax(p2) + 1)
        dim2_flag = False

    selected_pixels = array[:, term1, term2].squeeze()

    if dim1_flag and dim2_flag:
        data_2d = selected_pixels[:, None]
    elif dim1_flag and not dim2_flag:
        data_2d = selected_pixels[:, None, p2 - np.amin(p2)]
    elif not dim1_flag and dim2_flag:
        data_2d = selected_pixels[:, p1 - np.amin(p1), None]
    else:
        data_2d = selected_pixels[:, p1 - np.amin(p1), p2 - np.amin(p2)]
    return data_2d


# From mask nmf
# For every signal, need to look at the temporal trace and the PMD average, superimposed
def get_roi_avg(array, p1, p2, normalize=True):
    """
    Given nonzero dim1 and dim2 indices p1 and p2, get the ROI average
    """
    data_2d = pixel_crop_stack(array, p1, p2)
    avg_trace = np.mean(data_2d, axis=1)
    if normalize:
        return avg_trace / np.amax(avg_trace)
    else:
        return avg_trace


def get_contours(
    dmr: masknmf.DemixingResults,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """
    Returns
        contours, masks, centers
    """
    sparse_data = dmr.a

    contours = list()
    centers = np.zeros((sparse_data.shape[1], 2), dtype=np.float32)

    # create masks
    masks_argwhere = list()
    for comp_index in tqdm(range(sparse_data.shape[1])):
        # TODO: keep this as a torch tensor to compute center, will b 10x faster
        mask = (
            sparse_data.T[comp_index].to_dense().cpu().numpy().reshape(dmr.fov_shape)
            > 1e-6
        )

        center = np.argwhere(mask).mean(axis=0)
        centers[comp_index] = center
        points = mask_to_contour_points(mask, outline_mode="top")
        contours.append(points)

        ixs = np.argwhere(mask)
        masks_argwhere.append(ixs)

    return contours, masks_argwhere, centers


def get_other_traces(
    dmr: masknmf.DemixingResults,
    demixing_array_names: list[str],
    masks: list[np.ndarray],
) -> dict[str, np.ndarray]:
    # demixing_array_names = [
    #     "pmd_array",
    #     "residual_array",
    #     "fluctuating_background_array",
    # ]

    roi_avgs = dict()

    # need to get ROI averages for each movie other than AC
    for name in demixing_array_names:
        movie = getattr(dmr, name)
        traces = np.zeros((len(masks), dmr.shape[0]))

        for i, ixs in enumerate(masks):
            traces[i] = get_roi_avg(movie, ixs[:, 0], ixs[:, 1])
        roi_avgs[name] = traces

    return roi_avgs


def texture_from_contours(
    contours: tuple[np.ndarray, ...], fov_shape: tuple[int, int], alpha: float = 0.05
) -> np.ndarray:

    texture_data = np.zeros((*fov_shape, 4), dtype=np.float32)

    for comp_index in range(len(contours)):
        for p in contours[comp_index]:
            texture_data[p[0], p[1]] += [1, 1, 1, alpha]

    return texture_data


class ContoursManager:
    def __init__(
        self,
        demixing_results: masknmf.DemixingResults | list[masknmf.DemixingResults],
        subplots,
    ):
        self._demixing_results = demixing_results
        self._subplots = subplots

        if isinstance(demixing_results, masknmf.DemixingResults):
            contours, masks, centers = get_contours(self._demixing_results)

            self._contours = {self._demixing_results: contours}
            self._masks = {self._demixing_results: masks}
            self._centers = {self._demixing_results: centers}

            self._sync_selection = True

        elif isinstance(demixing_results, (list, tuple)):
            # multi-session is an example for this use case
            if len(subplots) != len(demixing_results):
                raise IndexError

            self._contours = dict.fromkeys(self._demixing_results)
            self._masks = dict.fromkeys(self._demixing_results)
            self._cetners = dict.fromkeys(self._demixing_results)

            for dmr in demixing_results:
                contours, masks, centers = get_contours(dmr)
                self._contours[dmr] = contours
                self._masks[dmr] = masks
                self._centers[dmr] = centers

            self._sync_selection = False

        self._create_contours()

        self._block_select_component_handler = False
        self._block_clear_selection_handler = False

        self._event_handlers = set()

        # list of current selections, contains the DMR and component index for each selection
        # so we can keep track of comps from different DMRs
        self._selection: list[tuple[masknmf.DemixingResults, int]] = list()

        self._colors = cmap.Colormap("tab20").lut()

    @property
    def original_contours_textures(self) -> dict[masknmf.DemixingResults, np.ndarray]:
        return self._original_contours_textures

    def _create_contours(self):
        # first clear any existing contour graphics
        for subplot in self._subplots:
            if "contours" in subplot:
                subplot["contours"].clear_event_handlers()
                subplot.delete_graphic(subplot["contours"])

        if self._sync_selection:
            # contours same for all subplots
            texture = texture_from_contours(
                contours=self._contours[self._demixing_results],
                fov_shape=self._demixing_results.fov_shape,
            )

            self._original_contours_textures = {self._demixing_results: texture}

            # the first image graphic
            contours_graphic = fpl.ImageGraphic(
                self.original_contours_textures[self._demixing_results],
                vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
                vmax=1,
                name="contours",
                offset=(0, 0, -0.1),  # make sure it's above the calcium video image
            )

            contours_graphic.tooltip_format = partial(
                self.tooltip_comp_index, self._demixing_results
            )

            self._subplots[0].add_graphic(contours_graphic)

            # make ImageGraphics for the rest
            # we already have the first ImageGraphic so we just
            # need to make the rest and share the buffer with
            # the first ImageGraphic
            for subplot in self._subplots[1:]:
                cg = subplot.add_image(
                    data=contours_graphic.data,  # this will use the same data buffer
                    vmin=0,
                    vmax=1,
                    name="contours",
                    offset=(0, 0, -0.1),
                )
                cg.tooltip_format = partial(
                    self.tooltip_comp_index, self._demixing_results
                )

            for subplot in self._subplots:
                subplot["contours"].add_event_handler(
                    partial(self._image_clicked, subplot, self._demixing_results),
                    "double_click",
                )

            self._contour_graphics = {self._demixing_results: contours_graphic}

        else:
            self._original_contours_textures = dict.fromkeys(self._demixing_results)

            for subplot, dmr in zip(self._subplots, self._demixing_results):
                texture = texture_from_contours(
                    contours=self._contours[dmr],
                    fov_shape=dmr.fov_shape,
                )

                self._original_contours_textures[dmr] = texture

                contours_graphic = fpl.ImageGraphic(
                    texture,
                    vmin=0,  # makes it easier to set the colors of the contour highlights using vals between 0 - 1
                    vmax=1,
                    name="contours",
                    offset=(0, 0, -0.1),  # make sure it's above the calcium video image
                )

                contours_graphic.add_event_handler(
                    partial(self._image_clicked, subplot, dmr), "double_click"
                )

                contours_graphic.tooltip_format = partial(
                    self.tooltip_comp_index, dmr
                )

    def select_component(self, dmr: masknmf.DemixingResults, comp_index: int):
        if (dmr, comp_index) in self._selection:
            # already selected
            return

        # get the contour that corresponds to this component index
        contour = self._contours[dmr][comp_index]

        # get the current selection color
        color = self._colors[len(self._selection)]

        # if the buffer is shared (synced selection), then this will change ALL contour ImageGraphics
        # if the contours are independent per-subplot, then this will change it for just that subplot
        self._contour_graphics[dmr].data[contour[:, 0], contour[:, 1]] = color

        self._selection.append((dmr, comp_index))

        self._selection_changed(self._selection)

    def clear_selection(self, subplot, dmr):
        if len(self._selection) < 1:
            # nothing is selected
            return

        # if the buffer is shared (synced selection), then this will clear ALL contour ImageGraphic textures
        # if the contours are independent per-subplot, then this will change it for just that subplot
        subplot["contours"].data = self.original_contours_textures[dmr]

        self._selection.clear()

        self._selection_changed(self._selection)

    def _image_clicked(
        self, subplot, dmr: masknmf.DemixingResults, ev: pygfx.PointerEvent
    ):
        if "Shift" not in ev.modifiers:
            # clear selection
            self.clear_selection(subplot, dmr)

        col, row = ev.pick_info["index"]

        index = self.find_closest_components(dmr, (row, col))[0]

        self.select_component(dmr, index)

    def find_closest_components(
        self, dmr: masknmf.DemixingResults, point: tuple[float, float]
    ) -> np.ndarray:
        """

        Args:
            point (float, float): [row, col] index of the point, NOT x, y coordinates

        Returns:
            Indices of components from closest to farthest
        """
        # need to use nanargmin because some centers will be nan if the contour is degenerate
        indices = np.argsort(np.linalg.norm(self._centers[dmr] - point, ord=2, axis=1))
        return indices

    def tooltip_comp_index(self, dmr, pick_info: dict) -> str:
        col, row = pick_info["index"]

        comp = self.find_closest_components(dmr, (row, col))[0]

        info = f"component: {comp}"

        # return this string to display it in the tooltip
        return info

    def add_event_handler(self, func: Callable):
        self._event_handlers.add(func)

    def _selection_changed(self, selection: list):
        for func in self._event_handlers:
            func(selection)

    def remove_event_handler(self, func: Callable):
        self._event_handlers.remove(func)

    def clear_event_handlers(self):
        self._event_handlers.clear()
