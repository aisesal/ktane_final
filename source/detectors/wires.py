import enum

import cv2
from numpy.typing import NDArray
from utils import filter_contours_by_area, find_contours, get_centroid


class _WireColor(enum.IntEnum):
    BLACK = 0
    BLUE = enum.auto()
    RED = enum.auto()
    YELLOW = enum.auto()
    WHITE = enum.auto()

_CR_BLACK_WIRE = ((0, 0, 0), (179, 255, 15))
_CR_BLUE_WIRE = ((102, 146, 108), (179, 255, 255))
_CR_RED_WIRE = ((0, 19, 137), (10, 255, 255))
_CR_YELLOW_WIRE = ((21, 171, 110), (39, 255, 255))
_CR_WHITE_WIRE = ((0, 11, 94), (33, 58, 255))
_THRESH_CONTOUR_AREA = 500

COLORS = {
    _WireColor.BLACK: _CR_BLACK_WIRE,
    _WireColor.BLUE: _CR_BLUE_WIRE,
    _WireColor.RED: _CR_RED_WIRE,
    _WireColor.YELLOW: _CR_YELLOW_WIRE,
    _WireColor.WHITE: _CR_WHITE_WIRE,
}

def detect_wires(
        hsv_image: NDArray, return_positions: bool) \
            -> list[_WireColor] | list[tuple[int, int]]:
    wires = list[list[_WireColor, NDArray]]()
    mask = None
    for color_id, color_range in COLORS.items():
        mask = cv2.inRange(hsv_image, *color_range, dst=mask)
        contours = filter_contours_by_area(
            find_contours(mask), _THRESH_CONTOUR_AREA)
        for c in contours:
            wires.append([color_id, c])

    wires.sort(key=lambda x: x[1][:, :, 1].min())
    if not return_positions:
        return [x[0] for x in wires]
    
    return [get_centroid(w[1], round=True) for w in wires]
