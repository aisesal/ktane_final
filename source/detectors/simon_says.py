import enum

import cv2
from numpy.typing import NDArray
from utils import crop_image


class SimonSaysColor(enum.IntEnum):
    NONE = 0,
    BLUE = enum.auto()
    GREEN = enum.auto()
    RED = enum.auto()
    YELLOW = enum.auto()

_CR_BRIGHTNESS = ((0, 0, 200), (179, 255, 255))
_CR_SOLVED = ((0, 0, 200), (179, 255, 255))

_THRESH_BRIGHTNESS = 200
_THRESH_SOLVED = 200

_BBOX_SOLVED = (244, 10, 32, 32)
_BBOX_SQUARES = {
    SimonSaysColor.BLUE: (138, 69, 16, 16),
    SimonSaysColor.GREEN: (138, 187, 16, 16),
    SimonSaysColor.RED: (78, 129, 16, 16),
    SimonSaysColor.YELLOW: (200, 129, 16, 16),
}

def detect_simon_says(hsv_image: NDArray) -> tuple[SimonSaysColor, bool]:
    mask = cv2.inRange(crop_image(hsv_image, _BBOX_SOLVED), *_CR_SOLVED)
    if cv2.countNonZero(mask) > _THRESH_SOLVED:
        return SimonSaysColor.NONE, True
    
    for color, bbox in _BBOX_SQUARES.items():
        mask = cv2.inRange(crop_image(hsv_image, bbox), *_CR_BRIGHTNESS)
        if cv2.countNonZero(mask) > _THRESH_BRIGHTNESS:
            return color, False
        
    return SimonSaysColor.NONE, False
