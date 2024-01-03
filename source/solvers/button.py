import enum
import typing
from pathlib import Path

import cv2
import numpy as np
import torch
from detectors.timer import detect_timer
from models.button import Button as Model
from numpy.typing import NDArray
from utils import bgr2hsv, bgr2tensor, crop_image

if typing.TYPE_CHECKING:
    from game_state import GameState

class _ButtonColor(enum.IntEnum):
    BLUE = 0
    RED = enum.auto()
    WHITE = enum.auto()
    YELLOW = enum.auto()

class _ButtonText(enum.IntEnum):
    ABORT = 0
    DETONATE = enum.auto()
    HOLD = enum.auto()
    PRESS = enum.auto()

_MODEL = Model().eval()
_MODEL.load_state_dict(torch.load(Path('models/Button.pt')))

_CR_BLUE_STRIP = ((99, 209, 0), (118, 255, 255))
_CR_RED_STRIP = ((0, 194, 0), (0, 255, 255))
_CR_WHITE_STRIP = ((0, 0, 197), (0, 0, 255))
_CR_YELLOW_STRIP = ((23, 212, 0), (31, 255, 255))
_THRESH_STRIP_AREA = 200

_STRIP_COLOR = {
    _ButtonColor.BLUE: _CR_BLUE_STRIP,
    _ButtonColor.RED: _CR_RED_STRIP,
    _ButtonColor.WHITE: _CR_WHITE_STRIP,
    _ButtonColor.YELLOW: _CR_YELLOW_STRIP,
}

_UI_CENTER = (950, 560)

_BBOX_STRIP = (1072-832, 524-387, 16, 16)

_BBOX_TIMER = (
    ( # Timer in 1st position.
        None,
        (535, 518, 228, 122),
        (195, 518, 233, 123),
        (877, 209, 216, 112),
        (553, 209, 218, 112),
        (227, 208, 223, 113),
    ),
    ( # Timer in 2nd position.
        (1207, 518, 227, 122),
        None,
        (535, 518, 227, 122),
        (1196, 209, 218, 112),
        (877, 209, 214, 112),
        (552, 209, 218, 112),
    ),
    None, # NOTE: It seems that timer never appears in 3rd position.
    ( # Timer in 4th position.
        (870, 855, 234, 135),
        (516, 855, 238, 134),
        (160, 856, 245, 134),
        None,
        (535, 518, 228, 122),
        (195, 518, 233, 123),
    ),
    ( # Timer in 5th position.
        (1218, 854, 238, 134),
        (870, 855, 234, 135),
        (515, 855, 239, 134),
        (1207, 518, 227, 122),
        None,
        (535, 518, 227, 122),
    ),
    None, # NOTE: It seems that timer never appears in 6th position.
)

class Button:
    def __init__(self, position: int, bgr_image: NDArray) -> None:
        image = bgr2tensor(bgr_image)
        color, text = _MODEL(image)
        self._color = _ButtonColor(color.argmax().item())
        self._text = _ButtonText(text.argmax().item())
        self._position = position
    
    def solve(self, state: 'GameState') -> None:
        has_lit_car = False
        has_lit_frk = False
        for is_lit, name in state.indicators:
            if name == 'CAR':
                has_lit_car |= is_lit
            elif name == 'FRK':
                has_lit_frk |= is_lit
        
        if not self._should_hold(state.battery_count, has_lit_car, has_lit_frk):
            state.mov(*_UI_CENTER).ldn().lup().slp()
            return
        
        state.mov(*_UI_CENTER).ldn().slp(0.5)

        strip_color = None
        mask = None
        while strip_color is None:
            image = bgr2hsv(crop_image(state.grab_active_module(), _BBOX_STRIP))
            for color, color_range in _STRIP_COLOR.items():
                mask = cv2.inRange(image, *color_range, dst=mask)
                if np.count_nonzero(mask) >= _THRESH_STRIP_AREA:
                    strip_color = color
                    break
                
        if strip_color == _ButtonColor.BLUE:
            target_digit = 4
        elif strip_color == _ButtonColor.YELLOW:
            target_digit = 5
        else:
            target_digit = 1
        
        bbox = _BBOX_TIMER[state.timer_position][self._position]
        while True:
            image = bgr2hsv(crop_image(state.grab_screen(), bbox))
            if target_digit in detect_timer(image):
                break

        state.lup().slp()
    
    def _should_hold(
            self, battery_count: int,
            has_lit_car: bool, has_lit_frk: bool) -> bool:
        if self._color == _ButtonColor.BLUE and self._text == _ButtonText.ABORT:
            return True
        if battery_count > 1 and self._text == _ButtonText.DETONATE:
            return False
        if self._color == _ButtonColor.WHITE and has_lit_car:
            return True
        if battery_count > 2 and has_lit_frk:
            return False
        if self._color == _ButtonColor.YELLOW:
            return True
        if self._color == _ButtonColor.RED and self._text == _ButtonText.HOLD:
            return False
        return True
