import time
import typing

import cv2
from utils import bgr2hsv, crop_image

if typing.TYPE_CHECKING:
    from game_state import GameState

_CR_BLINKER = ((0, 177, 198), (34, 255, 255))
_BBOX_BLINKER = (94, 35, 16, 16)
_UI_RIGHT_ARROW = (1065, 560)
_UI_TX_BUTTON = (980, 625)

_WORDS = (
    '... .... . .-.. .-..', # shell
    '.... .- .-.. .-.. ...', # halls
    '... .-.. .. -.-. -.-', # slick
    '- .-. .. -.-. -.-', # trick
    '-... --- -..- . ...', # boxes
    '.-.. . .- -.- ...', # leaks
    '... - .-. --- -... .', # strobe
    '-... .. ... - .-. ---', # bistro
    '..-. .-.. .. -.-. -.-', # flick
    '-... --- -- -... ...', # bombs
    '-... .-. . .- -.-', # break
    '-... .-. .. -.-. -.-', # brick
    '... - . .- -.-', # steak
    '... - .. -. --.', # sting
    '...- . -.-. - --- .-.', # vector
    '-... . .- - ...', # beats
)
_WORDS = tuple('/'.join((x, x)) for x in _WORDS)

class MorseCode:
    def __init__(self) -> None:
        pass
    
    def solve(self, state: 'GameState') -> None:
        self._wait_for_blinker(True, state)
        last_off = self._wait_for_blinker(False, state)
        last_on = self._wait_for_blinker(True, state)

        was_on = True
        code = ''

        while True:
            curr_tm = time.monotonic_ns()
            is_on = self._is_blinker_on(state)
            if was_on and not is_on:
                delta = (curr_tm - last_on) * 1e-9
                code += '-' if delta > 0.7 else '.'
                last_off = curr_tm
                was_on = False
                if self._check_code(code, state):
                    break
            elif not was_on and is_on:
                delta = delta = (curr_tm - last_off) * 1e-9
                last_on = curr_tm
                was_on = True
                if delta < 0.5:
                    continue
                code += '/' if delta > 1.5 else ' '
                if self._check_code(code, state):
                    break

    @staticmethod
    def _check_code(code: str, state: 'GameState') -> bool:
        result = [i for i, word in enumerate(_WORDS) if code in word]
        if len(result) != 1:
            return False
        
        if result[0] != 0:
            state.mov(*_UI_RIGHT_ARROW)
        for _ in range(result[0]):
            state.ldn().lup().slp()
        state.mov(*_UI_TX_BUTTON).ldn().lup().slp()

        return True
    
    @staticmethod
    def _is_blinker_on(state: 'GameState') -> bool:
        hsv_image = bgr2hsv(state.grab_active_module())
        hsv_image = crop_image(hsv_image, _BBOX_BLINKER)
        mask = cv2.inRange(hsv_image, *_CR_BLINKER)
        return cv2.countNonZero(mask) > 200
    
    @staticmethod
    def _wait_for_blinker(wait_for_on: bool, state: 'GameState') -> int:
        tm = time.monotonic_ns()
        while MorseCode._is_blinker_on(state) == wait_for_on:
            pass
        return tm
