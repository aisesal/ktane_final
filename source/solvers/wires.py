import typing

import cv2
from detectors.wires import _WireColor, detect_wires
from numpy.typing import NDArray
from utils import bgr2hsv

if typing.TYPE_CHECKING:
    from game_state import GameState

class Wires:
    def __init__(self, bgr_image: NDArray) -> None:
        hsv_image = bgr2hsv(bgr_image)
        self._wires = detect_wires(hsv_image, False)
    
    def solve(self, state: 'GameState') -> None:
        ndx = self._find_solution(int(state.serial[-1])%2 == 1)

        while True:
            hsv_image = bgr2hsv(state.grab_active_module())
            if hsv_image[:, :, 2].mean() > 100:
                break

        positions = detect_wires(hsv_image, True)
        x = positions[ndx][0] + 832
        y = positions[ndx][1] + 387
        state.mov(x, y).ldn().lup().slp()
    
    def _find_solution(self, serial_is_odd: bool) -> int:
        wire_count = len(self._wires)

        if wire_count == 3:
            if _WireColor.RED not in self._wires:
                return 1
            if self._wires[-1] == _WireColor.WHITE:
                return 2
            if self._wires.count(_WireColor.BLUE) > 1:
                return self._rindex(_WireColor.BLUE)
            return 2
        
        if wire_count == 4:
            if self._wires.count(_WireColor.RED) > 1 and serial_is_odd:
                return self._rindex(_WireColor.RED)
            if self._wires[-1] == _WireColor.YELLOW and _WireColor.RED not in self._wires:
                return 0
            if self._wires.count(_WireColor.BLUE) == 1:
                return 0
            if self._wires.count(_WireColor.YELLOW) > 1:
                return 3
            return 1
        
        if wire_count == 5:
            if self._wires[-1] == _WireColor.BLACK and serial_is_odd:
                return 3
            if self._wires.count(_WireColor.RED) == 1 and self._wires.count(_WireColor.YELLOW) > 1:
                return 0
            if _WireColor.BLACK not in self._wires:
                return 1
            return 0
        
        if wire_count == 6:
            if _WireColor.YELLOW not in self._wires and serial_is_odd:
                return 2
            if self._wires.count(_WireColor.YELLOW) == 1 and self._wires.count(_WireColor.WHITE) > 1:
                return 3
            if _WireColor.RED not in self._wires:
                return 5
            return 3
    
    def _rindex(self, color: _WireColor) -> int:
        return len(self._wires)-1 - self._wires[::-1].index(color)
