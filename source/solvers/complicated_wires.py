import typing

from detectors.complicated_wires import detect_complicated_wires
from numpy.typing import NDArray
from utils import bgr2hsv

if typing.TYPE_CHECKING:
    from game_state import GameState

_UI_WIRES = ((873, 595), (909, 595), (951, 595), (995, 595), (1029, 595), (1068, 595))

_DIAGRAM = {
    # red blue led star
    (False, False, False, False): 'C',
    (False, False, False, True): 'C',
    (False, False, True, False): 'D',
    (False, False, True, True): 'B',

    (False, True, False, False): 'S',
    (False, True, False, True): 'D',
    (False, True, True, False): 'P',
    (False, True, True, True): 'P',

    (True, False, False, False): 'S',
    (True, False, False, True): 'C',
    (True, False, True, False): 'B',
    (True, False, True, True): 'B',
    
    (True, True, False, False): 'S',
    (True, True, False, True): 'P',
    (True, True, True, False): 'S',
    (True, True, True, True): 'D',
}

class ComplicatedWires:
    def __init__(self, bgr_image: NDArray) -> None:
        self._wires = detect_complicated_wires(bgr2hsv(bgr_image))
    
    def solve(self, state: 'GameState') -> None:
        for ndx, (led, star, color) in enumerate(self._wires):
            if color == '':
                continue
            #action = _DIAGRAM[(led, star, 'b' in color, 'r' in color)]
            action = _DIAGRAM[('r' in color, 'b' in color, led, star)]
            if self._should_cut(action, state):
                state.mov(*_UI_WIRES[ndx]).ldn().lup().slp()
    
    @staticmethod
    def _should_cut(action: str, state: 'GameState') -> bool:
        if action == 'C':
            return True
        if action == 'S' and state.serial[-1] in '02468':
            return True
        if action == 'P' and state.parallel_port_count > 0:
            return True
        if action == 'B' and state.battery_count >= 2:
            return True
        return False