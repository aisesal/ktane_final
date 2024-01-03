from detectors.bomb_modules import BombModuleType
from numpy.typing import NDArray

from .button import Button
from .complicated_wires import ComplicatedWires
from .keypad import Keypad
from .maze import Maze
from .memory import Memory
from .morse_code import MorseCode
from .password import Password
from .simon_says import SimonSays
from .whos_on_first import WhosOnFirst
from .wire_sequences import WireSequences
from .wires import Wires


def get_solver(
        module_type: BombModuleType, position: int, bgr_image: NDArray):
    if module_type == BombModuleType.BUTTON:
        return Button(position, bgr_image)
    if module_type == BombModuleType.COMPLICATED_WIRES:
        return ComplicatedWires(bgr_image)
    if module_type == BombModuleType.KEYPAD:
        return Keypad(bgr_image)
    if module_type == BombModuleType.MAZE:
        return Maze(bgr_image)
    if module_type == BombModuleType.MEMORY:
        return Memory()
    if module_type == BombModuleType.MORSE_CODE:
        return MorseCode()
    if module_type == BombModuleType.PASSWORD:
        return Password(bgr_image)
    if module_type == BombModuleType.SIMON_SAYS:
        return SimonSays()
    if module_type == BombModuleType.WHOS_ON_FIRST:
        return WhosOnFirst()
    if module_type == BombModuleType.WIRE_SEQUENCES:
        return WireSequences()
    if module_type == BombModuleType.WIRES:
        return Wires(bgr_image)
    return None
