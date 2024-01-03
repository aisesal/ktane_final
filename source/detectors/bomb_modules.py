import enum
from pathlib import Path

import torch
from models.bomb_modules import BombModules as Model
from numpy.typing import NDArray
from utils import bgr2tensor

_MODEL = Model().eval()
_MODEL.load_state_dict(torch.load(Path('models/BombModules.pt')))

class BombModuleType(enum.IntEnum):
    BUTTON = 0
    CAPACITOR_DISCHARGE = enum.auto()
    COMPLICATED_WIRES = enum.auto()
    EMPTY = enum.auto()
    KEYPAD = enum.auto()
    KNOB = enum.auto()
    MAZE = enum.auto()
    MEMORY = enum.auto()
    MORSE_CODE = enum.auto()
    PASSWORD = enum.auto()
    SIMON_SAYS = enum.auto()
    TIMER = enum.auto()
    VENTING_GAS = enum.auto()
    WHOS_ON_FIRST = enum.auto()
    WIRES = enum.auto()
    WIRE_SEQUENCES = enum.auto()

@torch.no_grad()
def detect_module_type(bgr_image: NDArray) -> BombModuleType:
    image = bgr2tensor(bgr_image)
    return BombModuleType(_MODEL(image).argmax().item())
