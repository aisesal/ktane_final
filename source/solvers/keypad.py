import enum
import typing
from pathlib import Path

import torch
from detectors.keypad import detect_keypad
from models.symbols import Symbols as Model
from numpy.typing import NDArray
from utils import bgr2hsv, fit_image_size, gray2tensor, tensors2batch

if typing.TYPE_CHECKING:
    from game_state import GameState

_MODEL = Model(27).eval()
_MODEL.load_state_dict(torch.load(Path('models/KeypadSymbols.pt')))

_UI_BUTTONS = ((905, 510), (1005, 510), (905, 600), (1005, 600))

class _KeypadSymbol(enum.IntEnum):
    AE = 0
    AT = enum.auto()
    BALLOON = enum.auto()
    BT = enum.auto()
    COPYRIGHT = enum.auto()
    CURSIVE = enum.auto()
    DOUBLEK = enum.auto()
    DRAGON = enum.auto()
    EURO = enum.auto()
    FILLEDSTAR = enum.auto()
    HOLLOWSTAR = enum.auto()
    HOOKN = enum.auto()
    LEFTC = enum.auto()
    MELTEDTHREE = enum.auto()
    NWITHHAT = enum.auto()
    OMEGA = enum.auto()
    PARAGRAPH = enum.auto()
    PITCHFORK = enum.auto()
    PUMPKIN = enum.auto()
    QUESTIONMARK = enum.auto()
    RIGHTC = enum.auto()
    SIX = enum.auto()
    SMILEYFACE = enum.auto()
    SQUIDKNIFE = enum.auto()
    SQUIGGLYN = enum.auto()
    TRACKS = enum.auto()
    UPSIDEDOWNY = enum.auto()

_COLUMNS = (
    (
        _KeypadSymbol.BALLOON,
        _KeypadSymbol.AT,
        _KeypadSymbol.UPSIDEDOWNY,
        _KeypadSymbol.SQUIGGLYN,
        _KeypadSymbol.SQUIDKNIFE,
        _KeypadSymbol.HOOKN,
        _KeypadSymbol.LEFTC
    ),
    (
        _KeypadSymbol.EURO,
        _KeypadSymbol.BALLOON,
        _KeypadSymbol.LEFTC,
        _KeypadSymbol.CURSIVE,
        _KeypadSymbol.HOLLOWSTAR,
        _KeypadSymbol.HOOKN,
        _KeypadSymbol.QUESTIONMARK
    ),
    (
        _KeypadSymbol.COPYRIGHT,
        _KeypadSymbol.PUMPKIN,
        _KeypadSymbol.CURSIVE,
        _KeypadSymbol.DOUBLEK,
        _KeypadSymbol.MELTEDTHREE,
        _KeypadSymbol.UPSIDEDOWNY,
        _KeypadSymbol.HOLLOWSTAR,
    ),
    (
        _KeypadSymbol.SIX,
        _KeypadSymbol.PARAGRAPH,
        _KeypadSymbol.BT,
        _KeypadSymbol.SQUIDKNIFE,
        _KeypadSymbol.DOUBLEK,
        _KeypadSymbol.QUESTIONMARK,
        _KeypadSymbol.SMILEYFACE
    ),
    (
        _KeypadSymbol.PITCHFORK,
        _KeypadSymbol.SMILEYFACE,
        _KeypadSymbol.BT,
        _KeypadSymbol.RIGHTC,
        _KeypadSymbol.PARAGRAPH,
        _KeypadSymbol.DRAGON,
        _KeypadSymbol.FILLEDSTAR
    ),
    (
        _KeypadSymbol.SIX,
        _KeypadSymbol.EURO,
        _KeypadSymbol.TRACKS,
        _KeypadSymbol.AE,
        _KeypadSymbol.PITCHFORK,
        _KeypadSymbol.NWITHHAT,
        _KeypadSymbol.OMEGA
    ),
)

class Keypad:
    def __init__(self, bgr_image: NDArray) -> None:
        images = detect_keypad(bgr_image, bgr2hsv(bgr_image))
        batch = [gray2tensor(fit_image_size(x, 64, 64)) for x in images]
        batch = tensors2batch(batch)
        prediction = _MODEL(batch).argmax(dim=1).tolist()
        self._symbols = [_KeypadSymbol(x) for x in prediction]
    
    def solve(self, state: 'GameState') -> None:
        column = next(filter(lambda x, y=set(self._symbols): y.issubset(x), _COLUMNS))
        indices = [column.index(x) for x in self._symbols]
        for i in sorted(range(4), key=lambda i: indices[i]):
            state.mov(*_UI_BUTTONS[i]).ldn().lup().slp()
