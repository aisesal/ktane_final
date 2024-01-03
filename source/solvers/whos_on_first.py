import typing
from pathlib import Path

import torch
from detectors.whos_on_first import detect_whos_on_first
from models.whos_on_first import WhosOnFirst as Model
from utils import bgr2hsv, gray2tensor, tensors2batch, fit_image_size

if typing.TYPE_CHECKING:
    from game_state import GameState

_UI_BUTTONS = (
    (910, 515), (995, 515), (910, 565), (995, 565), (910, 610), (995, 610))

_MODEL = Model().eval()
_MODEL.load_state_dict(torch.load(Path('models/WhosOnFirstText.pt')))
_LABELS = (
    'blank', 'c', 'cee', 'display', 'done', 'first', 'hold', 'hold on',
    'lead', 'led', 'leed', 'left', 'like', 'middle', 'next', 'no',
    'nothing', '', 'okay', 'press', 'read', 'ready', 'red', 'reed',
    'right', 'says', 'see', 'sure', 'their', 'there', 'they are',
    "they're", 'u', 'uhhh', 'uh huh', 'uh uh', 'ur', 'wait', 'what',
    'what?', 'yes', 'you', 'your', 'you are', "you're",
)

_DISPLAY_TABLE = {
    'yes': 2, 'first': 1, 'display': 5, 'okay': 1, 'says': 5, 'nothing': 2, '': 4,
    'blank': 3, 'no': 5, 'led': 2, 'lead': 5, 'read': 3, 'red': 3, 'reed': 4,
    'leed': 4, 'hold on': 5, 'you': 3, 'you are': 5, 'your': 3, "you're": 3, 'ur': 0,
    'there': 5, "they're": 4, 'their': 3, 'they are': 2, 'see': 5, 'c': 1, 'cee': 5,
}

_LABEL_TABLE: dict[str, tuple[str, ...]] = {
    'ready': ('yes', 'okay', 'what', 'middle', 'left', 'press', 'right', 'blank', 'ready', 'no', 'first', 'uhhh', 'nothing', 'wait'),
    'first': ('left', 'okay', 'yes', 'middle', 'no', 'right', 'nothing', 'uhhh', 'wait', 'ready', 'blank', 'what', 'press', 'first'),
    'no': ('blank', 'uhhh', 'wait', 'first', 'what', 'ready', 'right', 'yes', 'nothing', 'left', 'press', 'okay', 'no', 'middle'),
    'blank': ('wait', 'right', 'okay', 'middle', 'blank', 'press', 'ready', 'nothing', 'no', 'what', 'left', 'uhhh', 'yes', 'first'),
    'nothing': ('uhhh', 'right', 'okay', 'middle', 'yes', 'blank', 'no', 'press', 'left', 'what', 'wait', 'first', 'nothing', 'ready'),
    'yes': ('okay', 'right', 'uhhh', 'middle', 'first', 'what', 'press', 'ready', 'nothing', 'yes', 'left', 'blank', 'no', 'wait'),
    'what': ('uhhh', 'what', 'left', 'nothing', 'ready', 'blank', 'middle', 'no', 'okay', 'first', 'wait', 'yes', 'press', 'right'),
    'uhhh': ('ready', 'nothing', 'left', 'what', 'okay', 'yes', 'right', 'no', 'press', 'blank', 'uhhh', 'middle', 'wait', 'first'),
    'left': ('right', 'left', 'first', 'no', 'middle', 'yes', 'blank', 'what', 'uhhh', 'wait', 'press', 'ready', 'okay', 'nothing'),
    'right': ('yes', 'nothing', 'ready', 'press', 'no', 'wait', 'what', 'right', 'middle', 'left', 'uhhh', 'blank', 'okay', 'first'),
    'middle': ('blank', 'ready', 'okay', 'what', 'nothing', 'press', 'no', 'wait', 'left', 'middle', 'right', 'first', 'uhhh', 'yes'),
    'okay': ('middle', 'no', 'first', 'yes', 'uhhh', 'nothing', 'wait', 'okay', 'left', 'ready', 'blank', 'press', 'what', 'right'),
    'wait': ('uhhh', 'no', 'blank', 'okay', 'yes', 'left', 'first', 'press', 'what', 'wait', 'nothing', 'ready', 'right', 'middle'),
    'press': ('right', 'middle', 'yes', 'ready', 'press', 'okay', 'nothing', 'uhhh', 'blank', 'left', 'first', 'what', 'no', 'wait'),
    'you': ('sure', 'you are', 'your', "you're", 'next', 'uh huh', 'ur', 'hold', 'what?', 'you', 'uh uh', 'like', 'done', 'u'),
    'you are': ('your', 'next', 'like', 'uh huh', 'what?', 'done', 'uh uh', 'hold', 'you', 'u', "you're", 'sure', 'ur', 'you are'),
    'your': ('uh uh', 'you are', 'uh huh', 'your', 'next', 'ur', 'sure', 'u', "you're", 'you', 'what?', 'hold', 'like', 'done'),
    "you're": ('you', "you're", 'ur', 'next', 'uh uh', 'you are', 'u', 'your', 'what?', 'uh huh', 'sure', 'done', 'like', 'hold'),
    'ur': ('done', 'u', 'ur', 'uh huh', 'what?', 'sure', 'your', 'hold', "you're", 'like', 'next', 'uh uh', 'you are', 'you'),
    'u': ('uh huh', 'sure', 'next', 'what?', "you're", 'ur', 'uh uh', 'done', 'u', 'you', 'like', 'hold', 'you are', 'your'),
    'uh huh': ('uh huh', 'your', 'you are', 'you', 'done', 'hold', 'uh uh', 'next', 'sure', 'like', "you're", 'ur', 'u', 'what?'),
    'uh uh': ('ur', 'u', 'you are', "you're", 'next', 'uh uh', 'done', 'you', 'uh huh', 'like', 'your', 'sure', 'hold', 'what?'),
    'what?': ('you', 'hold', "you're", 'your', 'u', 'done', 'uh uh', 'like', 'you are', 'uh huh', 'ur', 'next', 'what?', 'sure'),
    'done': ('sure', 'uh huh', 'next', 'what?', 'your', 'ur', "you're", 'hold', 'like', 'you', 'u', 'you are', 'uh uh', 'done'),
    'next': ('what?', 'uh huh', 'uh uh', 'your', 'hold', 'sure', 'next', 'like', 'done', 'you are', 'ur', "you're", 'u', 'you'),
    'hold': ('you are', 'u', 'done', 'uh uh', 'you', 'ur', 'sure', 'what?', "you're", 'next', 'hold', 'uh huh', 'your', 'like'),
    'sure': ('you are', 'done', 'like', "you're", 'you', 'hold', 'uh huh', 'ur', 'sure', 'u', 'what?', 'next', 'your', 'uh uh'),
    'like': ("you're", 'next', 'u', 'ur', 'hold', 'done', 'uh uh', 'what?', 'uh huh', 'you', 'like', 'sure', 'you are', 'your'),
}

class WhosOnFirst:
    def __init__(self) -> None:
        pass

    def _solve_step(self, state: 'GameState') -> None:
        while True:
            bgr_image = state.grab_active_module()
            hsv_image = bgr2hsv(bgr_image)
            if hsv_image[:, :, 2].mean() > 100:
                break

        batch = [gray2tensor(fit_image_size(image, 128, 64))
                 for image in detect_whos_on_first(bgr_image, hsv_image)]
        with torch.no_grad():
            prediction = _MODEL(tensors2batch(batch)).argmax(dim=1).tolist()
        display, *labels = [_LABELS[x] for x in prediction]

        row = _LABEL_TABLE[labels[_DISPLAY_TABLE[display]]]
        indices = [row.index(x) for x in labels]
        position = sorted(range(6), key=lambda i: indices[i])[0]
        state.mov(*_UI_BUTTONS[position]).ldn().lup().slp()
        state.mov(885, 400)

    def solve(self, state: 'GameState') -> None:
        self._solve_step(state)
        state.slp(3.5)
        self._solve_step(state)
        state.slp(3.5)
        self._solve_step(state)
