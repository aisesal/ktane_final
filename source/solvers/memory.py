import typing
from pathlib import Path

import torch
from detectors.memory import detect_memory
from models.symbols import Symbols
from utils import bgr2hsv, fit_image_size, gray2tensor, tensors2batch

if typing.TYPE_CHECKING:
    from game_state import GameState

_MODEL = Symbols(4).eval()
_MODEL.load_state_dict(torch.load(Path('models/MemorySymbols.pt')))

_UI_BUTTONS = ((885, 600), (925, 600), (970, 600), (1015, 600))

class Memory:
    def __init__(self) -> None:
        self._buttons = list[tuple[int, int, int, int]]()
        self._presses = list[int]()
    
    def _get_label(self, stage):
        return self._buttons[stage][self._presses[stage]]

    def solve(self, state: 'GameState') -> None:
        for stage in range(5):
            while True:
                bgr_image = state.grab_active_module()
                hsv_image = bgr2hsv(bgr_image)
                if hsv_image[:, :, 2].mean() > 100:
                    break

            digits = detect_memory(bgr_image, hsv_image)

            batch = [gray2tensor(fit_image_size(x, 64, 64)) for x in digits]
            display, *buttons = (_MODEL(tensors2batch(batch)).argmax(dim=1)+1).tolist()
            
            if stage == 0:
                pressed = display-1 if display > 2 else 1
            elif stage == 1:
                if display == 1:
                    pressed = buttons.index(4)
                elif display == 3:
                    pressed = 0
                else:
                    pressed = self._presses[0]
            elif stage == 2:
                if display == 1:
                    pressed = buttons.index(self._get_label(1))
                elif display == 2:
                    pressed = buttons.index(self._get_label(0))
                elif display == 3:
                    pressed = 2
                elif display == 4:
                    pressed = buttons.index(4)
            elif stage == 3:
                if display == 1:
                    pressed = self._presses[0]
                elif display == 2:
                    pressed = 0
                else:
                    pressed = self._presses[1]
            elif stage == 4:
                if display == 1:
                    pressed = buttons.index(self._get_label(0))
                elif display == 2:
                    pressed = buttons.index(self._get_label(1))
                elif display == 3:
                    pressed = buttons.index(self._get_label(3))
                elif display == 4:
                    pressed = buttons.index(self._get_label(2))
            
            self._buttons.append(tuple(buttons))
            self._presses.append(pressed)

            state.mov(*_UI_BUTTONS[pressed]).ldn().lup().slp()
            if stage != 4:
                state.mov(885, 400).slp(3.25)
