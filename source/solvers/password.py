import typing

import cv2
import numpy as np
import torch
from detectors.password import detect_password
from models.symbols import Symbols
from numpy.typing import NDArray
from utils import bgr2hsv, gray2tensor, tensors2batch, fit_image_size

if typing.TYPE_CHECKING:
    from game_state import GameState

_MODEL = Symbols(26).eval()
_MODEL.load_state_dict(torch.load('models/PasswordSymbols.pt'))

_UI_UP_ARROWS = ((890, 460), (932, 460), (972, 460), (1014, 460), (1055, 460))
_UI_DN_ARROWS = ((890, 605), (932, 605), (972, 605), (1014, 605), (1055, 605))
_UI_SUBMIT = (960, 640)

_TABLE = (
    'about', 'after', 'again', 'below', 'could',
    'every', 'first', 'found', 'great', 'house',
    'large', 'learn', 'never', 'other', 'place',
    'plant', 'point', 'right', 'small', 'sound',
    'spell', 'still', 'study', 'their', 'there',
    'these', 'thing', 'think', 'three', 'water',
    'where', 'which', 'world', 'would', 'write'
)

class Password:
    def __init__(self, bgr_image: NDArray) -> None:
        hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        letters = detect_password(bgr_image, hsv_image)
        self._state = self._detect_text(letters)
    
    def _detect_text(self, images):
        batch = [gray2tensor(fit_image_size(image, 64, 64))
                 for image in images]
        with torch.no_grad():
            prediction = _MODEL(tensors2batch(batch)).argmax(dim=1).tolist()
            return ['abcdefghijklmnopqrstuvwxyz'[x] for x in prediction]

    def solve(self, state: 'GameState') -> None:
        for _ in range(5):
            state.mov(*_UI_DN_ARROWS[0])
            for pos in _UI_DN_ARROWS[1:]:
                state.ldn().lup().slp().mov(*pos)
            state.ldn().lup().slp()

            bgr_image = state.grab_active_module()
            hsv_image = bgr2hsv(bgr_image)
            letters = detect_password(bgr_image, hsv_image)
            letters = self._detect_text(letters)
            self._state = [y+x for x, y in zip(self._state, letters)]

            result = None
            for word in _TABLE:
                for word_letter, state_letters in zip(word, self._state):
                    if word_letter not in state_letters:
                        break
                else:
                    result = word
                    break

            if result is None:
                continue

            indices = [2-(x.index(y)+2)%6 for x, y in zip(self._state, result)]
            for position, count in enumerate(indices):
                if count == 0:
                    continue

                arrow = _UI_UP_ARROWS if count < 0 else _UI_DN_ARROWS
                state.mov(*arrow[position])
                for _ in range(abs(count)):
                    state.ldn().lup().slp()
            break
        state.mov(*_UI_SUBMIT).ldn().lup()
