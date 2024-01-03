import typing
from pathlib import Path

import cv2
import numpy as np
import torch
from models.wire_sequences import WireSequences as Model
from numpy.typing import NDArray
from utils import bgr2hsv, bgr2tensor, crop_image

if typing.TYPE_CHECKING:
    from game_state import GameState


_MODEL = Model().eval()
_MODEL.load_state_dict(torch.load(Path('models/WireSequences.pt')))

_UI_WIRES = ((910, 490), (910, 530), (910, 575))
_UI_NEXT_PANEL = (950, 635)

_WIRE_OCCURENCIES = {
    'R': ('C', 'B', 'A', 'AC', 'B', 'AC', 'ABC', 'AB', 'B'),
    'B': ('B', 'AC', 'B', 'A', 'B', 'BC', 'C', 'AC', 'A'),
    'K': ('ABC', 'AC', 'B', 'AC', 'B', 'BC', 'AB', 'C', 'C') }

class WireSequences:
    def __init__(self) -> None:
        pass
    
    def solve(self, state: 'GameState') -> None:
        counts = {'R': 0, 'B': 0, 'K': 0}
        for panel_id in range(4):
            while True:
                bgr_image = state.grab_active_module()
                hsv_image = bgr2hsv(bgr_image)
                if hsv_image[:, :, 2].mean() > 100:
                    break
            
            tensor = bgr2tensor(state.grab_active_module())
            result = torch.stack(_MODEL(tensor)).argmax(dim=-1).squeeze().tolist()
            direction, color = result[:3], result[3:]
            direction = ['XABC'[x] for x in direction]
            color = ['XKBR'[x] for x in color]
            
            do_cut = [False]*3
            for i, (d, c) in enumerate(zip(direction, color)):
                if d == 'X' or c == 'X':
                    continue
                if d in _WIRE_OCCURENCIES[c][counts[c]]:
                    do_cut[i] = True
                counts[c] += 1
            
            for i, cut in enumerate(do_cut):
                if cut:
                    state.mov(*_UI_WIRES[i]).ldn().lup().slp()
            
            state.mov(*_UI_NEXT_PANEL).ldn().lup()
            if panel_id != 3:
                state.slp(2.0)
            else:
                state.slp()
