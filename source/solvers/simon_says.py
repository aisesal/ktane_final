import typing

import cv2
from detectors.simon_says import SimonSaysColor, detect_simon_says

if typing.TYPE_CHECKING:
    from game_state import GameState

_UI_SQUARES = {
    SimonSaysColor.BLUE: (138+832, 69+387),
    SimonSaysColor.GREEN: (138+832, 187+387),
    SimonSaysColor.RED: (78+832, 129+387),
    SimonSaysColor.YELLOW: (200+832, 129+387),
}

_MAPPING = {
    # Serial contains vowel, no strikes.
    (True, 0, SimonSaysColor.RED): SimonSaysColor.BLUE,
    (True, 0, SimonSaysColor.BLUE): SimonSaysColor.RED,
    (True, 0, SimonSaysColor.GREEN): SimonSaysColor.YELLOW,
    (True, 0, SimonSaysColor.YELLOW): SimonSaysColor.GREEN,

    # Serial contains vowel, one strike.
    (True, 1, SimonSaysColor.RED): SimonSaysColor.YELLOW,
    (True, 1, SimonSaysColor.BLUE): SimonSaysColor.GREEN,
    (True, 1, SimonSaysColor.GREEN): SimonSaysColor.BLUE,
    (True, 1, SimonSaysColor.YELLOW): SimonSaysColor.RED,

    # Serial contains vowel, two strikes.
    (True, 2, SimonSaysColor.RED): SimonSaysColor.GREEN,
    (True, 2, SimonSaysColor.BLUE): SimonSaysColor.RED,
    (True, 2, SimonSaysColor.GREEN): SimonSaysColor.YELLOW,
    (True, 2, SimonSaysColor.YELLOW): SimonSaysColor.BLUE,

    # Serial does not contain a vowel, no strikes.
    (False, 0, SimonSaysColor.RED): SimonSaysColor.BLUE,
    (False, 0, SimonSaysColor.BLUE): SimonSaysColor.YELLOW,
    (False, 0, SimonSaysColor.GREEN): SimonSaysColor.GREEN,
    (False, 0, SimonSaysColor.YELLOW): SimonSaysColor.RED,

    # Serial does not contain a vowel, one strike.
    (False, 1, SimonSaysColor.RED): SimonSaysColor.RED,
    (False, 1, SimonSaysColor.BLUE): SimonSaysColor.BLUE,
    (False, 1, SimonSaysColor.GREEN): SimonSaysColor.YELLOW,
    (False, 1, SimonSaysColor.YELLOW): SimonSaysColor.GREEN,

    # Serial does not contain a vowel, two strikes.
    (False, 2, SimonSaysColor.RED): SimonSaysColor.YELLOW,
    (False, 2, SimonSaysColor.BLUE): SimonSaysColor.GREEN,
    (False, 2, SimonSaysColor.GREEN): SimonSaysColor.BLUE,
    (False, 2, SimonSaysColor.YELLOW): SimonSaysColor.RED,
}

class SimonSays:
    def __init__(self):
        pass
    
    def solve(self, state: 'GameState') -> None:
        self._has_vowel = not set('AEIOU').isdisjoint(state.serial)
        self._num_strikes = 0
        self._sequence = list[SimonSaysColor]()
        cur_step = 0

        is_solved = False
        while not is_solved:
            image = state.grab_active_module()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            color, is_solved = detect_simon_says(image)

            if is_solved:
                break
            if color == SimonSaysColor.NONE:
                continue

            if cur_step != len(self._sequence):
                while color != SimonSaysColor.NONE and not is_solved:
                    image = state.grab_active_module()
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    color, is_solved = detect_simon_says(image)
                cur_step += 1
                continue

            cur_step = 0
            self._sequence.append(color)
            self._solve_step(state)
    
    def _solve_step(self, state: 'GameState') -> None:
        for color in self._sequence:
            other_color = _MAPPING[(
                self._has_vowel, self._num_strikes, color)]
            state.mov(*_UI_SQUARES[other_color]).ldn().lup().slp()
        state.mov(885, 400).slp(1.25)
