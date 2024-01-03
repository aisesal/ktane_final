import time

import cv2
from detectors.battery import detect_batteries
from detectors.bomb_modules import BombModuleType, detect_module_type
from detectors.indicator import detect_indicators
from detectors.parallel_port import detect_parallel_ports
from detectors.serial import detect_serial
from detectors.side_border import detect_side_border
from mouse import Mouse
from screen_capture import ScreenCapture
from solvers import get_solver
from typing_extensions import Self

_BOMB_MODULE_BBOXES = (
    (560, 291, 262, 243),
    (849, 291, 251, 243),
    (1127, 291, 255, 243),
    (545, 558, 272, 262),
    (843, 558, 262, 262),
    (1133, 558, 266, 262))
_BOMB_ZOOMED_MODULE_BBOX = (832, 387, 297, 293)

_UI_TIME_ARROWS = ((720, 315), (860, 315))
_UI_MODULE_ARROWS = ((720, 400), (860, 400))
_UI_NEEDY_FLIP = (750, 540)
_UI_HARDCORE_FLIP = (750, 600)
_UI_START_BUTTON = (820, 750)

_UI_MODULE_SPOTS = (
    (695, 410),
    (970, 410),
    (1235, 410),
    (680, 680),
    (950, 680),
    (1250, 680))

class GameState:
    def __init__(self) -> None:
        self._mouse = Mouse()
        self._screen_capture = ScreenCapture()

        self._time_limit_ndx = 9 # max=19
        self._module_count_ndx = 0 # max 8
        self._needy_enabled = False
        self._hardcore_enabled = False
    
    def slp(self, seconds: float = 1/30) -> Self:
        time.sleep(seconds)
        return self
    
    def mov(self, x: int, y: int) -> Self:
        self._mouse.move(x, y)
        return self
    
    def ldn(self) -> Self:
        self._mouse.left_down()
        return self
    
    def lup(self) -> Self:
        self._mouse.left_up()
        return self
    
    def rdn(self) -> Self:
        self._mouse.right_down()
        return self
    
    def rup(self) -> Self:
        self._mouse.right_up()
        return self
    
    def grab_screen(self):
        return self._screen_capture()
    
    def grab_active_module(self):
        image = self._screen_capture()
        bbox = _BOMB_ZOOMED_MODULE_BBOX
        return image[bbox[1]:bbox[1]+bbox[3],
                     bbox[0]:bbox[0]+bbox[2]]
    
    def open_free_play(self) -> None:
        self.mov(500, 570).ldn().lup().slp(0.5)
    
    def set_settings(self,
                     time_limit_ndx: int,
                     module_count_ndx: int,
                     needy_enabled: bool,
                     hardcore_enabled: bool) -> None:
        
        self._time_limit_ndx = self._change_range_setting(
            _UI_TIME_ARROWS, self._time_limit_ndx, time_limit_ndx)
        
        self._module_count_ndx = self._change_range_setting(
            _UI_MODULE_ARROWS, self._module_count_ndx, module_count_ndx)
        
        if self._needy_enabled != needy_enabled:
            self.mov(*_UI_NEEDY_FLIP).ldn().lup().slp()
            self._needy_enabled = needy_enabled
        
        if self._hardcore_enabled != hardcore_enabled:
            self.mov(*_UI_HARDCORE_FLIP).ldn().lup().slp()
            self._hardcore_enabled = hardcore_enabled

    def start_game(self) -> None:
        self._serial = None
        self._indicators = []
        self._parallel_port_count = 0
        self._battery_count = 0
        self._front_modules = []
        self._back_modules = []
        self._timer_position = 0

        self.mov(*_UI_START_BUTTON).ldn().lup().slp(14)
        self._inspect_bomb()
    
    def _inspect_bomb(self) -> None:
        self.mov(960, 540).ldn().lup().slp()
        
        self.mov(960, 540).rdn().slp()
        self.mov(960, 880).slp(0.75)
        self._inspect_side('top')

        self.mov(960, 200).slp(0.5).rup()
        self._inspect_side('bot')

        self.rdn().rup().slp()
        self.mov(960, 540).ldn().lup().slp()
        self.mov(200, 200).slp(1.25)
        self._inspect_side('frn')

        self.mov(960, 540).rdn().slp()
        self.mov(960+255, 540).slp(0.5).rup()
        self._inspect_side('lft')

        self.rdn().rup().slp()
        self.mov(960, 540).ldn().lup().slp()
        self.rdn().slp()
        self.mov(960-265, 540).slp(1.25)
        self._inspect_side('rgh')

        self.mov(960-400, 540).rup().slp()
        self.rdn().rup().slp()
        self.mov(960, 540).ldn().lup().slp()
        self.mov(200, 200).slp(1.25)

        self._inspect_side('bck')
        
        self._solve(self._back_modules)

        self.mov(960, 540).rdn().slp()
        self.mov(960+400, 540).slp(0.5).rup().slp()
        self.rdn().rup().slp(0.1)
        self.mov(960, 540).ldn().lup().slp(1.5)

        self._solve(self._front_modules)
    
    def _inspect_side(self, side: str) -> None:
        bgr_screen = self.grab_screen()
        if side in ('bck', 'frn'):
            modules_list = [self._front_modules, self._back_modules]
            modules_list = modules_list[side == 'bck'] 
            for pos, bbox in enumerate(_BOMB_MODULE_BBOXES):
                bgr_image = bgr_screen[bbox[1]:bbox[1]+bbox[3],
                                       bbox[0]:bbox[0]+bbox[2]]
                module_type = detect_module_type(bgr_image)
                if module_type == BombModuleType.TIMER:
                    self._timer_position = pos
                modules_list.append(get_solver(module_type, pos, bgr_image))
        else:
            hsv_screen = cv2.cvtColor(bgr_screen, cv2.COLOR_BGR2HSV)
            x0, y0, x1, y1 = detect_side_border(hsv_screen)
            bgr_screen = bgr_screen[y0:y1, x0:x1]
            hsv_screen = hsv_screen[y0:y1, x0:x1]

            self._battery_count += detect_batteries(hsv_screen)
            self._parallel_port_count += detect_parallel_ports(hsv_screen)
            
            if side == 'lft':
                bgr_screen = cv2.rotate(bgr_screen, cv2.ROTATE_90_COUNTERCLOCKWISE)
                hsv_screen = cv2.rotate(hsv_screen, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif side == 'rgh':
                bgr_screen = cv2.rotate(bgr_screen, cv2.ROTATE_90_CLOCKWISE)
                hsv_screen = cv2.rotate(hsv_screen, cv2.ROTATE_90_CLOCKWISE)
            elif side == 'top':
                bgr_screen = cv2.rotate(bgr_screen, cv2.ROTATE_180)
                hsv_screen = cv2.rotate(hsv_screen, cv2.ROTATE_180)
            
            for indicator in detect_indicators(bgr_screen, hsv_screen):
                self._indicators.append(indicator)

            if self._serial is None:
                serial = detect_serial(bgr_screen, hsv_screen)
                if serial is not None:
                    self._serial = serial
    
    def _solve(self, module_list):
        for ndx, module in enumerate(module_list):
            if module is None:
                continue
            self.mov(*_UI_MODULE_SPOTS[ndx]).ldn().lup().slp()
            self.mov(*((1720, 200) if ndx == 5 else (200, 200))).slp(0.75)
            module.solve(self)
            self.rdn().rup().slp(1)

    def _change_range_setting(self,
                              arrows: tuple[tuple[int, int]],
                              current_value: int,
                              new_value: int) -> int:
        diff = new_value - current_value
        if diff != 0:
            self.mov(*arrows[diff > 0])
            for _ in range(abs(diff)):
                self.ldn().lup().slp()
        return new_value

    @property
    def serial(self) -> str:
        return self._serial
    
    @property
    def indicators(self) -> tuple[tuple[bool, str], ...]:
        return self._indicators
    
    @property
    def parallel_port_count(self) -> int:
        return self._parallel_port_count
    
    @property
    def battery_count(self) -> int:
        return self._battery_count

    @property
    def timer_position(self) -> int:
        return self._timer_position
