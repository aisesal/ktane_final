import pyautogui

pyautogui.PAUSE = 0

class Mouse:
    def __init__(self):
        pass

    def left_down(self):
        pyautogui.mouseDown(button='left', _pause=False)
    
    def left_up(self):
        pyautogui.mouseUp(button='left', _pause=False)
    
    def right_down(self):
        pyautogui.mouseDown(button='right', _pause=False)
    
    def right_up(self):
        pyautogui.mouseUp(button='right', _pause=False)
    
    def move(self, x, y):
        pyautogui.moveTo(x, y, _pause=False)
