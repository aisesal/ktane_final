import platform

if platform.system() == 'Linux' or platform.system() == 'Windows':
    from .pyautogui_mouse import Mouse
else:
    raise ImportError('OS is not supported.')
