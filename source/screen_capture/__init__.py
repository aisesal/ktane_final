import platform

if platform.system() == 'Linux':
    from .linux_hook import ScreenCapture
elif platform.system() == 'Windows':
    from .win_dxcam import ScreenCapture
else:
    raise ImportError('Unsupported OS')
