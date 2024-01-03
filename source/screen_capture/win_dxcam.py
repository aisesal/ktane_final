import dxcam
import numpy as np

class ScreenCapture:
    def __init__(self):
        self.camera = dxcam.create(output_color='BGR', max_buffer_len=1)
    
    def __call__(self):
        while (screen := self.camera.grab()) is None:
            pass
        return screen
