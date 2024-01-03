import time
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np


class ScreenCapture:
    def __init__(self):
        self.ctl_shm = SharedMemory('ktane_ctl', size=4)
        self.img_shm = SharedMemory('ktane_img', size=1920*1080*3)

        resource_tracker.unregister(self.ctl_shm._name, 'shared_memory')
        resource_tracker.unregister(self.img_shm._name, 'shared_memory')

        self.ctl = np.ndarray(1, dtype=np.uint32, buffer=self.ctl_shm.buf)
        self.img = np.ndarray((1080, 1920, 3), dtype=np.uint8, buffer=self.img_shm.buf)
    
    def __call__(self):
        self.ctl[0] = 1
        start = time.monotonic_ns()
        while self.ctl[0] != 0:
            time.sleep(0)
            end = time.monotonic_ns()
            if (end - start) >= 2e9:
                raise TimeoutError('Timeout')
        return cv2.flip(self.img, 0)
