from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
from utils import find_contours

_CR_MARKER = ((10, 86, 40), (36, 174, 255))
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_side_border(hsv_screen: NDArray) -> \
        Optional[tuple[int, int, int, int]]:
    mask = cv2.inRange(hsv_screen, *_CR_MARKER)
    cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=2, dst=mask)
    contours, hierarchy = find_contours(mask, cv2.RETR_CCOMP)
    
    index = -1 if len(contours) == 0 else 0
    interests = list[NDArray]()
    while index != -1:
        if _count_children(hierarchy, index) == 2:
            interests.append(contours[index])
        index = hierarchy[0, index, 0]
    
    detected_cnt = len(interests)
    if detected_cnt not in (2, 3):
        return None
    
    extremes = np.array([
        (c[:, :, 0].min(), c[:, :, 0].max(),
         c[:, :, 1].min(), c[:, :, 1].max()) for c in interests])
    
    if detected_cnt == 3:
        extremes = np.delete(extremes, extremes[:,2].argmin(), axis=0)
    
    return extremes[:, :2].min().item(), extremes[:, 2:].min().item(), \
           extremes[:, :2].max().item(), extremes[:, 2:].max().item()

def _count_children(hierarchy: NDArray, index: int) -> int:
    next_child = hierarchy[0, index, 2]
    count = 0
    while next_child != -1:
        count += 1
        next_child = hierarchy[0, next_child, 0]
    return count
