import cv2
from numpy.typing import NDArray
from utils import filter_contours_by_area, find_contours

_CR_BATTERY = ((16, 178, 171), (27, 255, 255))
_THRESH_AREA = 50
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_batteries(hsv_image: NDArray) -> int:
    mask = cv2.inRange(hsv_image, *_CR_BATTERY)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, dst=mask)
    contours = filter_contours_by_area(find_contours(mask), _THRESH_AREA)
    return len(contours)//2
