import cv2
from numpy.typing import NDArray
from utils import filter_contours_by_area, find_contours

_CR_PARALLEL_PORT = ((166, 96, 104), (172, 255, 255))
_THREH_AREA = 1000
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_parallel_ports(hsv_image: NDArray) -> int:
    mask = cv2.inRange(hsv_image, *_CR_PARALLEL_PORT)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, dst=mask)
    contours = filter_contours_by_area(find_contours(mask), _THREH_AREA)
    return len(contours)
