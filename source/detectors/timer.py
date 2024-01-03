import cv2
from numpy.typing import NDArray
from utils import (crop_image, filter_contours_by_area, find_contours,
                   get_centroid)

_CR_TIMER = ((0, 72, 137), (4, 255, 255))
_CR_OCCLUDED = ((170, 24, 115), (179, 133, 255))
_THRESH_BAR_AREA = 100

_Point = tuple[float, float]

def detect_timer(hsv_image: NDArray) -> list[int]:
    mask = cv2.inRange(hsv_image, *_CR_TIMER)
    mask |= cv2.inRange(hsv_image, *_CR_OCCLUDED)

    mask = crop_image(mask, cv2.boundingRect(mask))
    contours = filter_contours_by_area(find_contours(mask), _THRESH_BAR_AREA)
    contours.sort(key=lambda c: c[:,:,0].min())

    p_horz, p_vert = [], []
    for point in contours:
        _, _, w, h = cv2.boundingRect(point)
        (p_horz if w > h else p_vert).append(get_centroid(point))

    p_htop, p_hmid, p_hbot = [], [], []
    for point in p_horz:
        if point[1] >= mask.shape[0]-20:
            p_hbot.append(point)
        else:
            (p_htop if point[1] < 20 else p_hmid).append(point)
    
    p_vtop, p_vbot = [], []
    for point in p_vert:
        (p_vtop if point[1] < mask.shape[0]/2 else p_vbot).append(point)
    
    result = []
    for i in range(4):
        if i == 2:
            p_vtop.pop(), p_vbot.pop()
        result.append(
            _detect_digit(p_htop, p_hmid, p_hbot, p_vtop, p_vbot))
    return result[::-1]


def _detect_digit(
        p_htop: list[_Point], p_hmid: list[_Point], p_hbot: list[_Point],
        p_vtop: list[_Point], p_vbot: list[_Point]) -> int:
    vtop = p_vtop.pop()
    if p_htop and (delta := p_htop[-1][0] - vtop[0]) > 0:
        p_htop.pop(), p_hmid.pop(), p_hbot.pop(), p_vbot.pop()
        if p_vbot and abs(vtop[0] - p_vbot[-1][0]) <= 10:
            p_vbot.pop()
            return 6
        return 5
    
    if not p_htop or delta < -30:
        p_vbot.pop()
        if p_hmid and vtop[0] - p_hmid[-1][0] <= 30:
            p_hmid.pop(), p_vtop.pop()
            return 4
        return 1
    
    if not p_hmid or vtop[0] - p_hmid[-1][0] > 30:
        p_htop.pop(), p_vbot.pop()
        if p_hbot and abs(vtop[0] - p_hbot[-1][0]) <= 30:
            p_hbot.pop(), p_vtop.pop(), p_vbot.pop()
            return 0
        return 7
    
    htop, _, _ = p_htop.pop(), p_hmid.pop(), p_hbot.pop()
    if abs(vtop[0] - p_vbot.pop()[0]) > 30:
        return 2
    
    if not p_vtop or abs(htop[0] - p_vtop[-1][0] + delta) > 10:
        return 3
    
    p_vtop.pop()
    if not p_vbot or abs(htop[0] - p_vbot[-1][0] + delta) > 15:
        return 9
    
    p_vbot.pop()
    return 8
