import cv2
import numpy as np
from numpy.typing import NDArray
from utils import bgr2gray, crop_image, draw_contour_mask, find_contours

_CR_BACKGROUND = ((0, 0, 194), (30, 46, 255))

def detect_keypad(bgr_image: NDArray, hsv_image: NDArray) -> list[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_BACKGROUND)
    contours = sorted(find_contours(mask), key=cv2.contourArea)[:-5:-1]

    result = []
    contours.sort(key=lambda x: x[:,:,1].min())
    top = sorted(contours[:2], key=lambda x: x[:,:,0].min())
    bot = sorted(contours[2:], key=lambda x: x[:,:,0].min())
    for contour in top+bot:
        result.append(_extract_symbol(bgr_image, contour))
    return result

def _extract_symbol(bgr_image: NDArray, contour: NDArray) -> NDArray:
    bbox = cv2.boundingRect(contour)
    gray = bgr2gray(crop_image(bgr_image, bbox))
    mask = draw_contour_mask(
        np.zeros_like(gray), contour, offset=(-bbox[0], -bbox[1]))

    _, gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    gray &= mask
    
    return crop_image(gray, cv2.boundingRect(gray))
