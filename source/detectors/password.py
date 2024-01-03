import cv2
import numpy as np
from numpy.typing import NDArray
from utils import bgr2gray, crop_image, draw_contour_mask, find_contours

_CR_BACKGROUND = ((34, 207, 169), (58, 255, 255))
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_password(bgr_image: NDArray, hsv_image: NDArray) -> list[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_BACKGROUND)
    contour = max(find_contours(mask), key=lambda x: cv2.contourArea(x))

    bbox = cv2.boundingRect(contour)
    gray = bgr2gray(crop_image(bgr_image, bbox))
    mask = draw_contour_mask(
        np.zeros_like(gray), contour, offset=(-bbox[0], -bbox[1]))
    
    _, gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    gray = cv2.morphologyEx(gray & mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)
    contours = sorted(find_contours(gray), key=lambda c: c[:, :, 0].min())

    result = []
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        symbol = crop_image(gray, bbox)
        mask = draw_contour_mask(
            np.zeros_like(symbol), contour, offset=(-bbox[0], -bbox[1]))
        result.append(symbol & mask)
    return result
