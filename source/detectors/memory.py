import cv2
import numpy as np
from numpy.typing import NDArray
from utils import bgr2gray, crop_image, draw_contour_mask, find_contours

_CR_ALL_DIGITS = ((0, 0, 141), (24, 255, 255))

def detect_memory(bgr_image: NDArray, hsv_image: NDArray) -> list[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_ALL_DIGITS)
    contours = find_contours(mask)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))[:-6:-1]
    contours.sort(key=lambda x: x[:,:,1].min())

    bbox = cv2.boundingRect(contours[0])
    top_digit = crop_image(mask, bbox)
    top_digit &= draw_contour_mask(
        np.zeros_like(top_digit), contours[0], offset=(-bbox[0], -bbox[1]))
    
    result = [top_digit]
    contours = contours[1:]
    contours.sort(key=lambda x: x[:,:,0].min())
    for contour in contours:
        result.append(_extract_number(bgr_image, contour))
    return result

def _extract_number(bgr_image: NDArray, contour: NDArray) -> NDArray:
    gray = bgr2gray(crop_image(bgr_image, cv2.boundingRect(contour)))
    _, gray = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contour = max(find_contours(gray), key=cv2.contourArea)

    bbox = cv2.boundingRect(contour)
    symbol = crop_image(gray, bbox)
    mask = draw_contour_mask(
        np.zeros_like(symbol), contour, offset=(-bbox[0], -bbox[1]))
    return symbol & mask
