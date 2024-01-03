from itertools import chain

import cv2
import numpy as np
from numpy.typing import NDArray
from utils import bgr2gray, crop_image, draw_contour_mask, find_contours

_CR_TOP_WORD_BG = ((88, 41, 80), (111, 110, 126))
_CR_BUTTON_BG = ((0, 0, 135), (24, 126, 255))

def detect_whos_on_first(bgr_image: NDArray, hsv_image: NDArray) -> list[str]:
    result = [_extract_top_word(bgr_image, hsv_image)]
    result += _extract_buttons(hsv_image)
    return result

def _extract_top_word(
        bgr_image: NDArray, hsv_image: NDArray) -> NDArray:
    mask = cv2.inRange(hsv_image, *_CR_TOP_WORD_BG)
    interests = None
    for contour in find_contours(mask):
        bbox = cv2.boundingRect(contour)
        if bbox[2] > bbox[3] and cv2.contourArea(contour) >= 1500:
            interests = contour
            break

    mask = crop_image(mask, bbox)
    result = bgr2gray(crop_image(bgr_image, bbox))
    hull = cv2.convexHull(interests)
    mask = draw_contour_mask(
        np.zeros_like(mask), hull, offset=(-bbox[0], -bbox[1]))
    _, text = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    text &= mask
    return crop_image(text, cv2.boundingRect(text))

def _extract_buttons(hsv_image: NDArray) -> list[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_BUTTON_BG)
    contours: list[NDArray] = sorted(find_contours(mask), key=cv2.contourArea)[:-7:-1]
    contours.sort(key=lambda c: c[:, :, 0].min())
    lside = sorted(contours[:3], key=lambda c: c[:, :, 1].min())
    rside = sorted(contours[3:], key=lambda c: c[:, :, 1].min())

    result = []
    for contour in chain(*zip(lside, rside)):
        bbox = cv2.boundingRect(contour)
        sub_mask0 = crop_image(mask, bbox)
        sub_mask1 = draw_contour_mask(
            np.zeros_like(sub_mask0), contour, offset=(-bbox[0], -bbox[1]))
        text = sub_mask1 ^ sub_mask0
        result.append(crop_image(text, cv2.boundingRect(text)))
    return result
