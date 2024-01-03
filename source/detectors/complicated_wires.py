from typing import Sequence

import cv2
import numpy as np
from numpy.typing import NDArray
from utils import (crop_image, draw_contour_mask, filter_contours_by_area,
                   find_contours, get_centroid)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

_CR_STAR_BACKGROUND = ((14, 87, 102), (26, 255, 199))
_CR_LIT_LED = ((0, 57, 231), (33, 104, 255))
_CR_UNLIT_LED = ((0, 78, 11), (66, 109, 31))
_CR_RED_WIRE = ((0, 194, 117), (12, 255, 255))
_CR_BLUE_WIRE = ((103, 146, 89), (121, 255, 255))
_CR_WHITE_WIRE = ((0, 0, 136), (32, 63, 255))
_CR_RED_MIX = ((136, 132, 75), (179, 255, 255))

_THRESH_STAR_AREA = 0.9
_THRESH_LIT_LED = 50
_THRESH_UNLIT_LED = 15
_THRESH_DISTANCE = 16

def detect_complicated_wires(hsv_image: NDArray) -> list[bool, bool, str]:
    stars = _detect_stars(hsv_image)
    leds = _detect_leds(hsv_image)
    red_wires = _detect_red_wires(hsv_image)
    blue_wires, blue_mask = _detect_blue_wires(hsv_image)
    white_wires, white_mask = _detect_white_wires(hsv_image)
    
    red_mask = cv2.inRange(hsv_image, *_CR_RED_MIX)
    blue_white_wires = _detect_mixed_wires(blue_mask | white_mask)
    red_white_wires = _detect_mixed_wires(red_mask | white_mask)
    blue_red_wires = _detect_mixed_wires(blue_mask | red_mask)

    wires = [('r', c) for c in red_wires]
    wires += [('b', c) for c in blue_wires]
    wires += [('w', c) for c in white_wires]
    wires += [('bw', c) for c in blue_white_wires]
    wires += [('rw', c) for c in red_white_wires]
    wires += [('br', c) for c in blue_red_wires]
    wires.sort(key=lambda x: x[1][:, :, 0].min())
    
    result = []
    for star, led in zip(stars, leds):
        if not wires:
            result.append((led, star[0], ''))
            continue

        c = wires[0][1]
        bot_pt = c[c[:, :, 1].argmax(), 0, 0]
        bbox = star[1]
        delta = abs(bbox[0]+bbox[2]/2-bot_pt)
        if delta < _THRESH_DISTANCE:
            result.append([led, star[0], wires.pop(0)[0]])
        else:
            result.append([led, star[0], ''])
    return result


def _detect_red_wires(hsv_image: NDArray) -> NDArray:
    contours = find_contours(cv2.inRange(hsv_image, *_CR_RED_WIRE))
    return filter_contours_by_area(contours, 100)

def _detect_blue_wires(hsv_image: NDArray) -> tuple[NDArray, NDArray]:
    mask = cv2.inRange (hsv_image, *_CR_BLUE_WIRE)
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, 100)
    contours = [c for c in contours if c[:, :, 1].max() - c[:, :, 1].min() >= 100]
    mask &= ~cv2.drawContours(np.zeros_like(mask), contours, -1, 255, -1)
    return contours, mask

def _detect_white_wires(hsv_image: NDArray) -> tuple[NDArray, NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_WHITE_WIRE)
    contours = find_contours(mask)
    contours = filter_contours_by_area(contours, 100)
    contours = [c for c in contours if c[:, :, 1].max() - c[:, :, 1].min() >= 100]
    mask &= ~cv2.drawContours(np.zeros_like(mask), contours, -1, 255, -1)
    return contours, mask

def _detect_mixed_wires(mask: NDArray) -> NDArray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, None)
    contours = filter_contours_by_area(find_contours(mask), 100)
    contours = [c for c in contours if c[:, :, 1].max() - c[:, :, 1].min() >= 100]
    return contours

def _detect_stars(hsv_image: NDArray) -> list[tuple[bool, Sequence[int]]]:
    mask = cv2.inRange(hsv_image, *_CR_STAR_BACKGROUND)
    dilated = cv2.dilate(mask, _MORPH_KERNEL)
    contours = sorted(find_contours(dilated), key=cv2.contourArea)[:-7:-1]
    contours.sort(key=lambda c: c[:, :, 0].min())

    result = []
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        star = crop_image(mask, bbox)
        star &= draw_contour_mask(
            np.zeros_like(star), contour, offset=(-bbox[0], -bbox[1]))
        area = cv2.contourArea(cv2.convexHull(np.concatenate(find_contours(star))))
        result.append((cv2.countNonZero(star) / area < _THRESH_STAR_AREA, bbox))
    return result

def _detect_leds(hsv_image: NDArray) -> list[bool]:
    mask = cv2.inRange(hsv_image, *_CR_LIT_LED)
    lit_contours = filter_contours_by_area(find_contours(mask), _THRESH_LIT_LED)
    if len(lit_contours) == 0:
        return [False] * 6
    
    mask = cv2.inRange(hsv_image, *_CR_UNLIT_LED)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, dst=mask)
    unlit_contours = filter_contours_by_area(find_contours(mask), _THRESH_UNLIT_LED)

    lit_centroids = np.array([get_centroid(x) for x in lit_contours])
    unlit_centroids = np.array([get_centroid(x) for x in unlit_contours])
    center = lit_centroids[:, 1].mean()

    leds = [(True, x) for x in lit_centroids]
    for unlit_centroid in unlit_centroids:
        if np.abs(unlit_centroid[1] - center) < 5.0:
            leds.append((False, unlit_centroid))
    leds.sort(key=lambda x: x[1][0])
    return [x[0] for x in leds]
