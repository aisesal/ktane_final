from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from models.symbols import Symbols as Model
from numpy.typing import NDArray
from utils import (bgr2gray, crop_image, draw_contour_mask,
                   filter_contours_by_area, find_contours, fit_image_size,
                   gray2tensor, tensors2batch)

_MODEL = Model(16).eval()
_MODEL.load_state_dict(torch.load(Path('models/IndicatorSymbols.pt')))
_LABELS = 'ABCDFGIKLMNOQRST'

_CR_BORDER = ((0, 30, 0), (6, 255, 232))
_CR_LETTERS = ((18, 0, 191), (179, 30, 255))
_THRESH_LETTER = 50
_THRESH_BORDER_AREA = 100
_THREH_BLOB_COLOR = 160
_THRESH_BLOB_AREA = 100
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def detect_indicators(
        bgr_image: NDArray,
        hsv_image: NDArray) -> list[tuple[bool | str]]:
    indicators = _filter(bgr_image, hsv_image, _find(hsv_image))
    if not indicators:
        return []
    
    is_lit = []
    batch = []
    for indicator in indicators:
        is_lit.append(indicator[0])
        for image in indicator[1:]:
            batch.append(gray2tensor(fit_image_size(image, 64, 64)))
    batch = tensors2batch(batch)
    prediction = _MODEL(batch).argmax(dim=1).tolist()

    result = []
    for i, lit in enumerate(is_lit):
        text = ''.join(_LABELS[x] for x in prediction[i * 3 : i * 3 + 3])
        result.append((lit, text))
    return result

def _find(hsv_image: NDArray) -> Sequence[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_BORDER)
    cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL, dst=mask)
    return find_contours(mask)

def _filter(
        bgr_image: NDArray, hsv_image: NDArray,
        contours: Sequence[NDArray]) -> list[list[bool | NDArray]]:
    result = []
    for contour in contours:
        if cv2.contourArea(contour) < _THRESH_BORDER_AREA:
            continue

        bbox = cv2.boundingRect(contour)
        mask = cv2.inRange(crop_image(hsv_image, bbox), *_CR_LETTERS)
        contours = filter_contours_by_area(
            find_contours(mask), _THRESH_LETTER)
        if len(contours) != 3:
            continue

        gray = bgr2gray(crop_image(bgr_image, bbox))
        mask = draw_contour_mask(
            np.zeros_like(gray), contour, offset=(-bbox[0], -bbox[1]))
        _, blobs = cv2.threshold(
            gray & mask, _THREH_BLOB_COLOR, 255, cv2.THRESH_BINARY)
        result += _filter_blobs(blobs)
    return result

def _filter_blobs(blobs: NDArray) -> list[list[bool | NDArray]]:
    contours = filter_contours_by_area(
        find_contours(blobs), _THRESH_BLOB_AREA)
    if len(contours) not in (3, 4):
        return []
    
    contours.sort(key=lambda x: x[:,:,0].min())
    if len(contours) == 3:
        result = [False]
    else:
        result = [True]
        contours = contours[1:]
    
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        symbol = crop_image(blobs, bbox)
        mask = draw_contour_mask(
            np.zeros_like(symbol), contour, offset=(-bbox[0], -bbox[1]))
        result.append(symbol & mask)
    
    return [result]
