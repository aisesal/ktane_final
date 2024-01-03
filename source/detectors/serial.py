from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch
from models.symbols import Symbols as Model
from numpy.typing import NDArray
from utils import (bgr2gray, crop_image, draw_contour_mask, find_contours,
                   fit_image_size, gray2tensor, tensors2batch)

_MODEL = Model(34).eval()
_MODEL.load_state_dict(torch.load(Path('models/SerialSymbols.pt')))
_LABELS = '0123456789ABCDEFGHIJKLMNPQRSTUVWXZ'

_CR_BORDER = ((0, 0, 109), (36, 43, 255))
_THRESH_BORDER_AREA = 6000
_THRESH_BLOB_COLOR = 50
_THRESH_BLOB_AREA = 100
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def detect_serial(
        bgr_image: NDArray, hsv_image: NDArray) -> Optional[str]:
    images = _filter(bgr_image, _find(hsv_image))
    if not images:
        return None
    
    batch = [gray2tensor(fit_image_size(x, 64, 64)) for x in images[0]]
    batch = tensors2batch(batch)
    prediction = _MODEL(batch).argmax(dim=1).tolist()
    return ''.join(_LABELS[x] for x in prediction)

def _find(hsv_image: NDArray) -> Sequence[NDArray]:
    mask = cv2.inRange(hsv_image, *_CR_BORDER)
    cv2.morphologyEx(mask, cv2.MORPH_OPEN, _MORPH_KERNEL, dst=mask)
    return find_contours(mask)

def _filter(
        bgr_image: NDArray,
        contours: Sequence[NDArray]) -> list[list[NDArray]]:
    result = []
    for contour in contours:
        if cv2.contourArea(contour) < _THRESH_BORDER_AREA:
            continue

        bbox = cv2.boundingRect(contour)
        gray = bgr2gray(crop_image(bgr_image, bbox))
        _, blobs = cv2.threshold(
            gray, _THRESH_BLOB_COLOR, 255, cv2.THRESH_BINARY_INV)
        result += _filter_blobs(blobs)
    return result

def _filter_blobs(blobs: NDArray) -> list[list[NDArray]]:
    contours = [c for c in find_contours(blobs) if _blob_match(c)]
    if len(contours) != 6:
        return []
    
    contours.sort(key=lambda x: x[:,:,0].min())

    result = []
    for contour in contours:
        bbox = cv2.boundingRect(contour)
        symbol = crop_image(blobs, bbox)
        mask = draw_contour_mask(
            np.zeros_like(symbol), contour, offset=(-bbox[0], -bbox[1]))
        result.append(symbol & mask)
    return [result]

def _blob_match(contour: NDArray) -> bool:
    if cv2.contourArea(contour) < _THRESH_BLOB_AREA:
        return False
    bbox = cv2.boundingRect(contour)
    if bbox[2] > bbox[3]:
        return False
    return True
