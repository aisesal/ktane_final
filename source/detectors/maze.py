import cv2
import numpy as np
from numpy.typing import NDArray
from utils import find_contours, get_centroid

_CR_EMPTY_CELL = ((0, 187, 79), (179, 187, 97))
_CR_WHITE_CELL = ((97, 0, 215), (179, 12, 255))
_CR_RED_CELL = ((179, 0, 0), (179, 255, 255))
_CR_GREEN_MARKER = ((36, 128, 91), (62, 180, 255))

def detect_maze(hsv_image: NDArray) -> tuple[int, int, tuple[int, int]]:
    mask = cv2.inRange(hsv_image, *_CR_EMPTY_CELL)
    contours = find_contours(mask)
    keypoints = [('e', get_centroid(c, round=True)) for c in contours]

    cv2.inRange(hsv_image, *_CR_WHITE_CELL, dst=mask)
    contours = find_contours(mask)
    keypoints += [('s', get_centroid(contours[0], round=True))]

    cv2.inRange(hsv_image, *_CR_RED_CELL, dst=mask)
    contours = find_contours(mask)
    keypoints += [('f', get_centroid(contours[0], round=True))]

    keypoints.sort(key=lambda x: _hash_position(x[1], mask.shape[1], 16))
    start_pos, finish_pos = -1, -1
    for index, keypoint in enumerate(keypoints):
        if keypoint[0] == 's':
            start_pos = index
        elif keypoint[0] == 'f':
            finish_pos = index
    keypoints = np.expand_dims(np.array([x[1] for x in keypoints]), 1)

    cv2.inRange(hsv_image, *_CR_GREEN_MARKER, dst=mask)
    contours = find_contours(mask)
    markers = np.array([get_centroid(c, round=True) for c in contours])

    delta = np.linalg.norm(keypoints-markers, axis=-1)
    delta = np.argmin(delta, axis=0)

    return start_pos, finish_pos, tuple(sorted(delta.tolist()))

def _hash_position(pos: tuple[int, int], width: int, stride: int) -> int:
    y = pos[1] & ~(stride-1)
    x = pos[0]
    return y * width + x
