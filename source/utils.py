from typing import Optional, Sequence

import cv2
import torch
import numpy as np
from numpy.typing import NDArray


def crop_image(image: NDArray, bbox: Sequence[int]) -> NDArray:
    return image[
        bbox[1] : bbox[1] + bbox[3],
        bbox[0] : bbox[0] + bbox[2]]

def bgr2hsv(image: NDArray, dst: Optional[NDArray] = None) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV, dst=dst)

def bgr2gray(image: NDArray, dst: Optional[NDArray] = None) -> NDArray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY, dst=dst)

def bgr2tensor(image: NDArray) -> torch.Tensor:
    tensor = torch.from_numpy(image) * (1/255)
    tensor = tensor.unsqueeze(0) \
        .permute(0, 3, 1, 2) \
        .to(memory_format=torch.channels_last)
    return tensor

def gray2tensor(image: NDArray) -> torch.Tensor:
    tensor = torch.from_numpy(image) * (1/255)
    return tensor.unsqueeze_(-1)

def tensors2batch(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    batch = torch.stack(tensors)
    batch = batch.permute(0, 3, 1, 2).to(memory_format=torch.channels_last)
    return batch

def get_centroid(contour: NDArray, round: bool = False) \
        -> tuple[float, float] | tuple[int, int]:
    m = cv2.moments(contour)
    inv_area = 1 / m['m00']
    x = m['m10'] * inv_area
    y = m['m01'] * inv_area
    
    if round:
        return int(x), int(y)
    else:
        return x, y

def draw_contour_mask(
        image: NDArray, contour: NDArray,
        offset: Optional[tuple[int, int]] = None) -> NDArray:
    return cv2.drawContours(image, [contour], -1, 255, -1, offset=offset)

def find_contours(
        mask: NDArray,
        mode: int = cv2.RETR_EXTERNAL) \
            -> Sequence[NDArray] | tuple[Sequence[NDArray], NDArray]:
    contours, hierarchy = cv2.findContours(
        mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    if mode != cv2.RETR_EXTERNAL:
        return contours, hierarchy
    else:
        return contours

def filter_contours_by_area(
        contours: Sequence[NDArray], threshold: float) -> list[NDArray]:
    return [c for c in contours if cv2.contourArea(c) >= threshold]

def fit_image_size(
        image: NDArray, max_width: int, max_height: int) -> NDArray:
    result = image

    input_height, input_width = result.shape[:2]
    if input_height > max_height or input_width > max_width:
        aspect = input_width / input_height

        new_width = min(input_width, max_width)
        new_height = min(input_height, max_height)

        if new_width / aspect > max_height:
            new_width = int(max_height * aspect)
        else:
            new_height = int(max_width / aspect)
        
        result = cv2.resize(
            result, (new_width, new_height), cv2.INTER_NEAREST)
        input_height, input_width = result.shape[:2]
    
    if input_height != max_height or input_width != max_width:
        pad_width = (
            ((max_height - input_height) // 2,
            (max_height - input_height + 1) // 2),
            ((max_width - input_width) // 2,
            (max_width - input_width + 1) // 2))
        result = np.pad(result, pad_width)

    return result
