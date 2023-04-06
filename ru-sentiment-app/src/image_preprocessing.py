"""Module for image preprocessing."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from typing import List, Tuple, Union

    from transformers import ViTFeatureExtractor


def resize_image(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resize image."""
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image."""
    normalized_image = np.zeros_like(image)
    normalized_image = cv2.normalize(image, normalized_image, 0, 255, cv2.NORM_MINMAX)
    return normalized_image


def equalize_image(image: np.ndarray) -> np.ndarray:
    """Image equalization."""
    clache = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_img[:, :, 0] = clache.apply(ycrcb_img[:, :, 0])
    equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
    return equalized_img


def preprocess_images(
    images: Union[List[Image], Image],
    size: Tuple[int, int] = (256, 256),
) -> Union[List[Image], Image]:
    """Preprocess image."""
    images_ = images
    if type(images_) != list:
        images_ = list(images_)

    preprocessed_images = []
    for image in images_:
        image_array = np.array(image)
        resized_image = resize_image(image_array, size)
        normalized_image = normalize_image(resized_image)
        equalized_image = equalize_image(normalized_image)
        preprocessed_image = Image.fromarray(equalized_image.astype("uint8"), "RGB")
        preprocessed_images.append(preprocessed_image)

    result = preprocessed_images[0] if len(preprocessed_images) == 1 else preprocessed_images
    return result


def extract_features_from_images(
    images: Union[List[Image], Image],
    image_feature_extractor: ViTFeatureExtractor,
    use_image_preprocessing: bool = False,
    size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """Extract features from images."""
    images_ = images

    if use_image_preprocessing:
        images_ = preprocess_images(images_, size)

    if type(images_) != list:
        images_ = [images_]

    pixel_values = []
    for i in range(len(images_)):
        image = images_[i]
        features = image_feature_extractor(image, return_tensors="pt")
        pixel_value = features.get("pixel_values")
        pixel_values.append(pixel_value)

    return torch.cat(pixel_values, dim=0)
