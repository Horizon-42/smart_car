import cv2
import numpy as np

def ehance_contrast_gamma(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    enhanced_image = cv2.LUT(image, table)
    return enhanced_image

def linear_contrast_stretch(image: np.ndarray, low_perc: float = 2.0, high_perc: float = 98.0) -> np.ndarray:
    # compute the low and high percentile values
    low_val = np.percentile(image, low_perc)
    high_val = np.percentile(image, high_perc)

    # stretch the contrast
    stretched = np.clip((image - low_val) * (255.0 / (high_val - low_val)), 0, 255).astype(np.uint8)
    return stretched

def linear_contrast_enhance(gray: np.ndarray, pivot: int = 128, scale: float = 1.5) -> np.ndarray:
    # apply linear contrast enhancement
    enhanced = np.clip((gray - pivot) * scale + pivot, 0, 255).astype(np.uint8)
    return enhanced