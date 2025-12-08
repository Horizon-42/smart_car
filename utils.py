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