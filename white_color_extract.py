import cv2
import numpy as np
from typing import Dict
import os
import glob

from color_lens_correction import build_radius_maps, apply_color_lens_correction
from undistort import undistort_image
from utils import ehance_contrast_gamma

def ehance_contrast_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    # Q: why inv_gamma?
    # A: because the formula is output = input^(1/gamma)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) *
                     255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    enhanced_image = cv2.LUT(image, table)
    return enhanced_image

def color_extract(image: np.ndarray, l_range: tuple = (0., 1.),
                  a_range: tuple = (0., 1.), b_range: tuple = (0., 1.),
                  display: bool = True) -> np.ndarray:
    # convert to float32
    image_float = image.astype(np.float32)/255.0
    # convert to lab
    image_lab = cv2.cvtColor(image_float, cv2.COLOR_BGR2Lab)
    # normalize lab channels to [0,1]
    image_lab[:, :, 0] /= 100.0
    image_lab[:, :, 1] = (image_lab[:, :, 1] + 128.0) / 255.0
    image_lab[:, :, 2] = (image_lab[:, :, 2] + 128.0) / 255.0
    # create mask
    l_mask = (image_lab[:, :, 0] >= l_range[0]) & (image_lab[:, :, 0] <= l_range[1])
    a_mask = (image_lab[:, :, 1] >= a_range[0]) & (image_lab[:, :, 1] <= a_range[1])
    b_mask = (image_lab[:, :, 2] >= b_range[0]) & (image_lab[:, :, 2] <= b_range[1])
    mask = (l_mask & a_mask & b_mask).astype(np.uint8) * 255

    # draw mask
    if display:
        cv2.imshow("L Channel", (image_lab[:, :, 0] * 255).astype(np.uint8))
        cv2.imshow("A Channel", (image_lab[:, :, 1] * 255).astype(np.uint8))
        cv2.imshow("B Channel", (image_lab[:, :, 2] * 255).astype(np.uint8))
        cv2.imshow("Color Mask", mask)
        cv2.waitKey(0)
    return mask


class TrackbarState:
    def __init__(self):
        # internal representation, updated in callback
        # values stored as 0-200 (trackbar range)
        self.values = {
            "l_min": 0,
            "l_max": 200,
            "a_min": 0,
            "a_max": 200,
            "b_min": 0,
            "b_max": 200,
        }


def create_trackbar_window(state: TrackbarState, window_name: str = "Color Extract"):
    """
    Create 6 trackbars:
      each for min and max of L, A, B channels in Lab color space.
    The trackbars range from 0 to 200, mapped to [0.0, 1.0] in the state.
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    def make_cb(param_key: str):
        def cb(val):
            state.values[param_key] = val  # store raw trackbar value (0-200)
        return cb

    # (trackbar name, param key, initial value)
    for name, key in [
        ("L Min", "l_min"),
        ("L Max", "l_max"),
        ("A Min", "a_min"),
        ("A Max", "a_max"),
        ("B Min", "b_min"),
        ("B Max", "b_max"),
    ]:
        initial_val = state.values[key]
        cv2.createTrackbar(name, window_name, initial_val, 200, make_cb(key))

def get_params_from_state(state: TrackbarState) -> Dict[str, float]:
    """Convert trackbar values (0-200) to normalized (0.0-1.0) for color_extract."""
    return {
        "l_min": state.values["l_min"] / 200.0,
        "l_max": state.values["l_max"] / 200.0,
        "a_min": state.values["a_min"] / 200.0,
        "a_max": state.values["a_max"] / 200.0,
        "b_min": state.values["b_min"] / 200.0,
        "b_max": state.values["b_max"] / 200.0,
    }

def list_road_images(road_dir: str) -> list:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(road_dir, e)))
    files.sort()
    return files

def pre_process(img):
    """
    Color undistortion and lens correction preprocessing function.
    
    :param img: Description
    """
    h, w = img.shape[:2]
    r2, r4 = build_radius_maps(h, w)

    # Define lens correction parameters
    params = {
        "a": 0.88, "b": 0.88,
        "c": 0.0, "d": 0.0,
        "e": 0.0, "f": 0.0
    }

    # Apply color lens correction
    img_corrected = apply_color_lens_correction(img, params, r2, r4)
    
    # Undistort the image
    undistorted_img = undistort_image(img_corrected)

    # undistorted_img = ehance_contrast_gamma(undistorted_img, gamma=1.5)
    
    return undistorted_img

def main():
    road_dir = os.path.join(os.path.dirname(__file__), "road")
    images = list_road_images(road_dir)
    if not images:
        print("No images found in 'road' directory.")
        return

    idx = 0
    img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to load image: {images[idx]}")
        return
    
    # undistort image
    img = pre_process(img)

    state = TrackbarState()
    win_name = "White Color Extract"
    create_trackbar_window(state, win_name)
    print("Controls:")
    print("  n: next image")
    print("  p: previous image")
    print("  q or ESC: quit")   

    while True:
        params = get_params_from_state(state)
        color_mask = color_extract(
            img,
            l_range=(params["l_min"], params["l_max"]),
            a_range=(params["a_min"], params["a_max"]),
            b_range=(params["b_min"], params["b_max"]),
            display=False
        )

        # Show original and mask side-by-side
        combined = np.hstack((img, cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow(win_name, combined)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('n'):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            img = pre_process(img)
        elif key == ord('p'):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            img = pre_process(img)
        elif key == ord('q') or key == 27:  # 'q' or ESC to quit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
