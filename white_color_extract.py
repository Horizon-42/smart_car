import cv2
import numpy as np
from typing import Dict
import json
import os
import glob

from color_lens_correction import build_radius_maps, apply_color_lens_correction
from undistort import undistort_image

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

MODE_SPECS = {
    "lab": {
        "labels": ("L", "A", "B"),
        "max": (255, 255, 255),
        "cvt": cv2.COLOR_BGR2Lab,
    },
    "hsv": {
        "labels": ("H", "S", "V"),
        "max": (179, 255, 255),
        "cvt": cv2.COLOR_BGR2HSV,
    },
}

def color_extract(image: np.ndarray,
                  mode: str,
                  c1_range: tuple,
                  c2_range: tuple,
                  c3_range: tuple,
                  display: bool = True) -> np.ndarray:
    spec = MODE_SPECS[mode]
    converted = cv2.cvtColor(image, spec["cvt"])
    c1, c2, c3 = cv2.split(converted)

    c1_mask = (c1 >= c1_range[0]) & (c1 <= c1_range[1])
    c2_mask = (c2 >= c2_range[0]) & (c2 <= c2_range[1])
    c3_mask = (c3 >= c3_range[0]) & (c3 <= c3_range[1])
    mask = (c1_mask & c2_mask & c3_mask).astype(np.uint8) * 255

    if display:
        cv2.imshow(f"{spec['labels'][0]} Channel", c1)
        cv2.imshow(f"{spec['labels'][1]} Channel", c2)
        cv2.imshow(f"{spec['labels'][2]} Channel", c3)
        cv2.imshow("Color Mask", mask)
        cv2.waitKey(0)
    return mask


class TrackbarState:
    def __init__(self, mode: str = "lab"):
        self.mode = mode
        self.values = {
            "lab": {
                "c1_min": 0, "c1_max": 255,
                "c2_min": 0, "c2_max": 255,
                "c3_min": 0, "c3_max": 255,
            },
            "hsv": {
                "c1_min": 0, "c1_max": 179,
                "c2_min": 0, "c2_max": 255,
                "c3_min": 0, "c3_max": 255,
            },
        }


def create_trackbar_window(state: TrackbarState, mode: str, window_name: str):
    """
    Create 6 trackbars for the active color space.
    """
    spec = MODE_SPECS[mode]
    labels = spec["labels"]
    max_vals = spec["max"]
    current = state.values[mode]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    def make_cb(param_key: str, mode_key: str):
        def cb(val):
            state.values[mode_key][param_key] = val
        return cb

    for idx, label in enumerate(labels, start=1):
        min_key = f"c{idx}_min"
        max_key = f"c{idx}_max"
        max_val = max_vals[idx - 1]
        cv2.createTrackbar(
            f"{label} Min",
            window_name,
            current[min_key],
            max_val,
            make_cb(min_key, mode),
        )
        cv2.createTrackbar(
            f"{label} Max",
            window_name,
            current[max_key],
            max_val,
            make_cb(max_key, mode),
        )

def get_params_from_state(state: TrackbarState, mode: str) -> Dict[str, int]:
    return dict(state.values[mode])

def build_json_payload(state: TrackbarState) -> Dict[str, Dict[str, int]]:
    def mode_payload(mode: str) -> Dict[str, int]:
        labels = MODE_SPECS[mode]["labels"]
        values = state.values[mode]
        return {
            f"{labels[0].lower()}_min": values["c1_min"],
            f"{labels[0].lower()}_max": values["c1_max"],
            f"{labels[1].lower()}_min": values["c2_min"],
            f"{labels[1].lower()}_max": values["c2_max"],
            f"{labels[2].lower()}_min": values["c3_min"],
            f"{labels[2].lower()}_max": values["c3_max"],
        }

    return {
        "active_mode": state.mode,
        "lab": mode_payload("lab"),
        "hsv": mode_payload("hsv"),
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

    state = TrackbarState(mode="lab")
    preview_name = "White Color Extract"
    controls_name = f"Controls ({state.mode.upper()})"
    create_trackbar_window(state, state.mode, controls_name)
    print("Controls:")
    print("  n: next image")
    print("  p: previous image")
    print("  m: switch mode (Lab/HSV)")
    print("  q or ESC: quit")   

    while True:
        params = get_params_from_state(state, state.mode)
        color_mask = color_extract(
            img,
            mode=state.mode,
            c1_range=(params["c1_min"], params["c1_max"]),
            c2_range=(params["c2_min"], params["c2_max"]),
            c3_range=(params["c3_min"], params["c3_max"]),
            display=False
        )

        # Show original and mask side-by-side
        combined = np.hstack((img, cv2.cvtColor(color_mask, cv2.COLOR_GRAY2BGR)))
        cv2.putText(
            combined,
            f"Mode: {state.mode.upper()}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(preview_name, combined)

        key = cv2.waitKey(100) & 0xFF
        if key == ord('n'):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            img = pre_process(img)
        elif key == ord('p'):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            img = pre_process(img)
        elif key == ord('m'):
            cv2.destroyWindow(controls_name)
            state.mode = "hsv" if state.mode == "lab" else "lab"
            controls_name = f"Controls ({state.mode.upper()})"
            create_trackbar_window(state, state.mode, controls_name)
        elif key == ord('q') or key == 27:  # 'q' or ESC to quit
            break

    cv2.destroyAllWindows()
    save_path = os.path.join(os.path.dirname(__file__), "white_color_extract_params.json")
    payload = build_json_payload(state)
    with open(save_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved params to {save_path}")

if __name__ == "__main__":
    main()
