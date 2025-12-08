import os
import glob
from typing import Tuple, Dict

import cv2
import numpy as np
from color_undistort import build_radius_maps, apply_color_lens_correction
from undistort import undistort_image

def list_road_images(road_dir: str) -> list:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(road_dir, e)))
    files.sort()
    return files


class TrackbarState:
    def __init__(self):
        # internal representation, updated in callback
        self.values = {
            "a": 0.0,
            "b": 0.0,
            "c": 0.0,
            "d": 0.0,
            "e": 0.0,
            "f": 0.0,
        }


def create_trackbar_window(state: TrackbarState, window_name: str = "Color Lens Correction"):
    """
    Create 6 trackbars:
      each in range [-1.0, 1.0] with step 0.01 → underlying integer 0..200 mapped to val = (pos - 100) / 100
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    def make_cb(key: str):
        def _cb(pos: int):
            state.values[key] = (pos - 100) / 100.0

        return _cb

    # (trackbar name, param key)
    for name, key in [
        ("a_R", "a"),
        ("b_R", "b"),
        ("c_G", "c"),
        ("d_G", "d"),
        ("e_B", "e"),
        ("f_B", "f"),
    ]:
        cv2.createTrackbar(name, window_name, 100, 200, make_cb(key))


def get_params_from_state(state: TrackbarState) -> Dict[str, float]:
    return dict(state.values)


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
    # img = undistort_image(img)

    h, w = img.shape[:2]
    r2, r4 = build_radius_maps(h, w)

    state = TrackbarState()
    win_name = "Color Lens Correction"
    create_trackbar_window(state, win_name)

    print("Controls:")
    print("  n: next image")
    print("  p: previous image")
    print("  s: save current parameters to 'color_lens_params.txt'")
    print("  q or ESC: quit")

    while True:
        params = get_params_from_state(state)
        corrected = apply_color_lens_correction(img, params, r2, r4)

        # Show original and corrected side-by-side
        disp = np.hstack([img, corrected])
        info = f"{os.path.basename(images[idx])} | a={params['a']:.2f}, b={params['b']:.2f}, " \
               f"c={params['c']:.2f}, d={params['d']:.2f}, e={params['e']:.2f}, f={params['f']:.2f}"
        # Draw info text directly on the display image (works without Qt support)
        cv2.putText(
            disp,
            info,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(50) & 0xFF

        if key in (ord("q"), 27):  # 'q' or ESC
            break
        elif key == ord("n"):
            idx = (idx + 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {images[idx]}")
                continue
            h, w = img.shape[:2]
            r2, r4 = build_radius_maps(h, w)
        elif key == ord("p"):
            idx = (idx - 1) % len(images)
            img = cv2.imread(images[idx], cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {images[idx]}")
                continue
            h, w = img.shape[:2]
            r2, r4 = build_radius_maps(h, w)
        elif key == ord("s"):
            params = get_params_from_state(state)
            out_path = os.path.join(os.path.dirname(__file__), "color_lens_params.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                for k in ["a", "b", "c", "d", "e", "f"]:
                    f.write(f"{k}: {params[k]:.6f}\n")
            print(f"Saved parameters to {out_path}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
