import cv2
import numpy as np
import os
from dataclasses import dataclass

@dataclass
class HSVBound:
    track_low: tuple = (0, 100, 120)
    track_high: tuple = (180, 255, 255)

    center_low: tuple = (0, 0, 200)
    center_high: tuple = (180, 30, 255)

def get_mask_hsv(frame:np.ndarray, hsv_low_bound=(0, 100, 120), hsv_high_bound=(180, 255, 255))->np.ndarray:
    # convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # create a binary mask where the specified HSV range is white and the rest is black
    mask = cv2.inRange(hsv_frame, hsv_low_bound, hsv_high_bound)
    return mask


if __name__ == "__main__":
    white = (255, 255, 255)
    white_mat = np.full((100, 100, 3), white, dtype=np.uint8)
    white_hsv = cv2.cvtColor(white_mat, cv2.COLOR_BGR2HSV)[0,0]
    print("White in HSV:", white_hsv)

    gray = (100, 100, 100)
    gray_mat = np.full((100, 100, 3), gray, dtype=np.uint8)
    gray_hsv = cv2.cvtColor(gray_mat, cv2.COLOR_BGR2HSV)[0,0]
    print("Gray in HSV:", gray_hsv)
    # exit(0)

    im_pathes = [os.path.join("road", f) for f in os.listdir("road") if f.endswith(".jpg")]
    for im_path in im_pathes:
        frame = cv2.imread(im_path)
        track_mask = get_mask_hsv(frame, HSVBound.track_low, HSVBound.track_high)
        center_mask = get_mask_hsv(frame, HSVBound.center_low, HSVBound.center_high)
        # stack original frame and mask side by side for comparison
        combined = np.hstack((frame, cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(center_mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Original Frame and Mask", combined)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break

    
    cv2.destroyAllWindows()