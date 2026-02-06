import cv2
import numpy as np
import os
from dataclasses import dataclass
from color_undistort import color_correct 
from undistort import undistort_image
@dataclass
class HSVBoundWaveShare:
    track_low: tuple = (0, 100, 120)
    track_high: tuple = (180, 255, 255)

    center_low: tuple = (0, 0, 200)
    center_high: tuple = (180, 30, 255)

@dataclass
class HSVBoundRealTrack:
    track_low: tuple = (99, 0, 172)
    track_high: tuple = (164, 145, 255)

    center_low: tuple = (99, 0, 172)
    center_high: tuple = (164, 145, 255)


def get_mask_hsv(frame:np.ndarray, hsv_low_bound=(0, 100, 120), hsv_high_bound=(180, 255, 255))->np.ndarray:
    # convert frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # create a binary mask where the specified HSV range is white and the rest is black
    mask = cv2.inRange(hsv_frame, hsv_low_bound, hsv_high_bound)
    return mask


if __name__ == "__main__":
    # HSVBound = HSVBoundWaveShare()
    HSVBound = HSVBoundRealTrack()

    white = (255, 255, 255)
    white_mat = np.full((100, 100, 3), white, dtype=np.uint8)
    white_hsv = cv2.cvtColor(white_mat, cv2.COLOR_BGR2HSV)[0,0]
    print("White in HSV:", white_hsv)

    gray = (100, 100, 100)
    gray_mat = np.full((100, 100, 3), gray, dtype=np.uint8)
    gray_hsv = cv2.cvtColor(gray_mat, cv2.COLOR_BGR2HSV)[0,0]
    print("Gray in HSV:", gray_hsv)
    # exit(0)

    image_folder = "object_detection/data/combined_dataset_640X480/images"

    im_pathes = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    # sort by name
    im_pathes.sort()
    for im_path in im_pathes:
        frame = cv2.imread(im_path)
        frame = color_correct(frame)
        undistorted = undistort_image(frame)
        # resize to 640x480
        undistorted = cv2.resize(undistorted, (640, 480))

        if frame is None:
            print(f"Failed to read image: {im_path}")
            continue
        track_mask = get_mask_hsv(frame, HSVBound.track_low, HSVBound.track_high)
        center_mask = get_mask_hsv(frame, HSVBound.center_low, HSVBound.center_high)
        # stack original frame and mask side by side for comparison
        combined = np.hstack((frame, undistorted, cv2.cvtColor(track_mask, cv2.COLOR_GRAY2BGR), cv2.cvtColor(center_mask, cv2.COLOR_GRAY2BGR)))
        cv2.imshow("Original Frame and Mask", combined)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break

    
    cv2.destroyAllWindows()