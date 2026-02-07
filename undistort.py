import cv2
import numpy as np

with open("my_car/camera_calibration.npz", "rb") as f:
    data = np.load(f)
    my_mtx = data['mtx']
    my_dist_coeffs = data['dist']

with np.load("nayans_car/camera_calibration.npz") as data:
    nayan_mtx = data['mtx']
    nayan_dist_coeffs = data['dist']


def undistort_image(img: np.ndarray, car_name:str ="my") -> np.ndarray:
    h, w = img.shape[:2]
    if car_name == "my":
        mtx = my_mtx
        dist_coeffs = my_dist_coeffs
    elif car_name == "nayan":
        mtx = nayan_mtx
        dist_coeffs = nayan_dist_coeffs
    else:
        raise ValueError(f"Unknown car name: {car_name}")
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        mtx, dist_coeffs, (w, h), 0, (w, h))

    undistorted_img = cv2.undistort(img, newcameramtx, dist_coeffs)
    x, y, w, h = roi
    dst = undistorted_img[y:y+h, x:x+w]
    return dst
