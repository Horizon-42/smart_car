# input a image of road, and detect the central line of the road
import cv2
import numpy as np

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    # convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # merge channels and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def enhace_contrast_equalized(image: np.ndarray) -> np.ndarray:
    # q: why only equalize Y channel works better?
    # because Y channel represents luminance, equalizing it enhances contrast without distorting colors.

    # convert to YUV color space
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    # apply histogram equalization to Y channel
    y_eq = cv2.equalizeHist(y)

    # merge channels and convert back to BGR
    yuv_eq = cv2.merge((y_eq, u, v))
    enhanced_image = cv2.cvtColor(yuv_eq, cv2.COLOR_YUV2BGR)
    return enhanced_image

def ehance_contrast_gamma(image: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    enhanced_image = cv2.LUT(image, table)
    return enhanced_image

def color_extract(image: np.ndarray, target_bgr_color: tuple = (255, 255, 255),
                  distance_trheshold: float = 0.1, display: bool = True) -> np.ndarray:
    # convert to float32
    image_float = image.astype(np.float32)/255.0
    target_color_float = np.array(
        [[target_bgr_color]], dtype=np.float32)/255.0
    # convert to hsv
    image_hsv = cv2.cvtColor(image_float, cv2.COLOR_BGR2HSV)
    target_color_hsv = cv2.cvtColor(
        target_color_float, cv2.COLOR_BGR2HSV)[0][0]
    target_color_hsv[0] /= 360.0

    # draw h s v
    h, s, v = cv2.split(image_hsv)
    h /= 360.0
    if display:
        cv2.imshow("H Channel", (h * 255).astype(np.uint8))
        cv2.imshow("S Channel", (s * 255).astype(np.uint8))
        cv2.imshow("V Channel", (v * 255).astype(np.uint8))
        cv2.waitKey(0)

    image_hsv[:, :, 0] /= 360.0

    # l1 norm distance in hsv space
    dh = np.abs(image_hsv[:, :, 0] - target_color_hsv[0])
    dh = np.minimum(dh, 1 - dh)  # circular distance
    ds = np.abs(image_hsv[:, :, 1] - target_color_hsv[1])
    dv = np.abs(image_hsv[:, :, 2] - target_color_hsv[2])

    # distance = (dh + ds + dv) / 3.0
    # take the max distance
    distance = np.minimum(np.minimum(dh, ds), dv)

    # draw distance map
    if display:
        distance_map = (distance * 255).astype(np.uint8)
        cv2.imshow("Distance Map", distance_map)
        cv2.waitKey(0)
    
    # print min max mean distance
    print(f"Distance - min: {np.min(distance)}, max: {np.max(distance)}, mean: {np.mean(distance)}")

    # create mask
    mask = (distance < distance_trheshold).astype(np.uint8) * 255
    return mask


def gradiant_extract_sobel(image: np.ndarray,
                           sobel_ksize: int = 5,
                           grad_thres_min: int = 20,
                           grad_thres_max: int = 100) -> np.ndarray:
    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # sobel operator
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)

    # gradient magnitude
    grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_magnitude = np.uint8(np.clip(grad_magnitude, 0, 255))

    # threshold
    mask = cv2.inRange(grad_magnitude, grad_thres_min, grad_thres_max)
    return mask


def central_curve_fit(mask: np.ndarray, polynomial_order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    # itearation over each row to find the central point
    height, width = mask.shape
    central_points = []
    # start from middle of the image
    for y in range(height//2, height):
        x_indices = np.where(mask[y, :] > 0)[0]
        if len(x_indices) > 0:
            central_x = int(np.mean(x_indices))
            central_points.append((central_x, y))
    central_points = np.array(central_points)

    # polynomial fit
    if len(central_points) < polynomial_order + 1:
        raise ValueError("Not enough points to fit the polynomial.")
    fit_coefficients = np.polyfit(
        central_points[:, 1], central_points[:, 0], polynomial_order)
    return fit_coefficients, central_points


def central_curve_fit_ransac(mask: np.ndarray, polynomial_order: int = 2, min_points_ratio: float = 0.3,
                             window_height: int = 20,
                             window_margin: int = 50) -> tuple[np.ndarray, np.ndarray]:
    # itearation over each row to find the central point
    height, width = mask.shape
    central_points = []

    for y in range(height//2, height, 1):
        x_indices = np.where(mask[y, :] > 0)[0]
        if len(x_indices) > 0:
            central_x = int(np.mean(x_indices))
            central_points.append((central_x, y))
    central_points = np.array(central_points)

    # polynomial fit with RANSAC
    if len(central_points) < polynomial_order + 1:
        raise ValueError("Not enough points to fit the polynomial.")

    X = central_points[:, 1].reshape(-1, 1)
    y = central_points[:, 0]

    model = make_pipeline(
        # Transforms [y] → [y, y², ...]
        PolynomialFeatures(polynomial_order, include_bias=False),
        # Fits: x = c₀ + c₁·y + c₂·y² + ...
        RANSACRegressor(min_samples=max(polynomial_order + 1, int(len(central_points) * min_points_ratio)),
                        residual_threshold=10.0,
                        random_state=42))
    model.fit(X, y)

    # extract coefficients
    ransac = model.named_steps['ransacregressor']
    coef = ransac.estimator_.coef_
    intercept = ransac.estimator_.intercept_

    fit_coefficients = np.concatenate([coef[::-1], [intercept]])

    return fit_coefficients, central_points

def fit_central_curve(image:np.ndarray):
    # enhance contrast
    # enhanced_image = enhance_contrast(image, clip_limit=4.0, tile_grid_size=(16, 16))
    enhanced_image = enhace_contrast_equalized(image)
    # enhanced_image = ehance_contrast_gamma(image, gamma=0.2)

    color_mask = color_extract(enhanced_image, target_bgr_color=(255, 255, 255),
                               distance_trheshold=0.04, display=True)

    # dilate to fill gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 3))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_DILATE, kernel)

    grad_mask = gradiant_extract_sobel(image, sobel_ksize=3,
                                       grad_thres_min=20, grad_thres_max=80)

    combined_mask = cv2.bitwise_and(color_mask, grad_mask)

    # morphx operation, erase noise
    # vertical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_ERODE, kernel)

    cv2.imshow("Color Mask", color_mask)
    cv2.imshow("Gradient Mask", grad_mask)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.waitKey(0)

    coefs, _ = central_curve_fit(
        combined_mask, polynomial_order=2)

    print("Polynomial Coefficients (without RANSAC):", coefs)

    fit_coefficients, central_points = central_curve_fit_ransac(
        combined_mask, polynomial_order=2)
    print("Polynomial Coefficients (with RANSAC):", fit_coefficients)

    # draw central points and fitted curve
    output_image = image.copy()
    for point in central_points:
        cv2.circle(output_image, (point[0], point[1]), 3, (0, 0, 255), -1)
    height = image.shape[0]
    for y in range(height):
        x = int(np.polyval(fit_coefficients, y))
        cv2.circle(output_image, (x, y), 2, (255, 0, 0), -1)
    return output_image, color_mask, grad_mask, combined_mask


if __name__ == "__main__":
    image_path = "wraped_img.jpg"  # Replace with your image path
    image = cv2.imread(image_path)

    output_image, color_mask, grad_mask, combined_mask = fit_central_curve(image)

    cv2.imshow("Color Mask", color_mask)
    cv2.imshow("Gradient Mask", grad_mask)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Central Curve Fit", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
