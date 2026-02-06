# input a image of road, and detect the central line of the road
import cv2
import numpy as np

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

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