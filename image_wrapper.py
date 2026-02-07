import cv2
import numpy as np
from undistort import undistort_image
import yaml
from undistort import undistort_image


def pick_points_from_image(image, window_name: str = "Image") -> list:
    """
    Display an image and allow the user to pick points by clicking on it.
    Left click to select points, right click to finish selection.

    Args:
        image_path (str): Path to the image file.
        window_name (str): Name of the display window.

    Returns:
        list: List of (x, y) tuples representing the selected points.
    """
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.waitKey(0)

    # only took last 4 points, and algin the y axis for 2 pairs
    points = points[-4:]
    
    # draw lines between points
    if len(points) == 4:
        colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]
        for i in range(4):
            cv2.circle(image, points[i], 5, colors[i], -1)
            cv2.line(image, points[i], points[(i+1)%4], colors[i], 2)
        cv2.imshow(window_name, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return points


def perspective_transform(image, src_points: list, dst_points: list, dst_image_size):
    """
    Apply perspective transformation to the image based on source and destination points.

    Args:
        image (cv2.Mat): Input image.
        src_points (list): List of 4 source points.
        dst_points (list): List of 4 destination points.

    Returns:
        cv2.Mat: Transformed image.
    """
    src_pts = np.array(src_points[:4], dtype=np.float32)
    dst_pts = np.array(dst_points[:4], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_image = cv2.warpPerspective(
        image, matrix, dsize=dst_image_size)
    return matrix, transformed_image

def select_points(image_path: str):

    selected_points = pick_points_from_image(image_path)
    print("Selected points:", selected_points)

    # save points to a yaml file
    with open("selected_points.yaml", "w") as f:
        yaml.dump({"selected_points": selected_points}, f)



if __name__ == "__main__":
    car_name = "my_car"
    image_path = "my_car/pic0.jpg"

    image = cv2.imread(image_path)
    image = undistort_image(image, car_name=car_name)

    selected_points = pick_points_from_image(image)

    # dst_image = (520,720)
    dst_image = (450, 200)
    pading = 0

    dst_points = [(pading, pading), (dst_image[0]-pading, pading), 
                  (dst_image[0]-pading, dst_image[1]-pading), (pading, dst_image[1]-pading)]
    print("Destination points:", dst_points)
    trans_matrix, transformed_image = perspective_transform(
        image, selected_points, dst_points, dst_image_size=dst_image)

    # save the transformation matrix to a npx file
    np.savez(f"{car_name}/perspective_transform.npz", matrix=trans_matrix, src_points=selected_points, dst_points=dst_points)

    # draw dst points on the transformed image
    for point in dst_points:
        cv2.circle(transformed_image, point, 5, (0, 0, 255), -1)

    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
