import cv2
import numpy as np
from undistort import undistort_image
import yaml


def pick_points_from_image(image_path: str, window_name: str = "Image") -> list:
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
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow(window_name, image)

    image = cv2.imread(image_path)
    
    # undistort the image
    # image = undistort_image(image)

    cv2.imshow(window_name, image)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.waitKey(0)

    # only took last 4 points, and algin the y axis for 2 pairs
    points = points[-4:]
    points[0] = (points[0][0], (points[0][1] + points[1][1]) // 2)
    points[1] = (points[1][0], (points[0][1]))
    points[2] = (points[2][0], (points[2][1] + points[3][1]) // 2)
    points[3] = (points[3][0], (points[2][1]))
    
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


def perspective_transform(image: cv2.Mat, src_points: list, dst_points: list, dst_image_size: tuple[int, int]) -> cv2.Mat:
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
    return transformed_image

def select_points(image_path: str):

    selected_points = pick_points_from_image(image_path)
    print("Selected points:", selected_points)

    # save points to a yaml file
    with open("selected_points.yaml", "w") as f:
        yaml.dump({"selected_points": selected_points}, f)



if __name__ == "__main__":
    image_path = "road/road_000.jpg"  # Replace with your image path
    image_path = "road_virtual/raw_0000020.png"
    # select_points(image_path)
    # load points from yaml file
    with open("selected_points.yaml", "r") as f:
        data = yaml.unsafe_load(f)
    selected_points = data["selected_points"]

    image = cv2.imread(image_path)
    # image = undistort_image(image)

    # dst_image = (520,720)
    dst_image = (320, 240)
    pading = 10

    dst_points = [(pading, pading), (dst_image[0]-pading, pading), 
                  (dst_image[0]-pading, dst_image[1]-pading), (pading, dst_image[1]-pading)]
    transformed_image = perspective_transform(
        image, selected_points, dst_points, dst_image_size=dst_image)

    cv2.imwrite("test_data/road_transformed.jpg", transformed_image)

    # draw dst points on the transformed image
    for point in dst_points:
        cv2.circle(transformed_image, point, 5, (0, 0, 255), -1)

    cv2.imshow("Transformed Image", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
