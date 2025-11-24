from image_wrapper import perspective_transform
from central_line_extract import fit_central_curve
import cv2
import numpy as np

def process(img):
    # image wrap
    selected_points = [(194, 302), (341, 302), (408, 470), (132, 470)]
    dst_image_size = (520,720)
    dst_points = [(20, 20), (500, 20), (500, 700), (20, 700)]
    wraped_img = perspective_transform(img, src_points=selected_points, dst_points=dst_points, dst_image_size=dst_image_size)

    cv2.imwrite("wraped_img.jpg", wraped_img)

    cv2.namedWindow("wraped_img", cv2.WINDOW_GUI_NORMAL)
    cv2.imshow("wraped_img", wraped_img)

    # central line extract
    output_image, color_mask, grad_mask, combined_mask = fit_central_curve(wraped_img)

    cv2.imshow("Color Mask", color_mask)
    cv2.imshow("Gradient Mask", grad_mask)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Central Curve Fit", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Load the image
    img = cv2.imread("road/road_000.jpg")
    if img is None:
        print("Error: Could not load image.")
    else:
        process(img)
