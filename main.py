from image_wrapper import perspective_transform
from central_line_extract import fit_central_curve
import cv2
import numpy as np
import os
from undistort import undistort_image
from color_undistort import apply_color_lens_correction, build_radius_maps

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
    
    return undistorted_img



def process(img):
    # image wrap
    selected_points = [(193, 142), (305, 142), (363, 251), (143, 251)]
    dst_image_size = (520, 720)
    dst_points = [(20, 20), (500, 20), (500, 700), (20, 700)]
    wraped_img = perspective_transform(
        img, src_points=selected_points, dst_points=dst_points, dst_image_size=dst_image_size)
    
    

    cv2.imwrite("wraped_img.jpg", wraped_img)

    cv2.namedWindow("wraped_img", cv2.WINDOW_GUI_NORMAL)

    cv2.imshow("img", img)
    cv2.imshow("wraped_img", wraped_img)

    cv2.waitKey(0)

    # central line extract
    output_image, color_mask, grad_mask, combined_mask = fit_central_curve(
        wraped_img)

    cv2.imshow("Color Mask", color_mask)
    cv2.imshow("Gradient Mask", grad_mask)
    cv2.imshow("Combined Mask", combined_mask)
    cv2.imshow("Central Curve Fit", output_image)



if __name__ == "__main__":
    # Load the image

    # load all images in the folder "road"
    img_folder = "road"
    img_names = os.listdir(img_folder)
    # sort by name
    img_names.sort()
    for img_name in img_names:
        print(f"Processing image: {img_name}")
        img_path = os.path.join(img_folder, img_name)
        # cehck if the file is an image
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            ori_img = cv2.imread(img_path)
            undistorted_img = pre_process(ori_img)
            cv2.imshow("Undistorted Image", undistorted_img)
            process(undistorted_img)

            key = cv2.waitKey(0)
            cv2.destroyAllWindows()

            if key == ord('q') or key == 27:  # 'q' or ESC to quit
                break
