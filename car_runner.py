import numpy as np
from track_mask_direcly import HSVBoundWaveShare, HSVBoundRealTrack, get_track_mask, get_track_mask_directly
import cv2
import color_undistort


class CarRunner:
    def __init__(self, car_name: str, track_type: str, image_size: tuple = (640, 480)):
        self.car_name = car_name
        self.hsv_bound = HSVBoundRealTrack()
        if track_type == "wave_share":
            self.hsv_bound = HSVBoundWaveShare()
        self.image_size = image_size
        self.trak_type = track_type

        # initialize undistort parameters
        self.__init_undistort(car_name)
        # initialize color correction parameters
        self.__init_color_correction()
        # initialize perspective transform parameters
        self.__init_perspective_transform()

    def __init_undistort(self, car_name: str):
        with open(f"{car_name}/camera_calibration.npz", "rb") as f:
            data = np.load(f)
            self.mtx = data['mtx']
            self.dist_coeffs = data['dist']
        if self.mtx is None or self.dist_coeffs is None:
            raise ValueError(
                f"Camera calibration data not found for car: {car_name}")

        self.newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist_coeffs, self.image_size, 0, self.image_size)

    def __init_color_correction(self):
        r2, r4 = color_undistort.build_radius_maps(
            self.image_size[1], self.image_size[0])
        params = {
            "a": 0.88, "b": 0.88,
            "c": 0.0, "d": 0.0,
            "e": 0.0, "f": 0.0
        }
        # get per-channel gain maps
        self.gain_R, self.gain_G, self.gain_B = color_undistort.build_channel_gain_maps(params, r2, r4)

    def __init_perspective_transform(self):
        # load perspective transform parameters from file
        try:
            data = np.load(f"{self.car_name}/perspective_transform.npz")
            self.trans_matrix = data['matrix']
            self.src_points = data['src_points']
            self.dst_points = data['dst_points']
        except Exception as e:
            print(f"Failed to load perspective transform parameters: {e}")
            raise ValueError(f"Perspective transform parameters not found for car: {self.car_name}")
        self.bird_view_size = (self.dst_points[1][0] - self.dst_points[0][0], self.dst_points[2][1] - self.dst_points[0][1])

    def frame_preprocess(self, frame: np.ndarray) -> np.ndarray:
        # color correction
        if self.trak_type == "wave_share":
            corrected = frame
        else:
            corrected = color_undistort.apply_color_correction(frame, self.gain_R, self.gain_G, self.gain_B)
        # undistort image
        undistorted_img=cv2.undistort(
            corrected, self.newcameramtx, self.dist_coeffs)
        x, y, w, h=self.roi
        dst=undistorted_img[y:y+h, x:x+w]
        return dst
    
    def get_track_mask_wave_share(self, frame: np.ndarray):
        track_mask = get_track_mask_directly(frame, self.hsv_bound)
        # in every row, add compute center of all white pixels, set the mean position to white, and the rest to black
        for i in range(track_mask.shape[0]):
            row = track_mask[i]
            white_pixels = np.where(row > 0)[0]
            if len(white_pixels) > 0:
                mean_pos = int(np.mean(white_pixels))
                track_mask[i] = 0
                track_mask[i, mean_pos] = 255
        return track_mask
    
    def bird_eye_view(self, frame: np.ndarray):
        if self.trans_matrix is None:
            raise ValueError("Perspective transform matrix not initialized")
        transformed_image = cv2.warpPerspective(
            frame, self.trans_matrix, dsize=self.bird_view_size)
        return transformed_image

if __name__ == "__main__":
    import os
    # runner = CarRunner(car_name="nayans_car", track_type="real")
    runner = CarRunner(car_name="my_car", track_type="wave_share")
    # image_folder = "road"
    image_folder = "object_detection/data/combined_dataset_640X480/images"
    im_pathes = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(".jpg")]
    im_pathes.sort()
    for im_path in im_pathes:
        frame = cv2.imread(im_path)
        preprocessed = runner.frame_preprocess(frame)
        track_mask = get_track_mask_directly(preprocessed, runner.hsv_bound)
        bird_view = runner.bird_eye_view(preprocessed)


        if frame is None:
            print(f"Failed to read image: {im_path}")
            continue
        
        cv2.imshow("Original", frame)
        cv2.imshow("Preprocessed", preprocessed)
        cv2.imshow("Bird Eye View", bird_view)
        cv2.imshow("Track Mask", track_mask)
        key = cv2.waitKey(0)
        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break