from ultralytics import YOLO

def pre_label_images(image_folder, model_name='yolo26n.pt', conf_threshold=0.25):
    # Load the YOLOv8 model
    model = YOLO(model_name)

    # Run inference on the images in the specified folder
    results = model(image_folder, conf=conf_threshold)

    # Save the results (labels and bounding boxes) in YOLO format
    for result in results:
        result.save(save_dir=image_folder)  # This will save labels in the same folder as images

if __name__ == "__main__":
    image_folder = "object_detection/data/combined_dataset/images"
    pre_label_images(image_folder)