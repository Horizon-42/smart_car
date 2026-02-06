from pathlib import Path
from tqdm import tqdm

from ultralytics import YOLO

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def _collect_images(image_folder, max_images=None):
    path = Path(image_folder)
    if path.is_dir():
        images = [p for p in sorted(path.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    else:
        images = [path] if path.suffix.lower() in IMAGE_EXTS else []
    if max_images is not None and max_images > 0:
        images = images[:max_images]
    return [str(p) for p in images]


def pre_label_images(
    image_folder,
    model_name='yolo26n.pt',
    conf_threshold=0.25,
    save_conf=True,
    max_images=None,
    batch_size=16,
    imgsz=None,
    device=None,
    half=False,
    chunk_size=1000,
):
    # Load the YOLOv8 model
    model = YOLO(model_name)

    # Run inference on the images in the specified folder (optionally limited)
    image_paths = _collect_images(image_folder, max_images=max_images)
    if not image_paths:
        return
    predict_kwargs = {
        'conf': conf_threshold,
        'stream': True,
        'verbose': False,
    }
    if batch_size is not None:
        predict_kwargs['batch'] = batch_size
    if imgsz is not None:
        predict_kwargs['imgsz'] = imgsz
    if device is not None:
        predict_kwargs['device'] = device
    if half:
        predict_kwargs['half'] = True

    total_images = len(image_paths)
    progress = None
    if tqdm is not None:
        progress = tqdm(total=total_images, desc='pre-label')
    else:
        print(f'pre-label: {total_images} images')

    processed = 0
    chunk_size = chunk_size if chunk_size and chunk_size > 0 else total_images
    for start in range(0, total_images, chunk_size):
        chunk = image_paths[start : start + chunk_size]
        results = model(chunk, **predict_kwargs)

        for result in results:
            image_path = Path(result.path)
            label_path = image_path.with_suffix('.txt')
            if label_path.exists():
                label_path.unlink()
            if result.boxes is None or len(result.boxes) == 0:
                label_path.write_text('')
            else:
                result.save_txt(label_path, save_conf=save_conf)
            processed += 1
            if progress is not None:
                progress.update(1)
            elif processed % 25 == 0:
                print(f'  processed {processed}/{total_images}')

    if progress is not None:
        progress.close()

if __name__ == "__main__":
    image_folder = "object_detection/data/combined_dataset/images"
    pre_label_images(image_folder, max_images=None)
