# YOLO Label Web

A simple React-based labeling UI that loads YOLO data from
`object_detection/data/signs_labeled` and saves edits to a separate folder so
original labels are never overwritten.

## Run

1. Install deps

```
cd /home/supercomputing/studys/smart_car/label_web
npm install
```

2. Start the dataset server (port 5174)

```
npm run server
```

3. Start the UI (port 5173)

```
npm run dev
```

Open `http://localhost:5173`.

## Data paths

- Default dataset: `object_detection/data/signs_labeled`
- Images: `object_detection/data/signs_labeled/images`
- Original labels: `object_detection/data/signs_labeled/labels`
- Output labels: `object_detection/data/signs_labeled/labels_edited`

You can override paths:

```
DATASET_DIR=/path/to/data OUTPUT_DIR=/path/to/output npm run server
```

## Notes

- The UI supports YOLO labels with 5 or 6 columns. If any label contains a
  confidence column, new boxes are saved with a default confidence of 1.0.
- Saving never writes to the original `labels` folder.
