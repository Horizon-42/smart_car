const express = require("express");
const path = require("path");
const fs = require("fs");
const fsp = require("fs/promises");

const IMAGE_EXTS = new Set([".jpg", ".jpeg", ".png", ".bmp"]);

const app = express();
app.use(express.json({ limit: "5mb" }));

const repoRoot = path.resolve(__dirname, "..", "..");
const defaultDatasetDir = path.join(
  repoRoot,
  "object_detection",
  "data",
  "signs_labeled"
);

const datasetDir = path.resolve(process.env.DATASET_DIR || defaultDatasetDir);

function resolveDirs(dataDir) {
  const imagesDir = path.join(dataDir, "images");
  const labelsDir = path.join(dataDir, "labels");
  if (fs.existsSync(imagesDir)) {
    return { imagesDir, labelsDir };
  }
  return { imagesDir: dataDir, labelsDir: dataDir };
}

const { imagesDir, labelsDir } = resolveDirs(datasetDir);
const outputDir = path.resolve(
  process.env.OUTPUT_DIR || path.join(datasetDir, "labels_edited")
);

async function listImages() {
  try {
    const entries = await fsp.readdir(imagesDir, { withFileTypes: true });
    return entries
      .filter((entry) => entry.isFile())
      .map((entry) => entry.name)
      .filter((name) => IMAGE_EXTS.has(path.extname(name).toLowerCase()))
      .sort();
  } catch (error) {
    return [];
  }
}

async function listEditedImages(imageNames) {
  try {
    const entries = await fsp.readdir(outputDir, { withFileTypes: true });
    const editedStems = new Set(
      entries
        .filter((entry) => entry.isFile())
        .map((entry) => path.parse(entry.name).name)
    );
    return imageNames.filter((name) => editedStems.has(path.parse(name).name));
  } catch (error) {
    return [];
  }
}

async function readClasses(fileName) {
  const classesPath = path.join(datasetDir, fileName);
  try {
    const content = await fsp.readFile(classesPath, "utf-8");
    return content
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
  } catch (error) {
    return [];
  }
}

function labelPathForImage(imageName) {
  const safeName = path.basename(imageName);
  const stem = path.parse(safeName).name;
  return path.join(labelsDir, `${stem}.txt`);
}

function outputLabelPathForImage(imageName) {
  const safeName = path.basename(imageName);
  const stem = path.parse(safeName).name;
  return path.join(outputDir, `${stem}.txt`);
}

async function parseLabelFile(labelPath) {
  try {
    const content = await fsp.readFile(labelPath, "utf-8");
    const lines = content.split(/\r?\n/);
    const boxes = [];
    let hasConf = false;
    for (const line of lines) {
      if (!line.trim()) continue;
      const parts = line.trim().split(/\s+/);
      if (parts.length < 5) continue;
      const classId = Number(parts[0]);
      const xCenter = Number(parts[1]);
      const yCenter = Number(parts[2]);
      const width = Number(parts[3]);
      const height = Number(parts[4]);
      if (
        Number.isNaN(classId) ||
        Number.isNaN(xCenter) ||
        Number.isNaN(yCenter) ||
        Number.isNaN(width) ||
        Number.isNaN(height)
      ) {
        continue;
      }
      let conf = null;
      if (parts.length >= 6) {
        const parsedConf = Number(parts[5]);
        if (!Number.isNaN(parsedConf)) {
          conf = parsedConf;
          hasConf = true;
        }
      }
      const x1 = Math.max(0, xCenter - width / 2);
      const y1 = Math.max(0, yCenter - height / 2);
      const x2 = Math.min(1, xCenter + width / 2);
      const y2 = Math.min(1, yCenter + height / 2);
      boxes.push({ classId, x1, y1, x2, y2, conf });
    }
    return { boxes, hasConf };
  } catch (error) {
    return { boxes: [], hasConf: false };
  }
}

async function detectSaveConf(imageNames) {
  for (const name of imageNames) {
    const outputPath = outputLabelPathForImage(name);
    if (fs.existsSync(outputPath)) {
      const { hasConf } = await parseLabelFile(outputPath);
      if (hasConf) return true;
      continue;
    }
    const labelPath = labelPathForImage(name);
    if (!fs.existsSync(labelPath)) continue;
    const { hasConf } = await parseLabelFile(labelPath);
    if (hasConf) return true;
  }
  return false;
}

app.use("/data/images", express.static(imagesDir));

app.get("/api/meta", async (_req, res) => {
  const images = await listImages();
  const classes = await readClasses("classes.txt");
  const customClasses = await readClasses("custom_class.txt");
  const editedImages = await listEditedImages(images);
  const saveConf = await detectSaveConf(images);
  res.json({
    images: images.map((name) => ({ name })),
    classes,
    customClasses,
    editedImages,
    datasetDir,
    outputDir,
    saveConf
  });
});

app.get("/api/labels/:imageName", async (req, res) => {
  const imageName = req.params.imageName;
  if (!imageName) {
    res.status(400).json({ error: "Missing image name" });
    return;
  }
  const outputPath = outputLabelPathForImage(imageName);
  if (fs.existsSync(outputPath)) {
    const result = await parseLabelFile(outputPath);
    res.json({ boxes: result.boxes, source: "edited" });
    return;
  }
  const labelPath = labelPathForImage(imageName);
  const result = await parseLabelFile(labelPath);
  res.json({ boxes: result.boxes, source: "original" });
});

app.post("/api/labels/:imageName", async (req, res) => {
  const imageName = req.params.imageName;
  const safeName = path.basename(imageName || "");
  if (!safeName) {
    res.status(400).json({ error: "Missing image name" });
    return;
  }
  if (path.resolve(outputDir) === path.resolve(labelsDir)) {
    res.status(400).json({ error: "Output directory cannot be the original labels directory." });
    return;
  }
  const stem = path.parse(safeName).name;
  const boxes = Array.isArray(req.body.boxes) ? req.body.boxes : [];
  const saveConf = Boolean(req.body.saveConf);

  const lines = boxes.map((box) => {
    const classId = Number(box.classId);
    const x1 = Math.max(0, Math.min(1, Number(box.x1)));
    const y1 = Math.max(0, Math.min(1, Number(box.y1)));
    const x2 = Math.max(0, Math.min(1, Number(box.x2)));
    const y2 = Math.max(0, Math.min(1, Number(box.y2)));
    const xCenter = (x1 + x2) / 2;
    const yCenter = (y1 + y2) / 2;
    const width = Math.max(0, x2 - x1);
    const height = Math.max(0, y2 - y1);
    if (saveConf) {
      const confValue =
        typeof box.conf === "number" && !Number.isNaN(box.conf)
          ? box.conf
          : 1.0;
      return `${classId} ${xCenter.toFixed(6)} ${yCenter.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)} ${confValue.toFixed(4)}`;
    }
    return `${classId} ${xCenter.toFixed(6)} ${yCenter.toFixed(6)} ${width.toFixed(6)} ${height.toFixed(6)}`;
  });

  await fsp.mkdir(outputDir, { recursive: true });
  const outPath = path.join(outputDir, `${stem}.txt`);
  await fsp.writeFile(outPath, lines.join("\n"));
  res.json({ ok: true, path: outPath });
});

const distDir = path.join(__dirname, "..", "dist");
if (fs.existsSync(distDir)) {
  app.use(express.static(distDir));
}

const port = Number(process.env.PORT || 5174);
app.listen(port, () => {
  // eslint-disable-next-line no-console
  console.log(`Label server running on http://localhost:${port}`);
});
