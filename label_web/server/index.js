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

function resolveDatasetDir(datasetParam) {
  const trimmed =
    typeof datasetParam === "string" ? datasetParam.trim() : "";
  if (!trimmed) {
    return defaultDatasetDir;
  }
  if (path.isAbsolute(trimmed)) {
    return trimmed;
  }
  return path.resolve(repoRoot, trimmed);
}

function resolveDirs(dataDir) {
  const imagesDir = path.join(dataDir, "images");
  const labelsDir = path.join(dataDir, "labels");
  if (fs.existsSync(imagesDir)) {
    return { imagesDir, labelsDir };
  }
  return { imagesDir: dataDir, labelsDir: dataDir };
}

function getDatasetContext(req, res) {
  const datasetDir = resolveDatasetDir(req.query.dataset);
  try {
    const stats = fs.statSync(datasetDir);
    if (!stats.isDirectory()) {
      res.status(400).json({ error: `Dataset is not a directory: ${datasetDir}` });
      return null;
    }
  } catch (error) {
    res.status(400).json({ error: `Dataset not found: ${datasetDir}` });
    return null;
  }

  const { imagesDir, labelsDir } = resolveDirs(datasetDir);
  const outputDir = path.resolve(
    process.env.OUTPUT_DIR || path.join(datasetDir, "labels_edited")
  );

  return { datasetDir, imagesDir, labelsDir, outputDir };
}

async function listImages(imagesDir) {
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

async function listEditedImages(imageNames, outputDir) {
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

async function readClasses(datasetDir, fileNames) {
  const names = Array.isArray(fileNames) ? fileNames : [fileNames];
  for (const fileName of names) {
    const classesPath = path.join(datasetDir, fileName);
    try {
      const content = await fsp.readFile(classesPath, "utf-8");
      return content
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter(Boolean);
    } catch (error) {
      // Keep trying other filenames.
    }
  }
  return [];
}

function labelPathForImage(imageName, labelsDir) {
  const safeName = path.basename(imageName);
  const stem = path.parse(safeName).name;
  return path.join(labelsDir, `${stem}.txt`);
}

function outputLabelPathForImage(imageName, outputDir) {
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

async function detectSaveConf(imageNames, labelsDir, outputDir) {
  for (const name of imageNames) {
    const outputPath = outputLabelPathForImage(name, outputDir);
    if (fs.existsSync(outputPath)) {
      const { hasConf } = await parseLabelFile(outputPath);
      if (hasConf) return true;
      continue;
    }
    const labelPath = labelPathForImage(name, labelsDir);
    if (!fs.existsSync(labelPath)) continue;
    const { hasConf } = await parseLabelFile(labelPath);
    if (hasConf) return true;
  }
  return false;
}

function imagePathForName(imageName, imagesDir) {
  const safeName = path.basename(imageName || "");
  if (!safeName) return null;
  return path.join(imagesDir, safeName);
}

app.get("/api/image/:imageName", async (req, res) => {
  const { imageName } = req.params;
  if (!imageName) {
    res.status(400).json({ error: "Missing image name" });
    return;
  }
  const context = getDatasetContext(req, res);
  if (!context) return;
  const imagePath = imagePathForName(imageName, context.imagesDir);
  if (!imagePath || !fs.existsSync(imagePath)) {
    res.status(404).json({ error: "Image not found" });
    return;
  }
  res.sendFile(imagePath);
});

app.get("/api/meta", async (req, res) => {
  const context = getDatasetContext(req, res);
  if (!context) return;
  const images = await listImages(context.imagesDir);
  const classes = await readClasses(context.datasetDir, "classes.txt");
  const customClasses = await readClasses(context.datasetDir, [
    "custom_classes.txt",
    "custom_class.txt"
  ]);
  const editedImages = await listEditedImages(images, context.outputDir);
  const saveConf = await detectSaveConf(images, context.labelsDir, context.outputDir);
  res.json({
    images: images.map((name) => ({ name })),
    classes,
    customClasses,
    editedImages,
    datasetDir: context.datasetDir,
    outputDir: context.outputDir,
    saveConf
  });
});

app.get("/api/labels/:imageName", async (req, res) => {
  const imageName = req.params.imageName;
  if (!imageName) {
    res.status(400).json({ error: "Missing image name" });
    return;
  }
  const context = getDatasetContext(req, res);
  if (!context) return;
  const outputPath = outputLabelPathForImage(imageName, context.outputDir);
  if (fs.existsSync(outputPath)) {
    const result = await parseLabelFile(outputPath);
    res.json({ boxes: result.boxes, source: "edited" });
    return;
  }
  const labelPath = labelPathForImage(imageName, context.labelsDir);
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
  const context = getDatasetContext(req, res);
  if (!context) return;
  if (path.resolve(context.outputDir) === path.resolve(context.labelsDir)) {
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

  await fsp.mkdir(context.outputDir, { recursive: true });
  const outPath = path.join(context.outputDir, `${stem}.txt`);
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
