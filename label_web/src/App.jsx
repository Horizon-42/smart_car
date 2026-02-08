import React, { useEffect, useRef, useState } from "react";

const PALETTE = [
  "#ff5252",
  "#00bdd6",
  "#41b37b",
  "#ffc400",
  "#9268ff",
  "#ff7043",
  "#26c6da",
  "#9ccc65",
  "#ab47bc",
  "#42a5f5"
];

const emptyMeta = {
  images: [],
  classes: [],
  customClasses: [],
  editedImages: [],
  outputDir: "",
  datasetDir: "",
  saveConf: false
};

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function colorForClass(classId) {
  return PALETTE[classId % PALETTE.length];
}

function getInvalidClassInfo(boxes, classCount) {
  if (!Number.isFinite(classCount) || classCount <= 0) return null;
  let maxClassId = -Infinity;
  let invalidCount = 0;
  for (const box of boxes) {
    const classId = Number(box.classId);
    if (Number.isNaN(classId)) continue;
    if (classId < 0 || classId >= classCount) {
      invalidCount += 1;
      if (classId > maxClassId) {
        maxClassId = classId;
      }
    }
  }
  if (invalidCount === 0) return null;
  return {
    invalidCount,
    maxClassId,
    limit: classCount - 1
  };
}

function drawAlertBanner(ctx, text) {
  const paddingX = 10;
  const paddingY = 6;
  ctx.save();
  ctx.font = "bold 14px sans-serif";
  const maxBoxW = Math.max(0, ctx.canvas.width - 20);
  if (maxBoxW === 0) {
    ctx.restore();
    return;
  }
  let textToRender = text;
  const maxTextWidth = maxBoxW - paddingX * 2;
  if (maxTextWidth > 0) {
    const metrics = ctx.measureText(text);
    if (metrics.width > maxTextWidth) {
      const ellipsis = "...";
      let trimmed = text;
      while (
        trimmed.length > 0 &&
        ctx.measureText(trimmed + ellipsis).width > maxTextWidth
      ) {
        trimmed = trimmed.slice(0, -1);
      }
      textToRender = trimmed ? `${trimmed}${ellipsis}` : ellipsis;
    }
  }
  const metrics = ctx.measureText(textToRender);
  const textWidth = Math.ceil(metrics.width);
  const boxW = Math.min(maxBoxW, textWidth + paddingX * 2);
  const boxH = 26 + paddingY * 0.5;
  const x = 10;
  const y = 10;
  ctx.fillStyle = "rgba(220, 38, 38, 0.9)";
  ctx.fillRect(x, y, boxW, boxH);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.9)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x, y, boxW, boxH);
  ctx.fillStyle = "#ffffff";
  ctx.textBaseline = "top";
  ctx.fillText(textToRender, x + paddingX, y + paddingY);
  ctx.restore();
}

export default function App() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const dragRef = useRef(null);

  const [meta, setMeta] = useState(emptyMeta);
  const [index, setIndex] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [status, setStatus] = useState("Loading dataset...");
  const [image, setImage] = useState(null);
  const [imageSize, setImageSize] = useState({ w: 1, h: 1 });
  const [dragBox, setDragBox] = useState(null);
  const [currentClass, setCurrentClass] = useState(0);
  const [jumpValue, setJumpValue] = useState("");
  const [showEditedOnly, setShowEditedOnly] = useState(false);
  const [editedMap, setEditedMap] = useState({});
  const [labelSourceMap, setLabelSourceMap] = useState({});
  const [datasetInput, setDatasetInput] = useState("");
  const [activeDataset, setActiveDataset] = useState("");
  const [datasetAlert, setDatasetAlert] = useState({
    status: "idle",
    invalidImages: [],
    classCount: 0,
    totalImages: 0
  });
  const autoSaveTimer = useRef(null);
  const fullIndexRef = useRef(0);
  const fullImageRef = useRef("");
  const dragStartRef = useRef(null);
  const scanIdRef = useRef(0);

  const visibleImages = showEditedOnly
    ? meta.images.filter((item) => editedMap[item.name])
    : meta.images;
  const currentImage = visibleImages[index];
  const currentImageName = currentImage ? currentImage.name : "";

  function withDataset(url) {
    if (!activeDataset) return url;
    const joiner = url.includes("?") ? "&" : "?";
    return `${url}${joiner}dataset=${encodeURIComponent(activeDataset)}`;
  }

  async function fetchMeta(datasetOverride) {
    const trimmed =
      typeof datasetOverride === "string" ? datasetOverride.trim() : "";
    const query = trimmed ? `?dataset=${encodeURIComponent(trimmed)}` : "";
    const response = await fetch(`/api/meta${query}`);
    if (!response.ok) {
      let message = "Failed to load dataset metadata";
      try {
        const data = await response.json();
        if (data && data.error) {
          message = data.error;
        }
      } catch (error) {
        // Ignore JSON errors.
      }
      throw new Error(message);
    }
    return response.json();
  }

  async function fetchInvalidLabels(datasetOverride) {
    const trimmed =
      typeof datasetOverride === "string" ? datasetOverride.trim() : "";
    const query = trimmed ? `?dataset=${encodeURIComponent(trimmed)}` : "";
    const response = await fetch(`/api/invalid-labels${query}`);
    if (!response.ok) {
      throw new Error("Failed to scan dataset labels");
    }
    return response.json();
  }

  async function loadInvalidLabels(datasetOverride) {
    const scanId = scanIdRef.current + 1;
    scanIdRef.current = scanId;
    setDatasetAlert((prev) => ({
      ...prev,
      status: "loading"
    }));
    try {
      const data = await fetchInvalidLabels(datasetOverride);
      if (scanIdRef.current !== scanId) return;
      setDatasetAlert({
        status: "ready",
        invalidImages: Array.isArray(data.invalidImages) ? data.invalidImages : [],
        classCount: Number.isFinite(data.classCount) ? data.classCount : 0,
        totalImages: Number.isFinite(data.totalImages) ? data.totalImages : 0
      });
    } catch (error) {
      if (scanIdRef.current !== scanId) return;
      setDatasetAlert({
        status: "error",
        invalidImages: [],
        classCount: 0,
        totalImages: 0
      });
    }
  }

  function applyMeta(data, datasetOverride) {
    setMeta(data);
    setActiveDataset(data.datasetDir || "");
    setDatasetInput(data.datasetDir || datasetOverride || "");
    if (Array.isArray(data.editedImages)) {
      const nextEdited = {};
      data.editedImages.forEach((name) => {
        nextEdited[name] = true;
      });
      setEditedMap(nextEdited);
    } else {
      setEditedMap({});
    }
    setLabelSourceMap({});
    setShowEditedOnly(false);
    fullIndexRef.current = 0;
    fullImageRef.current = "";
    setIndex(0);
    setCurrentClass(0);
    setBoxes([]);
    setSelectedIdx(null);
    setDragBox(null);
    setStatus(data.images.length ? "" : "No images found.");
    loadInvalidLabels(data.datasetDir || datasetOverride || "");
  }

  useEffect(() => {
    let active = true;
    async function loadInitialMeta() {
      try {
        const data = await fetchMeta();
        if (!active) return;
        applyMeta(data, "");
      } catch (error) {
        if (active) {
          setStatus(error.message || "Failed to load dataset.");
        }
      }
    }
    loadInitialMeta();
    return () => {
      active = false;
    };
  }, []);

  async function loadDataset() {
    setStatus("Loading dataset...");
    try {
      const data = await fetchMeta(datasetInput);
      applyMeta(data, datasetInput.trim());
    } catch (error) {
      setStatus(error.message || "Failed to load dataset.");
    }
  }

  useEffect(() => {
    if (!currentImageName) {
      return;
    }
    let active = true;
    setStatus("Loading image...");
    const img = new Image();
    img.onload = () => {
      if (!active) return;
      setImage(img);
      setImageSize({ w: img.naturalWidth, h: img.naturalHeight });
      setStatus("");
    };
    img.onerror = () => {
      if (!active) return;
      setImage(null);
      setStatus("Failed to load image.");
    };
    img.src = withDataset(`/api/image/${encodeURIComponent(currentImageName)}`);

    async function loadLabels() {
      try {
        const response = await fetch(
          withDataset(`/api/labels/${encodeURIComponent(currentImageName)}`)
        );
        if (!response.ok) {
          throw new Error("Failed to load labels");
        }
        const data = await response.json();
        if (!active) return;
        const source = data.source === "edited" ? "edited" : "original";
        const nextBoxes = Array.isArray(data.boxes)
          ? data.boxes.map((box) => ({ ...box, edited: source === "edited" }))
          : [];
        setBoxes(nextBoxes);
        setLabelSourceMap((prev) => ({ ...prev, [currentImageName]: source }));
        if (nextBoxes.some((box) => typeof box.conf === "number")) {
          setMeta((prev) => ({ ...prev, saveConf: true }));
        }
        if (source === "edited") {
          setEditedMap((prev) => ({ ...prev, [currentImageName]: true }));
        }
        setSelectedIdx(null);
        setDragBox(null);
      } catch (error) {
        if (active) {
          setBoxes([]);
          setSelectedIdx(null);
          setDragBox(null);
        }
      }
    }

    loadLabels();
    return () => {
      active = false;
    };
  }, [currentImageName, activeDataset]);

  useEffect(() => {
    const handleResize = () => {
      resizeCanvas();
      drawCanvas();
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    resizeCanvas();
    drawCanvas();
  }, [image, boxes, selectedIdx, dragBox]);

  function queueSave(nextBoxes, imageName = currentImageName) {
    if (!imageName) return;
    setEditedMap((prev) => ({ ...prev, [imageName]: true }));
    setLabelSourceMap((prev) => ({ ...prev, [imageName]: "edited" }));
    if (autoSaveTimer.current) {
      clearTimeout(autoSaveTimer.current);
    }
    autoSaveTimer.current = setTimeout(() => {
      saveLabels({ silent: true, boxesOverride: nextBoxes, imageName });
    }, 400);
  }

  useEffect(() => {
    if (index >= visibleImages.length) {
      setIndex(Math.max(0, visibleImages.length - 1));
    }
    if (showEditedOnly && visibleImages.length === 0) {
      setStatus("No edited images.");
    }
    if (!showEditedOnly && status === "No edited images.") {
      setStatus("");
    }
    if (showEditedOnly && visibleImages.length > 0 && status === "No edited images.") {
      setStatus("");
    }
  }, [visibleImages.length, index, showEditedOnly, status]);

  useEffect(() => {
    const handler = (event) => {
      const tag = event.target.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA") {
        return;
      }
      if (event.key === "ArrowRight") {
        event.preventDefault();
        nextImage();
      } else if (event.key === "ArrowLeft") {
        event.preventDefault();
        prevImage();
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        cycleClass(-1);
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        cycleClass(1);
      } else if (event.key === "End") {
        event.preventDefault();
        clearAll();
      } else if (event.key === "s" || event.key === "S") {
        saveLabels();
      } else if (event.key === "Delete") {
        deleteSelected();
      }
    };
    window.addEventListener("keydown", handler, true);
    return () => window.removeEventListener("keydown", handler, true);
  }, [index, visibleImages.length, boxes, selectedIdx, currentClass]);

  function resizeCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    canvas.width = width;
    canvas.height = height;
  }

  function getTransform() {
    const canvas = canvasRef.current;
    if (!canvas || !image) {
      return { scale: 1, offsetX: 0, offsetY: 0, displayW: 0, displayH: 0 };
    }
    const canvasW = canvas.width;
    const canvasH = canvas.height;
    const scale = Math.min(canvasW / imageSize.w, canvasH / imageSize.h);
    const displayW = imageSize.w * scale;
    const displayH = imageSize.h * scale;
    const offsetX = (canvasW - displayW) / 2;
    const offsetY = (canvasH - displayH) / 2;
    return { scale, offsetX, offsetY, displayW, displayH };
  }

  function labelNameFor(box) {
    const classId = box.classId;
    const source = labelSourceMap[currentImageName];
    const useCustom =
      box.edited || source === "edited";
    if (useCustom && meta.customClasses.length && classId < meta.customClasses.length) {
      return meta.customClasses[classId];
    }
    if (meta.classes.length && classId < meta.classes.length) {
      return meta.classes[classId];
    }
    return `class_${classId}`;
  }

  function getBoxAreaStats(box) {
    const x1 = clamp01(Math.min(box.x1, box.x2));
    const x2 = clamp01(Math.max(box.x1, box.x2));
    const y1 = clamp01(Math.min(box.y1, box.y2));
    const y2 = clamp01(Math.max(box.y1, box.y2));
    const ratioArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const pixelArea = ratioArea * imageSize.w * imageSize.h;
    return { ratioArea, pixelArea };
  }

  function drawCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) {
      ctx.fillStyle = "#2b2b2b";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      return;
    }
    const { scale, offsetX, offsetY, displayW, displayH } = getTransform();
    ctx.imageSmoothingEnabled = true;
    ctx.fillStyle = "#1e1e1e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, offsetX, offsetY, displayW, displayH);

    boxes.forEach((box, idx) => {
      const x1 = offsetX + box.x1 * imageSize.w * scale;
      const y1 = offsetY + box.y1 * imageSize.h * scale;
      const x2 = offsetX + box.x2 * imageSize.w * scale;
      const y2 = offsetY + box.y2 * imageSize.h * scale;
      const color = colorForClass(box.classId);
      ctx.strokeStyle = color;
      ctx.lineWidth = idx === selectedIdx ? 3 : 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      const className = labelNameFor(box);
      const confText =
        meta.saveConf && typeof box.conf === "number"
          ? ` ${box.conf.toFixed(2)}`
          : "";
      ctx.fillStyle = color;
      ctx.font = "12px sans-serif";
      ctx.fillText(`${className}${confText}`, x1 + 4, y1 + 12);
    });

    const invalidInfo = getInvalidClassInfo(boxes, meta.classes.length);
    if (invalidInfo) {
      const datasetIndex =
        meta.images.findIndex((item) => item.name === currentImageName) + 1;
      const imageIndex = datasetIndex > 0 ? datasetIndex : index + 1;
      drawAlertBanner(
        ctx,
        `Image ${imageIndex}: label index exceeds classes.txt (max ${invalidInfo.maxClassId}, limit ${invalidInfo.limit})`
      );
    }

    if (dragBox) {
      const x1 = offsetX + dragBox.x1 * imageSize.w * scale;
      const y1 = offsetY + dragBox.y1 * imageSize.h * scale;
      const x2 = offsetX + dragBox.x2 * imageSize.w * scale;
      const y2 = offsetY + dragBox.y2 * imageSize.h * scale;
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    }
  }

  function canvasToNorm(event) {
    const canvas = canvasRef.current;
    if (!canvas || !image) return null;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const { scale, offsetX, offsetY } = getTransform();
    const imgX = (x - offsetX) / scale;
    const imgY = (y - offsetY) / scale;
    if (imgX < 0 || imgY < 0 || imgX > imageSize.w || imgY > imageSize.h) {
      return null;
    }
    return {
      x: imgX / imageSize.w,
      y: imgY / imageSize.h
    };
  }

  function onMouseDown(event) {
    const norm = canvasToNorm(event);
    if (!norm) return;
    dragStartRef.current = {
      norm,
      screenX: event.clientX,
      screenY: event.clientY
    };
    dragRef.current = norm;
    setDragBox(null);
  }

  function onMouseMove(event) {
    if (!dragRef.current || !dragStartRef.current) return;
    const norm = canvasToNorm(event);
    if (!norm) return;
    const start = dragStartRef.current.norm;
    const dx = event.clientX - dragStartRef.current.screenX;
    const dy = event.clientY - dragStartRef.current.screenY;
    if (Math.hypot(dx, dy) < 4 && !dragBox) {
      return;
    }
    const x1 = Math.min(start.x, norm.x);
    const y1 = Math.min(start.y, norm.y);
    const x2 = Math.max(start.x, norm.x);
    const y2 = Math.max(start.y, norm.y);
    setDragBox({ x1, y1, x2, y2 });
  }

  function onMouseUp(event) {
    if (!dragRef.current || !dragStartRef.current) return;
    const norm = canvasToNorm(event);
    const start = dragStartRef.current.norm;
    const dx = event.clientX - dragStartRef.current.screenX;
    const dy = event.clientY - dragStartRef.current.screenY;
    const moved = Math.hypot(dx, dy) >= 4;
    dragRef.current = null;
    dragStartRef.current = null;
    if (!norm) {
      setDragBox(null);
      return;
    }
    if (!moved) {
      setDragBox(null);
      selectBox(start.x, start.y);
      return;
    }
    const x1 = clamp01(Math.min(start.x, norm.x));
    const y1 = clamp01(Math.min(start.y, norm.y));
    const x2 = clamp01(Math.max(start.x, norm.x));
    const y2 = clamp01(Math.max(start.y, norm.y));

    const widthPx = Math.abs(x2 - x1) * imageSize.w;
    const heightPx = Math.abs(y2 - y1) * imageSize.h;
    setDragBox(null);
    if (widthPx < 2 || heightPx < 2) {
      return;
    }

    const newBox = {
      classId: currentClass,
      x1,
      y1,
      x2,
      y2,
      conf: meta.saveConf ? 1.0 : null,
      edited: true
    };
    const nextBoxes = [...boxes, newBox];
    setBoxes(nextBoxes);
    setSelectedIdx(nextBoxes.length - 1);
    queueSave(nextBoxes);
  }

  function selectBox(nx, ny) {
    for (let i = boxes.length - 1; i >= 0; i -= 1) {
      const box = boxes[i];
      if (nx >= box.x1 && nx <= box.x2 && ny >= box.y1 && ny <= box.y2) {
        setSelectedIdx(i);
        setCurrentClass(box.classId);
        return;
      }
    }
    setSelectedIdx(null);
  }

  function cycleClass(delta) {
    const list = meta.customClasses.length ? meta.customClasses : meta.classes;
    const total = list.length || 1;
    const next = (currentClass + delta + total) % total;
    setCurrentClass(next);
    if (selectedIdx === null) return;
    const nextBoxes = [...boxes];
    if (nextBoxes[selectedIdx]) {
      nextBoxes[selectedIdx] = {
        ...nextBoxes[selectedIdx],
        classId: next,
        edited: true
      };
      setBoxes(nextBoxes);
      queueSave(nextBoxes);
    }
  }

  function updateClass(event) {
    const value = Number(event.target.value);
    if (Number.isNaN(value)) return;
    setCurrentClass(value);
    if (selectedIdx === null) return;
    const nextBoxes = [...boxes];
    if (nextBoxes[selectedIdx]) {
      nextBoxes[selectedIdx] = {
        ...nextBoxes[selectedIdx],
        classId: value,
        edited: true
      };
      setBoxes(nextBoxes);
      queueSave(nextBoxes);
    }
  }

  function deleteSelected() {
    if (selectedIdx === null) return;
    const nextBoxes = boxes.filter((_, idx) => idx !== selectedIdx);
    setBoxes(nextBoxes);
    setSelectedIdx(null);
    queueSave(nextBoxes);
  }

  function clearAll() {
    if (!boxes.length) return;
    setBoxes([]);
    setSelectedIdx(null);
    queueSave([]);
  }

  function nextImage() {
    if (index < visibleImages.length - 1) {
      setIndex(index + 1);
    }
  }

  function prevImage() {
    if (index > 0) {
      setIndex(index - 1);
    }
  }

  async function saveLabels({ silent = false, boxesOverride, imageName } = {}) {
    const targetName = imageName || currentImageName;
    if (!targetName) return;
    if (!silent) {
      setStatus("Saving...");
    }
    try {
      const payloadBoxes = Array.isArray(boxesOverride) ? boxesOverride : boxes;
      const response = await fetch(withDataset(`/api/labels/${encodeURIComponent(targetName)}`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ boxes: payloadBoxes, saveConf: meta.saveConf })
      });
      if (!response.ok) {
        throw new Error("Save failed");
      }
      const data = await response.json();
      setLabelSourceMap((prev) => ({ ...prev, [targetName]: "edited" }));
      setEditedMap((prev) => ({ ...prev, [targetName]: true }));
      if (silent) {
        setStatus("Auto-saved");
        setTimeout(() => setStatus(""), 1200);
      } else {
        setStatus(`Saved to ${data.path}`);
        setTimeout(() => setStatus(""), 1800);
      }
    } catch (error) {
      setStatus(error.message || "Save failed");
    }
  }

  function jumpToIndex() {
    const value = Number(jumpValue);
    if (Number.isNaN(value)) return;
    const target = value - 1;
    if (target >= 0 && target < visibleImages.length) {
      setIndex(target);
    }
  }

  function toggleEditedOnly() {
    if (!showEditedOnly) {
      fullIndexRef.current = index;
      fullImageRef.current = currentImageName;
      const editedList = meta.images.filter((item) => editedMap[item.name]);
      if (editedList.length === 0) {
        setStatus("No edited images.");
        return;
      }
      const currentIdx = editedList.findIndex((item) => item.name === currentImageName);
      setShowEditedOnly(true);
      setIndex(currentIdx >= 0 ? currentIdx : 0);
      return;
    }
    setShowEditedOnly(false);
    const desiredName = fullImageRef.current;
    const fullIdx = desiredName
      ? meta.images.findIndex((item) => item.name === desiredName)
      : fullIndexRef.current;
    setIndex(fullIdx >= 0 ? fullIdx : 0);
  }

  const invalidInfo = getInvalidClassInfo(boxes, meta.classes.length);
  const datasetHasWarning =
    datasetAlert.classCount > 0 && datasetAlert.invalidImages.length > 0;
  const datasetIndices = datasetAlert.invalidImages
    .map((item) => item.index)
    .filter((value) => Number.isFinite(value));
  const datasetIndicesPreview = datasetIndices.slice(0, 12).join(", ");
  const datasetIndicesMore =
    datasetIndices.length > 12 ? datasetIndices.length - 12 : 0;
  const datasetMaxFound = datasetAlert.invalidImages.reduce((max, item) => {
    const value = Number(item.maxClassId);
    if (Number.isNaN(value)) return max;
    return Math.max(max, value);
  }, -Infinity);
  const datasetLimit = datasetAlert.classCount - 1;
  const datasetWarningText = datasetHasWarning
    ? `classes.txt max is ${datasetLimit}, but ${datasetAlert.invalidImages.length} image(s) exceed it${
        Number.isFinite(datasetMaxFound) ? ` (max found ${datasetMaxFound})` : ""
      }. Dataset indices: ${datasetIndicesPreview}${
        datasetIndicesMore ? ` (+${datasetIndicesMore} more)` : ""
      }.`
    : "";

  return (
    <div className="app">
      <header className="header">
        <div className="title">YOLO Label Web</div>
        <div className="meta">
          <span>{meta.datasetDir ? `Dataset: ${meta.datasetDir}` : ""}</span>
          <span>{meta.outputDir ? `Output: ${meta.outputDir}` : ""}</span>
        </div>
      </header>
      {datasetHasWarning && (
        <div className="global-alert" role="alert">
          <span className="global-alert-title">Dataset warning</span>
          <span>{datasetWarningText}</span>
        </div>
      )}
      <div className="content">
        <section className="canvas-panel" ref={containerRef}>
          <canvas
            ref={canvasRef}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
          />
          <div className="canvas-footer">
            <div className="image-name">
              {currentImageName || "No image loaded"}
            </div>
            <div className="status">{status}</div>
          </div>
        </section>
        <aside className="side-panel">
          <div className="section">
            <div className="section-title">Dataset</div>
            <div className="row">
              <input
                className="input"
                value={datasetInput}
                onChange={(event) => setDatasetInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    loadDataset();
                  }
                }}
                placeholder="Path to dataset"
              />
            </div>
            <div className="row">
              <button className="btn" onClick={loadDataset}>
                Load
              </button>
            </div>
          </div>
          <div className="section">
            <div className="section-title">Navigation</div>
            <div className="row">
              <button className="btn" onClick={prevImage} disabled={index === 0}>
                Prev
              </button>
              <button
                className="btn"
                onClick={nextImage}
                disabled={index >= visibleImages.length - 1}
              >
                Next
              </button>
            </div>
            <div className="row">
              <span className="small">
                {visibleImages.length ? `${index + 1} / ${visibleImages.length}` : "0 / 0"}
              </span>
            </div>
            <div className="row">
              <input
                className="input"
                value={jumpValue}
                onChange={(event) => setJumpValue(event.target.value)}
                placeholder="Jump to"
              />
              <button className="btn" onClick={jumpToIndex}>Go</button>
            </div>
            <div className="row">
              <button
                className={`btn ${showEditedOnly ? "btn-active" : ""}`}
                onClick={toggleEditedOnly}
                disabled={!showEditedOnly && Object.keys(editedMap).length === 0}
              >
                {showEditedOnly ? "Show All" : "Edited Only"}
              </button>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Labeling</div>
            <div className="row">
              <button className="btn" onClick={clearAll} disabled={!boxes.length}>
                Clear
              </button>
            </div>
            <div className="row">
              <select className="select" value={currentClass} onChange={updateClass}>
                {(meta.customClasses.length ? meta.customClasses : meta.classes).length
                  ? (meta.customClasses.length ? meta.customClasses : meta.classes).map(
                      (name, idx) => (
                        <option key={`${idx}-${name}`} value={idx}>
                          {idx}: {name}
                        </option>
                      )
                    )
                  : [
                      <option key="0" value={0}>
                        0: class_0
                      </option>
                    ]}
              </select>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Boxes</div>
            <div className="box-list">
              {boxes.length === 0 && <div className="empty">No boxes</div>}
              {boxes.map((box, idx) => {
                const name = labelNameFor(box);
                const confText =
                  meta.saveConf && typeof box.conf === "number"
                    ? ` ${box.conf.toFixed(2)}`
                    : "";
                const { ratioArea, pixelArea } = getBoxAreaStats(box);
                const areaText = `${Math.round(pixelArea)} px^2 · ${ratioArea.toFixed(4)}`;
                return (
                  <button
                    key={`${idx}-${name}`}
                    className={`box-row ${selectedIdx === idx ? "selected" : ""}`}
                    onClick={() => {
                      setSelectedIdx(idx);
                      setCurrentClass(box.classId);
                    }}
                  >
                    <span>{idx + 1}.</span>
                    <span>{name}</span>
                    <span className="small box-conf">{confText}</span>
                    <span className="small box-area">{areaText}</span>
                  </button>
                );
              })}
            </div>
            <div className="row">
              <button className="btn" onClick={deleteSelected} disabled={selectedIdx === null}>
                Delete
              </button>
              <button className="btn" onClick={saveLabels}>
                Save
              </button>
            </div>
          </div>

          <div className="section">
            <div className="section-title">Tips</div>
            <div className="tip">Drag on image to draw</div>
            <div className="tip">Save: S</div>
            <div className="tip">Delete: Del</div>
            <div className="tip">Prev/Next image: ← →</div>
            <div className="tip">Change class: ↑ ↓</div>
          </div>
        </aside>
      </div>
    </div>
  );
}
