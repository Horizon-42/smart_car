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

export default function App() {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const dragRef = useRef(null);

  const [meta, setMeta] = useState(emptyMeta);
  const [index, setIndex] = useState(0);
  const [boxes, setBoxes] = useState([]);
  const [selectedIdx, setSelectedIdx] = useState(null);
  const [drawMode, setDrawMode] = useState(false);
  const [status, setStatus] = useState("Loading dataset...");
  const [image, setImage] = useState(null);
  const [imageSize, setImageSize] = useState({ w: 1, h: 1 });
  const [dragBox, setDragBox] = useState(null);
  const [currentClass, setCurrentClass] = useState(0);
  const [jumpValue, setJumpValue] = useState("");

  const currentImage = meta.images[index];
  const currentImageName = currentImage ? currentImage.name : "";

  useEffect(() => {
    let active = true;
    async function loadMeta() {
      try {
        const response = await fetch("/api/meta");
        if (!response.ok) {
          throw new Error("Failed to load dataset metadata");
        }
        const data = await response.json();
        if (!active) return;
        setMeta(data);
        setIndex(0);
        setCurrentClass(0);
        setStatus(data.images.length ? "" : "No images found.");
      } catch (error) {
        if (active) {
          setStatus(error.message || "Failed to load dataset.");
        }
      }
    }
    loadMeta();
    return () => {
      active = false;
    };
  }, []);

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
    img.src = `/data/images/${encodeURIComponent(currentImageName)}`;

    async function loadLabels() {
      try {
        const response = await fetch(`/api/labels/${encodeURIComponent(currentImageName)}`);
        if (!response.ok) {
          throw new Error("Failed to load labels");
        }
        const data = await response.json();
        if (!active) return;
        const nextBoxes = Array.isArray(data.boxes) ? data.boxes : [];
        setBoxes(nextBoxes);
        if (nextBoxes.some((box) => typeof box.conf === "number")) {
          setMeta((prev) => ({ ...prev, saveConf: true }));
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
  }, [currentImageName]);

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
  }, [image, boxes, selectedIdx, dragBox, drawMode]);

  useEffect(() => {
    const handler = (event) => {
      const tag = event.target.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
        return;
      }
      if (event.key === "ArrowRight") {
        nextImage();
      } else if (event.key === "ArrowLeft") {
        prevImage();
      } else if (event.key === "s" || event.key === "S") {
        saveLabels();
      } else if (event.key === "a" || event.key === "A") {
        setDrawMode((prev) => !prev);
      } else if (event.key === "Delete") {
        deleteSelected();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [index, meta.images.length, boxes, selectedIdx, drawMode, currentClass]);

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

      const className = meta.classes[box.classId] || `class_${box.classId}`;
      const confText =
        meta.saveConf && typeof box.conf === "number"
          ? ` ${box.conf.toFixed(2)}`
          : "";
      ctx.fillStyle = color;
      ctx.font = "12px sans-serif";
      ctx.fillText(`${className}${confText}`, x1 + 4, y1 + 12);
    });

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
    if (drawMode) {
      dragRef.current = norm;
      setDragBox({ x1: norm.x, y1: norm.y, x2: norm.x, y2: norm.y });
      return;
    }
    selectBox(norm.x, norm.y);
  }

  function onMouseMove(event) {
    if (!drawMode || !dragRef.current) return;
    const norm = canvasToNorm(event);
    if (!norm) return;
    const start = dragRef.current;
    const x1 = Math.min(start.x, norm.x);
    const y1 = Math.min(start.y, norm.y);
    const x2 = Math.max(start.x, norm.x);
    const y2 = Math.max(start.y, norm.y);
    setDragBox({ x1, y1, x2, y2 });
  }

  function onMouseUp(event) {
    if (!drawMode || !dragRef.current) return;
    const norm = canvasToNorm(event);
    const start = dragRef.current;
    dragRef.current = null;
    if (!norm) {
      setDragBox(null);
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
      conf: meta.saveConf ? 1.0 : null
    };
    setBoxes((prev) => {
      const next = [...prev, newBox];
      setSelectedIdx(next.length - 1);
      return next;
    });
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

  function updateClass(event) {
    const value = Number(event.target.value);
    if (Number.isNaN(value)) return;
    setCurrentClass(value);
    if (selectedIdx === null) return;
    setBoxes((prev) => {
      const next = [...prev];
      if (next[selectedIdx]) {
        next[selectedIdx] = { ...next[selectedIdx], classId: value };
      }
      return next;
    });
  }

  function deleteSelected() {
    if (selectedIdx === null) return;
    setBoxes((prev) => prev.filter((_, idx) => idx !== selectedIdx));
    setSelectedIdx(null);
  }

  function clearAll() {
    if (!boxes.length) return;
    const confirmed = window.confirm("Clear all boxes for this image?");
    if (!confirmed) return;
    setBoxes([]);
    setSelectedIdx(null);
  }

  function nextImage() {
    if (index < meta.images.length - 1) {
      setIndex(index + 1);
    }
  }

  function prevImage() {
    if (index > 0) {
      setIndex(index - 1);
    }
  }

  async function saveLabels() {
    if (!currentImageName) return;
    setStatus("Saving...");
    try {
      const response = await fetch(`/api/labels/${encodeURIComponent(currentImageName)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ boxes, saveConf: meta.saveConf })
      });
      if (!response.ok) {
        throw new Error("Save failed");
      }
      const data = await response.json();
      setStatus(`Saved to ${data.path}`);
      setTimeout(() => setStatus(""), 1800);
    } catch (error) {
      setStatus(error.message || "Save failed");
    }
  }

  function jumpToIndex() {
    const value = Number(jumpValue);
    if (Number.isNaN(value)) return;
    const target = value - 1;
    if (target >= 0 && target < meta.images.length) {
      setIndex(target);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="title">YOLO Label Web</div>
        <div className="meta">
          <span>{meta.datasetDir ? `Dataset: ${meta.datasetDir}` : ""}</span>
          <span>{meta.outputDir ? `Output: ${meta.outputDir}` : ""}</span>
        </div>
      </header>
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
            <div className="section-title">Navigation</div>
            <div className="row">
              <button className="btn" onClick={prevImage} disabled={index === 0}>
                Prev
              </button>
              <button
                className="btn"
                onClick={nextImage}
                disabled={index >= meta.images.length - 1}
              >
                Next
              </button>
            </div>
            <div className="row">
              <span className="small">{meta.images.length ? `${index + 1} / ${meta.images.length}` : "0 / 0"}</span>
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
          </div>

          <div className="section">
            <div className="section-title">Draw</div>
            <div className="row">
              <button className={`btn ${drawMode ? "btn-active" : ""}`} onClick={() => setDrawMode((prev) => !prev)}>
                {drawMode ? "Drawing" : "Draw Box"}
              </button>
              <button className="btn" onClick={clearAll} disabled={!boxes.length}>
                Clear
              </button>
            </div>
            <div className="row">
              <select className="select" value={currentClass} onChange={updateClass}>
                {meta.classes.length
                  ? meta.classes.map((name, idx) => (
                      <option key={name} value={idx}>
                        {idx}: {name}
                      </option>
                    ))
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
                const name = meta.classes[box.classId] || `class_${box.classId}`;
                const confText =
                  meta.saveConf && typeof box.conf === "number"
                    ? ` ${box.conf.toFixed(2)}`
                    : "";
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
                    <span className="small">{confText}</span>
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
            <div className="tip">Toggle draw: A</div>
            <div className="tip">Save: S</div>
            <div className="tip">Delete: Del</div>
            <div className="tip">Prev/Next: ← →</div>
          </div>
        </aside>
      </div>
    </div>
  );
}
