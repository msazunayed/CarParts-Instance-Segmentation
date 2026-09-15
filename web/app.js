(() => {
  const $ = (sel) => document.querySelector(sel);

  const fileInput = $("#file-input");
  const dropzone = $("#dropzone");
  const sampleStrip = $("#sample-strip");
  const confSlider = $("#conf-slider");
  const iouSlider = $("#iou-slider");
  const confValue = $("#conf-value");
  const iouValue = $("#iou-value");
  const showLabels = $("#show-labels");
  const showConf = $("#show-conf");
  const modelBadge = $("#model-badge");
  const modelBadgeText = $("#model-badge-text");

  const emptyState = $("#empty-state");
  const loadingState = $("#loading-state");
  const errorState = $("#error-state");
  const errorMessage = $("#error-message");
  const results = $("#results");

  const imgOriginal = $("#img-original");
  const imgSegmented = $("#img-segmented");
  const viewerClip = $("#viewer-clip");
  const labelOverlay = $("#label-overlay");
  const compareSlider = $("#compare-slider");
  const viewer = $("#viewer");
  const downloadBtn = $("#download-btn");

  const statParts = $("#stat-parts");
  const statClasses = $("#stat-classes");
  const statFps = $("#stat-fps");
  const statMs = $("#stat-ms");
  const specBody = $("#spec-body");
  const specTable = $("#spec-table");
  const specEmpty = $("#spec-empty");
  const aboutClasses = $("#about-classes");

  let currentSource = null; // The selected upload or sample.
  let lastAnnotatedDataUrl = null;
  let debounceTimer = null;
  let currentDetections = []; // Detections from the latest response.
  let currentViewerMode = "compare";

  // Show the loaded model and its default threshold.
  fetch("/api/model-info")
    .then((r) => r.json())
    .then((info) => {
      confSlider.value = info.default_conf;
      confValue.textContent = Number(info.default_conf).toFixed(2);
      if (info.is_custom) {
        modelBadge.classList.add("ok");
        modelBadgeText.textContent = "Custom trained weights loaded";
      } else {
        modelBadge.classList.add("warn");
        modelBadgeText.textContent = "Stock yolo26n-seg.pt (no trained weights found)";
      }
      aboutClasses.textContent = info.classes.join(", ");
    })
    .catch(() => {
      modelBadgeText.textContent = "Model status unavailable";
    });

  // Load the sample images for the sidebar.
  fetch("/api/samples")
    .then((r) => r.json())
    .then(({ samples }) => {
      if (!samples || samples.length === 0) return;
      sampleStrip.innerHTML = "";
      samples.forEach((name) => {
        const img = document.createElement("img");
        img.className = "sample-thumb";
        img.src = `/api/samples/${encodeURIComponent(name)}`;
        img.alt = name;
        img.title = name;
        img.addEventListener("click", () => {
          document.querySelectorAll(".sample-thumb").forEach((el) => el.classList.remove("selected"));
          img.classList.add("selected");
          currentSource = { kind: "sample", name };
          runInference();
        });
        sampleStrip.appendChild(img);
      });
    })
    .catch(() => {});

  // Handle uploads from the file picker and drop zone.
  dropzone.addEventListener("click", (e) => {
  });

  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    currentSource = { kind: "file", file };
    document.querySelectorAll(".sample-thumb").forEach((el) => el.classList.remove("selected"));
    runInference();
  });

  ["dragover", "dragenter"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (!file) return;
    fileInput.files = e.dataTransfer.files;
    currentSource = { kind: "file", file };
    document.querySelectorAll(".sample-thumb").forEach((el) => el.classList.remove("selected"));
    runInference();
  });

  // Wait briefly before running inference while a slider is moving.
  function debouncedRerun() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (currentSource) runInference();
    }, 350);
  }

  confSlider.addEventListener("input", () => {
    confValue.textContent = Number(confSlider.value).toFixed(2);
    debouncedRerun();
  });
  iouSlider.addEventListener("input", () => {
    iouValue.textContent = Number(iouSlider.value).toFixed(2);
    debouncedRerun();
  });
  // These toggles only change the existing label overlay.
  showLabels.addEventListener("change", updateLabelOverlay);
  showConf.addEventListener("change", updateLabelOverlay);

  function setState(state) {
    emptyState.classList.add("hidden");
    loadingState.classList.add("hidden");
    errorState.classList.add("hidden");
    results.classList.add("hidden");
    if (state === "empty") emptyState.classList.remove("hidden");
    if (state === "loading") loadingState.classList.remove("hidden");
    if (state === "error") errorState.classList.remove("hidden");
    if (state === "results") results.classList.remove("hidden");
  }

  async function runInference() {
    if (!currentSource) return;
    setState("loading");

    const form = new FormData();
    if (currentSource.kind === "file") {
      form.append("file", currentSource.file);
    } else {
      form.append("sample", currentSource.name);
    }
    form.append("conf", confSlider.value);
    form.append("iou", iouSlider.value);

    try {
      const res = await fetch("/api/predict", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      renderResults(data);
    } catch (err) {
      errorMessage.textContent = err.message || "Something went wrong.";
      setState("error");
    }
  }

  function renderResults(data) {
    imgOriginal.src = data.original_image;
    imgSegmented.src = data.annotated_image;
    lastAnnotatedDataUrl = data.annotated_image;
    currentDetections = data.detections;

    statParts.textContent = data.stats.parts_detected;
    statClasses.textContent = data.stats.unique_classes;
    statFps.textContent = data.stats.fps.toFixed(1);
    statMs.textContent = data.stats.elapsed_ms;

    specBody.innerHTML = "";
    if (data.detections.length === 0) {
      specEmpty.classList.remove("hidden");
      specTable.classList.add("hidden");
    } else {
      specEmpty.classList.add("hidden");
      specTable.classList.remove("hidden");
      data.detections.forEach((d) => {
        const pct = Math.round(d.confidence * 100);
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td class="spec-part">${escapeHtml(d.part)}</td>
          <td class="spec-conf">${d.confidence.toFixed(3)}</td>
          <td class="spec-bar-cell"><div class="spec-bar-track"><div class="spec-bar-fill" style="width:${pct}%"></div></div></td>
        `;
        specBody.appendChild(tr);
      });
    }

    buildLabelOverlay(data.detections);
    setViewerMode("compare");
    setState("results");
  }

  // Place one label at each detection's bounding-box corner.
  function buildLabelOverlay(detections) {
    labelOverlay.innerHTML = "";
    detections.forEach((d, i) => {
      const chip = document.createElement("div");
      chip.className = "part-label";
      chip.dataset.x1 = d.bbox.x1;
      chip.dataset.x2 = d.bbox.x2;
      chip.style.left = `${d.bbox.x1 * 100}%`;
      chip.style.top = `${Math.max(d.bbox.y1, 0.02) * 100}%`;
      chip.innerHTML = `<span class="part-label-name">${escapeHtml(d.part)}</span><span class="part-label-conf">${d.confidence.toFixed(2)}</span>`;
      labelOverlay.appendChild(chip);
    });
    updateLabelOverlay();
  }

  // Keep labels on the visible side of the comparison divider.
  function updateLabelOverlay() {
    const pct = Number(compareSlider.value) / 100;
    const chips = labelOverlay.querySelectorAll(".part-label");
    chips.forEach((chip) => {
      const nameEl = chip.querySelector(".part-label-name");
      const confEl = chip.querySelector(".part-label-conf");
      if (nameEl) nameEl.classList.toggle("hidden", !showLabels.checked);
      if (confEl) confEl.classList.toggle("hidden", !showConf.checked);

      let visible = showLabels.checked || showConf.checked;
      if (visible && currentViewerMode === "compare") {
        const x2 = Number(chip.dataset.x2);
        visible = x2 <= pct;
      } else if (visible && currentViewerMode === "original") {
        visible = false;
      }
      chip.classList.toggle("hidden", !visible);
    });
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }

  // Keep the segmented image aligned with the comparison slider.
  function applyClip(pct) {
    viewerClip.style.width = `${pct}%`;
    $("#compare-handle").style.left = `${pct}%`;
    imgSegmented.style.width = `${(100 / pct) * 100}%`;
  }

  compareSlider.addEventListener("input", () => {
    const pct = Number(compareSlider.value);
    applyClip(pct);
    updateLabelOverlay();
  });

  new ResizeObserver(() => applyClip(Number(compareSlider.value))).observe(viewer);

  function setViewerMode(mode) {
    currentViewerMode = mode;
    document.querySelectorAll(".viewer-tab").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.mode === mode);
    });
    const frame = document.querySelector(".viewer-frame");
    const compareHandle = $("#compare-handle");
    if (mode === "compare") {
      compareSlider.classList.remove("hidden");
      compareHandle.classList.remove("hidden");
      imgOriginal.classList.remove("hidden");
      viewerClip.classList.remove("hidden");
      applyClip(Number(compareSlider.value));
    } else if (mode === "original") {
      compareSlider.classList.add("hidden");
      compareHandle.classList.add("hidden");
      imgOriginal.classList.remove("hidden");
      viewerClip.classList.add("hidden");
    } else if (mode === "segmented") {
      compareSlider.classList.add("hidden");
      compareHandle.classList.add("hidden");
      imgOriginal.classList.add("hidden");
      viewerClip.classList.remove("hidden");
      viewerClip.style.width = "100%";
      imgSegmented.style.width = "100%";
    }
    updateLabelOverlay();
  }

  document.querySelectorAll(".viewer-tab").forEach((btn) => {
    btn.addEventListener("click", () => setViewerMode(btn.dataset.mode));
  });

  downloadBtn.addEventListener("click", () => {
    if (!lastAnnotatedDataUrl) return;
    const a = document.createElement("a");
    a.href = lastAnnotatedDataUrl;
    a.download = "carparts_segmented.png";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  setState("empty");
})();
