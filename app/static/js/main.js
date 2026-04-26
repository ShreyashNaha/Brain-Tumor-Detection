/* ───────────────────────────────────────────────────────────────────────────
   main.js — NeuroScan Frontend Logic
─────────────────────────────────────────────────────────────────────────── */

// ── DOM References ─────────────────────────────────────────────────────────────
const modelSelect    = document.getElementById('modelSelect');
const modelMeta      = document.getElementById('modelMeta');
const dropZone       = document.getElementById('dropZone');
const fileInput      = document.getElementById('fileInput');
const dropPreview    = document.getElementById('dropPreview');
const previewImg     = document.getElementById('previewImg');
const previewName    = document.getElementById('previewName');
const runBtn         = document.getElementById('runBtn');
const runBtnText     = document.getElementById('runBtnText');
const progressWrap   = document.getElementById('progressWrap');
const progressBar    = document.getElementById('progressBar');
const progressLabel  = document.getElementById('progressLabel');
const statusDot      = document.getElementById('statusDot');
const statusText     = document.getElementById('statusText');
const emptyState     = document.getElementById('emptyState');
const resultsContent = document.getElementById('resultsContent');
const resultHint     = document.getElementById('resultHint');

// Result elements
const predBanner  = document.getElementById('predictionBanner');
const predClass   = document.getElementById('predClass');
const predConf    = document.getElementById('predConf');
const predInd     = document.getElementById('predIndicator');
const imgOriginal = document.getElementById('imgOriginal');
const imgEnhanced = document.getElementById('imgEnhanced');
const imgOverlay  = document.getElementById('imgOverlay');
const imageBadge  = document.getElementById('imageBadge');
const confBars    = document.getElementById('confBars');

// ── State ──────────────────────────────────────────────────────────────────────
let selectedFile = null;

// ── Init ───────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadModels();
  setupDropZone();
  modelSelect.addEventListener('change', onModelChange);
  runBtn.addEventListener('click', runAnalysis);
});

// ── Status helpers ─────────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusText.textContent = text;
  statusDot.className = 'status-dot' + (state === 'busy' ? ' busy' : state === 'error' ? ' error' : '');
}

// ── Load model list ────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const res  = await fetch('/api/models');
    const data = await res.json();
    modelSelect.innerHTML = '';

    if (!data.models || data.models.length === 0) {
      modelSelect.innerHTML = '<option value="">— no models found in app/models/ —</option>';
      setStatus('error', 'NO MODELS FOUND');
      return;
    }

    modelSelect.innerHTML = '<option value="">— select a model —</option>';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m; opt.textContent = m;
      modelSelect.appendChild(opt);
    });
    setStatus('ready', 'SYSTEM READY');
  } catch (e) {
    modelSelect.innerHTML = '<option value="">— failed to fetch —</option>';
    setStatus('error', 'API ERROR');
  }
}

// ── Model selection → fetch meta ───────────────────────────────────────────────
async function onModelChange() {
  const filename = modelSelect.value;
  modelMeta.classList.add('hidden');
  if (!filename) { checkReady(); return; }

  try {
    const res  = await fetch(`/api/model-info?filename=${encodeURIComponent(filename)}`);
    const data = await res.json();
    if (data.meta) {
      const m = data.meta;
      const acc = m.best_val_acc !== null ? (m.best_val_acc * 100).toFixed(2) + '%' : 'N/A';
      modelMeta.innerHTML = `
        <span>Architecture:</span> ${m.model_name}<br/>
        <span>Classes:</span> ${m.num_classes}<br/>
        <span>Input size:</span> ${m.img_size}×${m.img_size}<br/>
        <span>Val Acc (training):</span> ${acc}
      `;
      modelMeta.classList.remove('hidden');
    }
  } catch (e) { /* ignore */ }
  checkReady();
}

// ── Drop Zone ──────────────────────────────────────────────────────────────────
function setupDropZone() {
  dropZone.addEventListener('click', () => fileInput.click());
  fileInput.addEventListener('change', e => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
  });

  dropZone.addEventListener('dragover', e => {
    e.preventDefault(); dropZone.classList.add('dragging');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });
}

function handleFile(file) {
  const allowed = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp'];
  if (!allowed.some(t => file.type === t || file.name.endsWith(t.split('/')[1]))) {
    alert('Unsupported file type. Please upload a PNG, JPG, BMP, TIFF, or WEBP image.');
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    dropPreview.classList.remove('hidden');
    dropZone.querySelector('.drop-icon').style.display = 'none';
    dropZone.querySelector('.drop-text').style.display = 'none';
  };
  reader.readAsDataURL(file);
  checkReady();
}

function checkReady() {
  const ready = selectedFile && modelSelect.value;
  runBtn.disabled = !ready;
}

// ── Analysis ───────────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!selectedFile || !modelSelect.value) return;

  // UI: loading state
  runBtn.disabled = true;
  runBtnText.textContent = 'ANALYZING…';
  progressWrap.classList.remove('hidden');
  setStatus('busy', 'PROCESSING SCAN…');
  resultHint.textContent = 'Running inference pipeline…';

  const steps = [
    [10,  'Loading model weights…'],
    [30,  'Preprocessing MRI image…'],
    [55,  'Running EfficientNet inference…'],
    [75,  'Applying IFFT reconstruction…'],
    [90,  'Generating GradCAM heatmap…'],
    [100, 'Finalizing results…'],
  ];

  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx < steps.length) {
      const [pct, label] = steps[stepIdx++];
      progressBar.style.width = pct + '%';
      progressLabel.textContent = label;
    } else {
      clearInterval(stepInterval);
    }
  }, 350);

  try {
    const formData = new FormData();
    formData.append('image', selectedFile);
    formData.append('model_filename', modelSelect.value);

    const res  = await fetch('/api/analyze', { method: 'POST', body: formData });
    const json = await res.json();

    clearInterval(stepInterval);
    progressBar.style.width = '100%';

    if (!json.success) throw new Error(json.error || 'Unknown error');

    displayResults(json.data);
    setStatus('ready', 'ANALYSIS COMPLETE');
    resultHint.textContent = `Last scan: ${selectedFile.name}`;

  } catch (err) {
    clearInterval(stepInterval);
    setStatus('error', 'PIPELINE ERROR');
    progressLabel.textContent = `Error: ${err.message}`;
    resultHint.textContent = `Error — ${err.message}`;
    console.error(err);
  } finally {
    runBtnText.textContent = 'RUN ANALYSIS';
    runBtn.disabled = false;
    setTimeout(() => progressWrap.classList.add('hidden'), 1800);
  }
}

// ── Display Results ────────────────────────────────────────────────────────────
function displayResults(data) {
  emptyState.classList.add('hidden');
  resultsContent.classList.remove('hidden');

  // Images
  imgOriginal.src = 'data:image/png;base64,' + data.original_b64;
  imgEnhanced.src = 'data:image/png;base64,' + data.enhanced_b64;
  imgOverlay.src  = 'data:image/png;base64,' + data.overlay_b64;

  // Prediction banner
  const hasTumor = data.has_tumor;
  predBanner.className = 'prediction-banner ' + (hasTumor ? 'tumor' : 'no-tumor');
  predClass.textContent = data.prediction.toUpperCase();
  predClass.style.color = hasTumor ? 'var(--red)' : 'var(--green)';

  // Top confidence
  const confs = data.confidences;
  const topConf = confs[data.prediction] || 0;
  predConf.textContent = topConf.toFixed(1) + '%';

  // Indicator stripe
  predInd.style.background = hasTumor ? 'var(--red)' : 'var(--green)';
  predInd.style.boxShadow  = hasTumor
    ? '0 0 14px var(--red-glow)'
    : '0 0 14px var(--green-glow)';

  // Overlay badge
  if (hasTumor) {
    imageBadge.textContent = '⚠ TUMOR DETECTED';
    imageBadge.className   = 'image-badge tumor-badge';
  } else {
    imageBadge.textContent = '✓ NO TUMOR';
    imageBadge.className   = 'image-badge clear-badge';
  }

  // Confidence bars
  confBars.innerHTML = '';
  const maxConf = Math.max(...Object.values(confs));
  Object.entries(confs)
    .sort((a, b) => b[1] - a[1])
    .forEach(([cls, pct]) => {
      const isTop = pct === maxConf;
      const row = document.createElement('div');
      row.className = 'conf-bar-row';
      row.innerHTML = `
        <span class="conf-bar-label">${cls}</span>
        <div class="conf-bar-track">
          <div class="conf-bar-fill ${isTop ? 'highest' : ''}"
               style="width: 0%"
               data-pct="${pct}">
          </div>
        </div>
        <span class="conf-bar-pct">${pct.toFixed(1)}%</span>
      `;
      confBars.appendChild(row);
    });

  // Animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.conf-bar-fill').forEach(el => {
      el.style.width = el.dataset.pct + '%';
    });
  });
}