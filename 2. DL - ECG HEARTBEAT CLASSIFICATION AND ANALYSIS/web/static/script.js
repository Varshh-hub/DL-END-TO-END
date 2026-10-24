const form = document.getElementById("uploadForm");
const input = document.getElementById("fileInput");
const dropzone = document.getElementById("dropzone");
const fileMeta = document.getElementById("fileMeta");
const analyzeBtn = document.getElementById("analyzeBtn");
const emptyState = document.getElementById("emptyState");
const loadingState = document.getElementById("loadingState");
const resultState = document.getElementById("resultState");
const errorBox = document.getElementById("errorBox");
const resultTag = document.getElementById("resultTag");
const chartPlaceholder = document.getElementById("chartPlaceholder");
const chartWrap = document.getElementById("chartWrap");
let chart = null;

function showFile(file) {
  if (!file) return;
  fileMeta.textContent = `SELECTED · ${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  fileMeta.classList.remove("hidden");
}
input.addEventListener("change", () => showFile(input.files[0]));
["dragenter", "dragover"].forEach(name => dropzone.addEventListener(name, event => {
  event.preventDefault(); dropzone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(name => dropzone.addEventListener(name, event => {
  event.preventDefault(); dropzone.classList.remove("dragover");
}));
dropzone.addEventListener("drop", event => {
  if (event.dataTransfer.files.length) {
    input.files = event.dataTransfer.files;
    showFile(input.files[0]);
  }
});
function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
}
function readValues(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const values = String(reader.result).trim().split(/[,\n\r]+/).map(v => Number(v.trim()));
      if (!values.length || values.some(v => !Number.isFinite(v))) {
        reject(new Error("CSV must contain only numeric ECG samples."));
      } else resolve(values);
    };
    reader.onerror = () => reject(new Error("Could not read this file."));
    reader.readAsText(file);
  });
}
function drawChart(values) {
  chartPlaceholder.classList.add("hidden");
  chartWrap.classList.remove("hidden");
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("ecgChart"), {
    type: "line",
    data: { labels: values.map((_, i) => i + 1), datasets: [{
      data: values, label: "ECG amplitude", borderColor: "#7C0E44",
      borderWidth: 2, pointRadius: 0, tension: .12, fill: false
    }]},
    options: { responsive: true, maintainAspectRatio: false, animation: false,
      plugins: { legend: { labels: { color: "#7C0E44", boxWidth: 12 } } },
      scales: {
        x: { title: { display: true, text: "Sample index", color: "#886979" },
          ticks: { color: "#886979", maxTicksLimit: 9 }, grid: { color: "rgba(124,14,68,.08)" } },
        y: { title: { display: true, text: "Amplitude", color: "#886979" },
          ticks: { color: "#886979" }, grid: { color: "rgba(124,14,68,.08)" } }
      }
    }
  });
}
form.addEventListener("submit", async event => {
  event.preventDefault();
  errorBox.classList.add("hidden");
  const file = input.files[0];
  if (!file) return showError("Choose a CSV file first.");
  if (!file.name.toLowerCase().endsWith(".csv")) return showError("Please upload a .csv file.");
  emptyState.classList.add("hidden");
  resultState.classList.add("hidden");
  loadingState.classList.remove("hidden");
  analyzeBtn.disabled = true;
  try {
    const values = await readValues(file);
    drawChart(values);
    const body = new FormData();
    body.append("file", file);
    const response = await fetch("/predict", { method: "POST", body });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Prediction failed.");
    const pred = document.getElementById("prediction");
    pred.textContent = result.prediction;
    pred.className = "prediction " + (result.class_id === 1 ? "abnormal" : "normal");
    document.getElementById("probability").textContent = (result.probability_abnormal * 100).toFixed(1) + "%";
    document.getElementById("meterFill").style.width = (result.probability_abnormal * 100) + "%";
    document.getElementById("threshold").textContent = result.threshold;
    document.getElementById("sampleCount").textContent = result.samples;
    document.getElementById("note").textContent = result.note;
    resultTag.textContent = "INFERENCE COMPLETE";
    loadingState.classList.add("hidden");
    resultState.classList.remove("hidden");
  } catch (error) {
    loadingState.classList.add("hidden");
    emptyState.classList.remove("hidden");
    showError(error.message || "An unexpected error occurred.");
  } finally {
    analyzeBtn.disabled = false;
  }
});
