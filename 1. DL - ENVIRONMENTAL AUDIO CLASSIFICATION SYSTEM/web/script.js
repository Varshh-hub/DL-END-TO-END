
const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");

const audioForm = document.getElementById("audioForm");
const audioFile = document.getElementById("audioFile");
const uploadZone = document.getElementById("uploadZone");
const uploadTitle = document.getElementById("uploadTitle");
const uploadDescription = document.getElementById("uploadDescription");

const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const audioPreview = document.getElementById("audioPreview");

const analyzeButton = document.getElementById("analyzeButton");
const formMessage = document.getElementById("formMessage");
const resultPanel = document.getElementById("resultPanel");
const predictionText = document.getElementById("predictionText");
const confidenceRow = document.getElementById("confidenceRow");
const confidenceText = document.getElementById("confidenceText");
const confidenceTrack = document.getElementById("confidenceTrack");
const confidenceBar = document.getElementById("confidenceBar");

const MAX_FILE_SIZE = 25 * 1024 * 1024;
const ALLOWED_EXTENSIONS = ["wav", "mp3", "ogg", "flac", "m4a"];

let currentAudioUrl = null;

menuToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("open");
    menuToggle.setAttribute("aria-expanded", String(isOpen));
});

navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
        navLinks.classList.remove("open");
        menuToggle.setAttribute("aria-expanded", "false");
    });
});

function showMessage(message, isError = false) {
    formMessage.textContent = message;
    formMessage.classList.toggle("error", isError);
}

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) {
        return `${(bytes / 1024).toFixed(1)} KB`;
    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function clearResult() {
    resultPanel.hidden = true;
    predictionText.textContent = "Waiting for prediction";
    confidenceRow.hidden = true;
    confidenceTrack.hidden = true;
    confidenceBar.style.width = "0%";
}

function resetFileInput() {
    audioFile.value = "";
    selectedFile.hidden = true;
    audioPreview.hidden = true;
    audioPreview.removeAttribute("src");

    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
        currentAudioUrl = null;
    }

    uploadTitle.textContent = "Drop your audio file here";
    uploadDescription.textContent = "or click to browse your device";

    clearResult();
    showMessage("");
}

function displaySelectedFile(file) {
    if (!file) {
        resetFileInput();
        return;
    }

    const extension = file.name.split(".").pop().toLowerCase();

    if (!ALLOWED_EXTENSIONS.includes(extension)) {
        resetFileInput();
        showMessage(
            "Unsupported file format. Choose WAV, MP3, OGG, FLAC, or M4A.",
            true
        );
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        resetFileInput();
        showMessage("Your audio file must be smaller than 25 MB.", true);
        return;
    }

    if (file.size === 0) {
        resetFileInput();
        showMessage("The selected file is empty.", true);
        return;
    }

    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);
    selectedFile.hidden = false;

    uploadTitle.textContent = "Audio file selected";
    uploadDescription.textContent = "Ready to analyze";

    if (currentAudioUrl) {
        URL.revokeObjectURL(currentAudioUrl);
    }

    currentAudioUrl = URL.createObjectURL(file);
    audioPreview.src = currentAudioUrl;
    audioPreview.hidden = false;

    clearResult();
    showMessage("");
}

audioFile.addEventListener("change", () => {
    displaySelectedFile(audioFile.files[0]);
});

removeFile.addEventListener("click", resetFileInput);

["dragenter", "dragover"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach((eventName) => {
    uploadZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        uploadZone.classList.remove("drag-over");
    });
});

uploadZone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];

    if (!file) {
        return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(file);
    audioFile.files = transfer.files;

    displaySelectedFile(file);
});

audioForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = audioFile.files[0];

    if (!file) {
        showMessage("Please select an audio file first.", true);
        return;
    }

    const formData = new FormData();
    formData.append("audio", file);

    analyzeButton.disabled = true;
    analyzeButton.innerHTML = "Analyzing audio...";
    resultPanel.hidden = true;
    showMessage("Processing audio and generating a prediction...");

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Audio analysis failed.");
        }

        predictionText.textContent = result.prediction;
        resultPanel.hidden = false;

        if (
            result.confidence !== null &&
            result.confidence !== undefined &&
            Number.isFinite(Number(result.confidence))
        ) {
            const confidence = Math.min(
                100,
                Math.max(0, Number(result.confidence))
            );

            confidenceText.textContent = `${confidence.toFixed(2)}%`;
            confidenceRow.hidden = false;
            confidenceTrack.hidden = false;

            requestAnimationFrame(() => {
                confidenceBar.style.width = `${confidence}%`;
            });
        } else {
            confidenceRow.hidden = true;
            confidenceTrack.hidden = true;
        }

        showMessage("Audio classification completed.");
        resultPanel.scrollIntoView({
            behavior: "smooth",
            block: "nearest"
        });
    } catch (error) {
        showMessage(error.message || "Something went wrong.", true);
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerHTML = 'Analyze audio <span>↗</span>';
    }
});

const pageSections = document.querySelectorAll("main section[id]");
const navItems = document.querySelectorAll(".nav-links a");

if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
        (entries) => {
            entries.forEach((entry) => {
                if (entry.isIntersecting) {
                    navItems.forEach((link) => {
                        link.classList.toggle(
                            "active",
                            link.getAttribute("href") === `#${entry.target.id}`
                        );
                    });
                }
            });
        },
        {
            rootMargin: "-30% 0px -60% 0px"
        }
    );

    pageSections.forEach((section) => observer.observe(section));
}