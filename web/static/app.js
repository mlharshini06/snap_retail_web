"use strict";

/* -----------------------------------------------------------------
   Snap Retail Mirror — frontend
   Flow: Start Camera -> Capture & Analyze -> Get Recommendations
   No keyboard shortcuts (mobile-friendly), everything is tap/click.
------------------------------------------------------------------ */

const els = {
  video: document.getElementById("video"),
  canvas: document.getElementById("captureCanvas"),
  placeholder: document.getElementById("mirrorPlaceholder"),
  capturedPreview: document.getElementById("capturedPreview"),
  liveDot: document.getElementById("liveDot"),

  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  captureBtn: document.getElementById("captureBtn"),
  retakeBtn: document.getElementById("retakeBtn"),

  statusMessage: document.getElementById("statusMessage"),

  detectionCard: document.getElementById("detectionCard"),
  skinSwatch: document.getElementById("skinSwatch"),
  garmentSwatch: document.getElementById("garmentSwatch"),
  clothingType: document.getElementById("clothingType"),
  recommendBtn: document.getElementById("recommendBtn"),

  emptyState: document.getElementById("emptyState"),
  loadingState: document.getElementById("loadingState"),
  errorState: document.getElementById("errorState"),
  errorText: document.getElementById("errorText"),
  retryBtn: document.getElementById("retryBtn"),
  recommendationContent: document.getElementById("recommendationContent"),
  colorList: document.getElementById("colorList"),
  outfitList: document.getElementById("outfitList"),
  tipList: document.getElementById("tipList"),
};

let mediaStream = null;
let capturedBlob = null;

const MSG = {
  cameraDenied: "Please allow camera access to use Snap Retail Mirror.",
  serverError: "Unable to connect to the AI service. Please try again.",
  recommendFailed: "We couldn't generate recommendations right now. Please try again.",
  invalidImage: "Please make sure a person is clearly visible.",
  noCameraDevice: "No camera was found on this device.",
};

// ------------------------------------------------------------------
// Status message helper
// ------------------------------------------------------------------
function showStatus(text, kind = "error") {
  els.statusMessage.textContent = text;
  els.statusMessage.classList.toggle("info", kind === "info");
  els.statusMessage.hidden = false;
}

function hideStatus() {
  els.statusMessage.hidden = true;
}

// ------------------------------------------------------------------
// Camera lifecycle
// ------------------------------------------------------------------
async function startCamera() {
  hideStatus();
  els.startBtn.disabled = true;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showStatus(MSG.noCameraDevice);
    els.startBtn.disabled = false;
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 960 } },
      audio: false,
    });
  } catch (err) {
    showStatus(MSG.cameraDenied);
    els.startBtn.disabled = false;
    return;
  }

  els.video.srcObject = mediaStream;
  els.video.hidden = false;
  els.placeholder.hidden = true;
  els.capturedPreview.hidden = true;
  els.liveDot.hidden = false;

  els.stopBtn.disabled = false;
  els.captureBtn.disabled = false;
  els.retakeBtn.hidden = true;
}

function stopCamera() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  els.video.srcObject = null;
  els.video.hidden = true;
  els.placeholder.hidden = false;
  els.capturedPreview.hidden = true;
  els.liveDot.hidden = true;

  els.startBtn.disabled = false;
  els.stopBtn.disabled = true;
  els.captureBtn.disabled = true;
  els.retakeBtn.hidden = true;

  hideDetectionCard();
}

// ------------------------------------------------------------------
// Capture -> /analyze
// ------------------------------------------------------------------
function captureFrame() {
  return new Promise((resolve) => {
    const video = els.video;
    const canvas = els.canvas;
    canvas.width = video.videoWidth || 1280;
    canvas.height = video.videoHeight || 960;
    const ctx = canvas.getContext("2d");
    // Mirror horizontally to match the on-screen preview.
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
  });
}

async function onCapture() {
  hideStatus();
  els.captureBtn.disabled = true;

  const blob = await captureFrame();
  if (!blob) {
    showStatus(MSG.serverError);
    els.captureBtn.disabled = false;
    return;
  }
  capturedBlob = blob;

  // Freeze the frame in the mirror so the user sees exactly what was analyzed.
  const url = URL.createObjectURL(blob);
  els.capturedPreview.src = url;
  els.capturedPreview.hidden = false;
  els.video.hidden = true;
  els.liveDot.hidden = true;
  els.retakeBtn.hidden = false;

  showStatus("Analyzing…", "info");

  try {
    const form = new FormData();
    form.append("file", blob, "capture.jpg");
    const resp = await fetch("/analyze", { method: "POST", body: form });
    const data = await resp.json();

    if (!resp.ok && resp.status >= 500) {
      showStatus(MSG.serverError);
      return;
    }
    if (!data.person_detected) {
      showStatus(data.message || MSG.invalidImage);
      hideDetectionCard();
      return;
    }

    hideStatus();
    if (data.message) showStatus(data.message, "info");
    renderDetection(data);
  } catch (err) {
    showStatus(MSG.serverError);
  } finally {
    els.captureBtn.disabled = false;
  }
}

function onRetake() {
  capturedBlob = null;
  els.capturedPreview.hidden = true;
  els.video.hidden = false;
  els.liveDot.hidden = false;
  els.retakeBtn.hidden = true;
  hideStatus();
  hideDetectionCard();
}

function renderDetection(data) {
  els.skinSwatch.style.background = data.skin_tone || "#2A261F";
  els.garmentSwatch.style.background = data.garment_color || "#2A261F";
  els.clothingType.textContent = data.clothing_type || "Unknown";
  els.detectionCard.hidden = false;
  els.recommendBtn.disabled = false;
}

function hideDetectionCard() {
  els.detectionCard.hidden = true;
  els.recommendBtn.disabled = true;
}

// ------------------------------------------------------------------
// Get Recommendations -> /recommend
// ------------------------------------------------------------------
async function onRecommend() {
  if (!capturedBlob) return;
  setResultsState("loading");
  els.recommendBtn.disabled = true;

  try {
    const form = new FormData();
    form.append("file", capturedBlob, "capture.jpg");
    const resp = await fetch("/recommend", { method: "POST", body: form });
    const data = await resp.json();

    if (!data.ok) {
      const friendly = data.message || (resp.status >= 500 ? MSG.serverError : MSG.recommendFailed);
      setResultsState("error", friendly);
      return;
    }

    renderRecommendations(data);
    setResultsState("content");
  } catch (err) {
    setResultsState("error", MSG.serverError);
  } finally {
    els.recommendBtn.disabled = false;
  }
}

function setResultsState(state, errorMessage) {
  els.emptyState.hidden = state !== "empty";
  els.loadingState.hidden = state !== "loading";
  els.errorState.hidden = state !== "error";
  els.recommendationContent.hidden = state !== "content";
  if (state === "error") els.errorText.textContent = errorMessage || MSG.recommendFailed;
}

function renderRecommendations(data) {
  const rec = data.recommendations || {};
  fillList(els.colorList, rec.recommended_colors, (text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  });
  fillList(els.outfitList, rec.recommended_outfits, (text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  });
  fillList(els.tipList, rec.styling_tips, (text) => {
    const li = document.createElement("li");
    li.textContent = text;
    return li;
  });
}

function fillList(ul, items, makeItem) {
  ul.innerHTML = "";
  (items || []).forEach((item) => ul.appendChild(makeItem(item)));
}

// ------------------------------------------------------------------
// Wire up events
// ------------------------------------------------------------------
els.startBtn.addEventListener("click", startCamera);
els.stopBtn.addEventListener("click", stopCamera);
els.captureBtn.addEventListener("click", onCapture);
els.retakeBtn.addEventListener("click", onRetake);
els.recommendBtn.addEventListener("click", onRecommend);
els.retryBtn.addEventListener("click", onRecommend);

window.addEventListener("beforeunload", () => {
  if (mediaStream) mediaStream.getTracks().forEach((t) => t.stop());
});
