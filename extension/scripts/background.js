/**
 * Manifest V3 Service Worker for Personalized CVD Browser Assistant.
 * Handles:
 * - Cross-origin image fetching & Base64 conversion
 * - Proxying inference requests to FastAPI /api/v1/recolor
 * - In-flight request deduplication & error handling
 * - Storage initialization
 */

const API_BASE = "http://localhost:8000";

// Initialize default storage on install
chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope"]);
  const updates = {};
  if (current.enabled === undefined) updates.enabled = false;
  if (current.condition_id === undefined) updates.condition_id = 1; // Default Deuteranopia
  if (current.severity === undefined) updates.severity = 0.85;
  if (current.scope === undefined) updates.scope = "images_only"; // Default to neural model image recoloring
  if (Object.keys(updates).length > 0) {
    await chrome.storage.local.set(updates);
  }
});

// Cache for in-flight requests to deduplicate concurrent requests for identical images
const inFlightRequests = new Map();

/**
 * Converts a Blob to a Base64 data URL string.
 */
function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

/**
 * Fetches an image URL and converts it to a Base64 data URL.
 */
async function fetchImageAsBase64(imageUrl) {
  if (imageUrl.startsWith("data:image/")) {
    return imageUrl;
  }

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 12000);

  try {
    const response = await fetch(imageUrl, {
      signal: controller.signal,
      cache: "force-cache",
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Failed to fetch image: HTTP ${response.status}`);
    }

    const blob = await response.blob();
    return await blobToBase64(blob);
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Sends image data URL to the FastAPI recolor endpoint.
 */
async function sendRecolorRequest(imageBase64, conditionId, severity) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 20000);

  try {
    const response = await fetch(`${API_BASE}/api/v1/recolor`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        image_base64: imageBase64,
        condition_id: conditionId,
        severity: severity,
        return_format: "png", // Lossless 24-bit sRGB PNG (preserves exact PyTorch model output)
      }),
    });
    clearTimeout(timeoutId);

    const data = await response.json();
    if (!response.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `API Error ${response.status}`);
    }

    return data;
  } catch (error) {
    clearTimeout(timeoutId);
    throw error;
  }
}

/**
 * Computes SHA-256 hex string for cryptographic data hashing.
 */
async function sha256Hex(text) {
  const enc = new TextEncoder().encode(text);
  const hashBuf = await crypto.subtle.digest("SHA-256", enc);
  const hashArr = Array.from(new Uint8Array(hashBuf));
  return hashArr.map(b => b.toString(16).padStart(2, "0")).join("");
}

// Handle messages from content script & popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "RECOLOR_IMAGE") {
    const { src, imageBase64, conditionId, severity } = message;

    const task = (async () => {
      try {
        const base64Data = (imageBase64 && imageBase64.startsWith("data:image/"))
          ? imageBase64
          : await fetchImageAsBase64(src);
        // Hash payload with SHA-256 so raw URLs or screens are not tracked
        const payloadHash = await sha256Hex(base64Data);
        const cacheKey = `${payloadHash}|${conditionId}|${severity}`;

        if (inFlightRequests.has(cacheKey)) {
          return await inFlightRequests.get(cacheKey);
        }

        const recolorPromise = (async () => {
          const recolorResult = await sendRecolorRequest(base64Data, conditionId, severity);
          return {
            success: true,
            recolored_base64: recolorResult.recolored_base64,
            inference_time_ms: recolorResult.inference_time_ms,
            cached: recolorResult.cached,
            data_hash: recolorResult.data_hash || payloadHash,
          };
        })();

        inFlightRequests.set(cacheKey, recolorPromise);
        recolorPromise.finally(() => inFlightRequests.delete(cacheKey));
        return await recolorPromise;
      } catch (error) {
        return {
          success: false,
          error: error.message || "Recoloring failed",
        };
      }
    })();

    task.then(sendResponse);
    return true; // Keep channel open for async response
  }

  if (message.action === "SET_ENABLED") {
    chrome.storage.local.set({ enabled: Boolean(message.enabled) }).then(() => {
      sendResponse({ success: true, enabled: Boolean(message.enabled) });
    });
    return true;
  }

  if (message.action === "CHECK_SECURITY") {
    sendResponse({
      success: true,
      hashing_algorithm: "SHA-256",
      screen_capture_allowed: false,
      ephemeral_memory_only: true,
    });
    return true;
  }

  if (message.action === "CHECK_HEALTH") {
    fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  if (message.action === "GET_FILTER") {
    fetch(`${API_BASE}/api/v1/filter`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: AbortSignal.timeout(10000),
      body: JSON.stringify({ condition_id: message.conditionId, severity: message.severity }),
    })
      .then(res => res.json())
      .then(data => sendResponse({ success: true, data }))
      .catch(err => sendResponse({ success: false, error: err.message }));
    return true;
  }

  return false;
});
