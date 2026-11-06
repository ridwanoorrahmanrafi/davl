/**
 * Content Script for Personalized CVD Web Assistant.
 * Scans page images, filters eligible graphics, queues inference requests,
 * and performs non-destructive image replacements while preserving layout.
 */

(() => {
  "use strict";

  // Prevent double-injection
  if (window.__cvdAssistantInjected) return;
  window.__cvdAssistantInjected = true;

  const STATUS_ATTR = "data-cvd-status";
  const ORIG_SRC_ATTR = "data-cvd-orig-src";
  const ORIG_SRCSET_ATTR = "data-cvd-orig-srcset";
  const PROFILE_KEY_ATTR = "data-cvd-profile";

  const FILTER_SVG_ID = "cvd-assistant-svg-filter";
  const FILTER_ID = "cvd-assistant-matrix";
  const FILTER_STYLE_ID = "cvd-assistant-filter-style";

  let currentSettings = {
    enabled: false,
    condition_id: 1,
    severity: 0.85,
    scope: "images_only", // "images_only" (neural model on photos) or "full_page" (affine SVG on entire website)
  };

  let cachedFilterMatrix = null;
  let cachedFilterKey = "";

  /**
   * Injects/updates an SVG color matrix filter into the document and applies it to <html>.
   * This recolors the entire website (CSS backgrounds, text, UI elements, palettes, images).
   */
  function applyPageFilter(matrixValues) {
    if (!matrixValues || matrixValues.length !== 20) return;

    let svg = document.getElementById(FILTER_SVG_ID);
    if (!svg) {
      svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
      svg.id = FILTER_SVG_ID;
      svg.setAttribute("style", "position:fixed; top:0; left:0; width:0; height:0; pointer-events:none; z-index:-1000; opacity:0;");
      svg.setAttribute("aria-hidden", "true");
      svg.innerHTML = `
        <defs>
          <filter id="${FILTER_ID}" color-interpolation-filters="sRGB">
            <feColorMatrix type="matrix" values="${matrixValues.join(' ')}" />
          </filter>
        </defs>
      `;
      (document.body || document.documentElement).appendChild(svg);
    } else {
      const feMatrix = svg.querySelector("feColorMatrix");
      if (feMatrix) {
        feMatrix.setAttribute("values", matrixValues.join(" "));
      }
    }

    let style = document.getElementById(FILTER_STYLE_ID);
    if (!style) {
      style = document.createElement("style");
      style.id = FILTER_STYLE_ID;
      document.head.appendChild(style);
    }
    style.textContent = `
      html {
        filter: url(#${FILTER_ID}) !important;
      }
    `;
  }

  /**
   * Removes the full-page SVG filter and restores standard rendering.
   */
  function removePageFilter() {
    const style = document.getElementById(FILTER_STYLE_ID);
    if (style) style.remove();
    const svg = document.getElementById(FILTER_SVG_ID);
    if (svg) svg.remove();
  }

  /**
   * Synchronizes full-page SVG filter with the backend.
   */
  async function syncPageFilter() {
    if (!currentSettings.enabled || currentSettings.scope === "images_only") {
      removePageFilter();
      return;
    }

    const filterKey = `${currentSettings.condition_id}|${currentSettings.severity}`;
    if (cachedFilterMatrix && cachedFilterKey === filterKey) {
      applyPageFilter(cachedFilterMatrix);
      return;
    }

    try {
      const response = await chrome.runtime.sendMessage({
        action: "GET_FILTER",
        conditionId: currentSettings.condition_id,
        severity: currentSettings.severity,
      });

      if (response && response.success && response.data?.matrix) {
        cachedFilterMatrix = response.data.matrix;
        cachedFilterKey = filterKey;
        applyPageFilter(cachedFilterMatrix);
      }
    } catch (err) {
      console.warn("[CVD Assistant] Failed to retrieve whole-page filter matrix:", err);
    }
  }

  // Queue and concurrency controller (max 2 concurrent requests to avoid server overloading)
  const MAX_CONCURRENT = 2;
  let activeRequests = 0;
  const requestQueue = [];

  function enqueue(task) {
    requestQueue.push(task);
    processNext();
  }

  function processNext() {
    if (activeRequests >= MAX_CONCURRENT || requestQueue.length === 0) return;
    activeRequests++;
    const task = requestQueue.shift();
    task().finally(() => {
      activeRequests--;
      processNext();
    });
  }

  /**
   * Directly extracts lossless base64 representation of an image loaded in DOM.
   * Works on file://, http://, data:, blob: without network fetch failures.
   */
  function extractBase64FromImage(img) {
    try {
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      if (!w || !h || w < 16 || h < 16) return null;
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
      return canvas.toDataURL("image/png");
    } catch (err) {
      return null;
    }
  }

  /**
   * Evaluates whether an <img> is eligible for neural recoloring.
   */
  function isEligibleImage(img) {
    if (!img || !(img instanceof HTMLImageElement)) return false;

    // Check dimensions (skip tiny icons, tracking pixels)
    const rect = img.getBoundingClientRect();
    const w = rect.width || img.naturalWidth || img.width;
    const h = rect.height || img.naturalHeight || img.height;

    if (w < 32 || h < 32) return false;

    const src = img.currentSrc || img.src;
    if (!src) return false;

    // Skip extension assets and SVG data URLs
    if (src.startsWith("chrome-extension://") || src.startsWith("moz-extension://")) return false;
    if (src.startsWith("data:image/svg+xml")) return false;

    // SECURITY: Strictly skip any graphics inside sensitive login, signup, or payment forms
    const parentForm = img.closest("form");
    if (parentForm) {
      const hasSensitiveInput = parentForm.querySelector('input[type="password"], input[autocomplete*="cc-"], input[name*="card"], input[name*="cvv"], input[id*="card"], input[id*="cvv"]');
      if (hasSensitiveInput) return false;
    }

    // Skip already processed images with same profile
    const profileKey = `${currentSettings.condition_id}|${currentSettings.severity}`;
    if (img.getAttribute(STATUS_ATTR) === "processed" && img.getAttribute(PROFILE_KEY_ATTR) === profileKey) {
      return false;
    }

    if (img.getAttribute(STATUS_ATTR) === "pending") return false;

    return true;
  }

  /**
   * Processes a single eligible image.
   */
  async function processImage(img) {
    if (!currentSettings.enabled || !isEligibleImage(img)) return;

    const src = img.currentSrc || img.src;
    if (!src) return;

    const profileKey = `${currentSettings.condition_id}|${currentSettings.severity}`;

    // Save original source for non-destructive restore
    if (!img.hasAttribute(ORIG_SRC_ATTR)) {
      img.setAttribute(ORIG_SRC_ATTR, src);
    }

    // Preserve original layout dimensions
    const computedStyle = window.getComputedStyle(img);
    if (!img.style.width && computedStyle.width) {
      img.style.width = computedStyle.width;
    }
    if (!img.style.height && computedStyle.height) {
      img.style.height = computedStyle.height;
    }

    img.setAttribute(STATUS_ATTR, "pending");

    enqueue(async () => {
      if (!currentSettings.enabled) {
        img.removeAttribute(STATUS_ATTR);
        return;
      }

      try {
        const base64Data = extractBase64FromImage(img);
        const response = await chrome.runtime.sendMessage({
          action: "RECOLOR_IMAGE",
          src: img.getAttribute(ORIG_SRC_ATTR) || src,
          imageBase64: base64Data,
          conditionId: currentSettings.condition_id,
          severity: currentSettings.severity,
        });

        if (!currentSettings.enabled) return;

        if (response && response.success && response.recolored_base64) {
          // Backup and clear responsive srcset so browser reliably displays recolored src
          if (img.hasAttribute("srcset") && !img.hasAttribute(ORIG_SRCSET_ATTR)) {
            img.setAttribute(ORIG_SRCSET_ATTR, img.getAttribute("srcset"));
            img.removeAttribute("srcset");
          }
          // Replace source non-destructively
          img.src = response.recolored_base64;
          img.setAttribute(STATUS_ATTR, "processed");
          img.setAttribute(PROFILE_KEY_ATTR, profileKey);
        } else {
          // Failure: keep original image untouched
          img.setAttribute(STATUS_ATTR, "failed");
          console.warn("[CVD Assistant] Recoloring failed for image, kept original:", response?.error);
        }
      } catch (err) {
        img.setAttribute(STATUS_ATTR, "failed");
        console.warn("[CVD Assistant] Communication error, kept original:", err.message);
      }
    });
  }

  /**
   * Scans all images within a root element.
   */
  function scanImages(root = document.body) {
    if (!root || !currentSettings.enabled) return;
    const images = root.querySelectorAll ? root.querySelectorAll("img") : [];
    images.forEach(img => {
      if (img.complete && img.naturalWidth > 0) {
        processImage(img);
      } else {
        img.addEventListener("load", () => processImage(img), { once: true });
      }
    });
  }

  /**
   * Restores all images back to their original sources when disabled.
   */
  function restoreOriginalImages() {
    requestQueue.length = 0; // Clear pending queue
    const processedImages = document.querySelectorAll(`img[${STATUS_ATTR}]`);
    processedImages.forEach(img => {
      const orig = img.getAttribute(ORIG_SRC_ATTR);
      if (orig && img.src !== orig) {
        img.src = orig;
      }
      const origSrcset = img.getAttribute(ORIG_SRCSET_ATTR);
      if (origSrcset) {
        img.setAttribute("srcset", origSrcset);
        img.removeAttribute(ORIG_SRCSET_ATTR);
      }
      img.removeAttribute(STATUS_ATTR);
      img.removeAttribute(PROFILE_KEY_ATTR);
    });
  }

  /**
   * Dynamic content monitoring via MutationObserver.
   */
  const observer = new MutationObserver(mutations => {
    if (!currentSettings.enabled) return;

    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) {
          if (node.tagName === "IMG") {
            processImage(node);
          } else if (node.querySelectorAll) {
            scanImages(node);
          }
        }
      }
    }
  });

  const BANNER_ID = "cvd-security-warning-banner";
  let securityBannerDismissed = false;

  /**
   * Detects if the current webpage or user interaction involves sensitive activity:
   * Login, Sign-up, or Financial Transactions (Checkout, Card inputs, Banking).
   */
  function detectSensitiveContext() {
    // 1. Check for password input fields (Login / Signup)
    const hasPassword = document.querySelector('input[type="password"]');
    if (hasPassword) {
      return { isSensitive: true, type: "login_signup", reason: "Account Login / Sign-up" };
    }

    // 2. Check for credit card or payment input fields (Transactions)
    const ccSelectors = [
      'input[autocomplete*="cc-"]',
      'input[name*="card"]',
      'input[id*="card"]',
      'input[name*="cvv"]',
      'input[id*="cvv"]',
      'input[name*="payment"]',
      'input[id*="payment"]',
      'input[name*="exp"]',
      'input[placeholder*="Card number" i]',
      'input[placeholder*="CVV" i]'
    ].join(", ");

    const hasPaymentInput = document.querySelector(ccSelectors);
    if (hasPaymentInput) {
      return { isSensitive: true, type: "transaction", reason: "Financial Payment & Transaction" };
    }

    // 3. Check URL path patterns
    const url = (window.location.pathname + window.location.search).toLowerCase();
    const sensitiveUrlPatterns = [
      { pattern: /\b(login|signin|sign-in)\b/, reason: "Account Login" },
      { pattern: /\b(signup|sign-up|register)\b/, reason: "Account Sign-Up" },
      { pattern: /\b(checkout|payment|billing|pay|cart|transaction|bank|banking|wallet)\b/, reason: "Financial Transaction & Checkout" }
    ];

    for (const item of sensitiveUrlPatterns) {
      if (item.pattern.test(url)) {
        return { isSensitive: true, type: "url_match", reason: item.reason };
      }
    }

    // 4. Check for form actions / attributes
    const sensitiveForm = document.querySelector('form[action*="login" i], form[action*="signin" i], form[action*="signup" i], form[action*="checkout" i], form[action*="payment" i]');
    if (sensitiveForm) {
      return { isSensitive: true, type: "form_action", reason: "Sensitive Form Submission" };
    }

    return { isSensitive: false, type: "", reason: "" };
  }

  /**
   * Renders a prominent privacy & security alert banner on sensitive pages.
   */
  function showSecurityWarningBanner(reason) {
    if (securityBannerDismissed) return;
    let banner = document.getElementById(BANNER_ID);

    if (!banner) {
      banner = document.createElement("div");
      banner.id = BANNER_ID;
      banner.setAttribute("role", "alert");
      banner.setAttribute("aria-live", "assertive");
      banner.style.cssText = `
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        right: 0 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        background: #161b22 !important;
        background: linear-gradient(180deg, #1c2128 0%, #161b22 100%) !important;
        border-bottom: 2px solid #d29922 !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5) !important;
        color: #e6edf3 !important;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        padding: 12px 18px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        gap: 14px !important;
        z-index: 2147483647 !important;
        font-size: 13px !important;
        line-height: 1.4 !important;
      `;
      (document.body || document.documentElement).appendChild(banner);
    }

    if (currentSettings.enabled) {
      banner.innerHTML = `
        <div style="display:flex; align-items:flex-start; gap:12px; flex:1;">
          <span style="font-size:20px; line-height:1; filter: drop-shadow(0 2px 4px rgba(210,153,34,0.4));">🛡️</span>
          <div>
            <div style="display:flex; align-items:center; gap:8px;">
              <strong style="color:#f0b72f; font-size:13px; text-transform:uppercase; letter-spacing:0.5px;">Security & Privacy Notice</strong>
              <span style="background:rgba(210,153,34,0.2); color:#f0b72f; border:1px solid rgba(210,153,34,0.4); border-radius:10px; font-size:10px; font-weight:700; padding:1px 6px;">${reason}</span>
            </div>
            <p style="margin:4px 0 0; color:#c9d1d9; font-size:12px;">
              Warning: You are on a sensitive page (${reason}). Active extensions interact with website data. For complete privacy of your passwords and financial information, <strong>we advise turning off the extension</strong> during this action. You can turn it back on once complete.
            </p>
          </div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
          <button id="cvd-turn-off-btn" style="background:#d29922; color:#12141a; border:none; padding:8px 14px; border-radius:6px; font-weight:700; font-size:12px; cursor:pointer; transition:background 0.2s;">
            Turn Off Extension
          </button>
          <button id="cvd-dismiss-btn" style="background:#21262d; color:#8b949e; border:1px solid #30363d; padding:8px 12px; border-radius:6px; font-weight:600; font-size:12px; cursor:pointer;">
            Dismiss
          </button>
        </div>
      `;

      banner.querySelector("#cvd-turn-off-btn").onclick = async () => {
        await chrome.storage.local.set({ enabled: false });
        showTurnedOffBanner(reason);
      };

      banner.querySelector("#cvd-dismiss-btn").onclick = () => {
        securityBannerDismissed = true;
        banner.remove();
      };
    } else {
      showTurnedOffBanner(reason);
    }
  }

  function showTurnedOffBanner(reason) {
    let banner = document.getElementById(BANNER_ID);
    if (!banner) return;

    banner.innerHTML = `
      <div style="display:flex; align-items:center; gap:12px; flex:1;">
        <span style="font-size:20px; color:#3fb950;">✓</span>
        <div>
          <strong style="color:#3fb950; font-size:13px;">CVD Assistant is OFF — Sensitive Action Isolated</strong>
          <p style="margin:2px 0 0; color:#8b949e; font-size:12px;">
            Your credentials and transactions are isolated. After completing your ${reason.toLowerCase()}, you can re-enable:
          </p>
        </div>
      </div>
      <div style="display:flex; align-items:center; gap:8px; flex-shrink:0;">
        <button id="cvd-turn-on-btn" style="background:#238636; color:#ffffff; border:none; padding:8px 14px; border-radius:6px; font-weight:700; font-size:12px; cursor:pointer;">
          Turn Back On
        </button>
        <button id="cvd-close-banner-btn" style="background:transparent; color:#8b949e; border:none; font-size:16px; cursor:pointer; padding:0 6px;">
          &times;
        </button>
      </div>
    `;

    banner.querySelector("#cvd-turn-on-btn").onclick = async () => {
      await chrome.storage.local.set({ enabled: true });
      banner.remove();
    };

    banner.querySelector("#cvd-close-banner-btn").onclick = () => {
      securityBannerDismissed = true;
      banner.remove();
    };
  }

  function checkSensitiveAndWarn() {
    if (!currentSettings.enabled) return;
    const check = detectSensitiveContext();
    if (check.isSensitive) {
      showSecurityWarningBanner(check.reason);
    }
  }

  // Trigger security check when user focuses on an input element
  document.addEventListener("focusin", (e) => {
    if (currentSettings.enabled && e.target && e.target.tagName === "INPUT") {
      const type = (e.target.type || "").toLowerCase();
      const autocomplete = (e.target.autocomplete || "").toLowerCase();
      const name = (e.target.name || "").toLowerCase();
      const id = (e.target.id || "").toLowerCase();

      if (
        type === "password" ||
        autocomplete.includes("password") ||
        autocomplete.includes("cc-") ||
        name.includes("card") ||
        id.includes("card") ||
        name.includes("cvv") ||
        id.includes("cvv")
      ) {
        showSecurityWarningBanner(type === "password" || autocomplete.includes("password") ? "Account Login / Sign-up" : "Payment & Financial Transaction");
      }
    }
  });

  async function updateSettings() {
    const prefs = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope"]);
    const wasEnabled = currentSettings.enabled;
    const prevScope = currentSettings.scope;
    const prevKey = `${currentSettings.condition_id}|${currentSettings.severity}|${currentSettings.scope}`;

    currentSettings.enabled = prefs.enabled ?? false;
    currentSettings.condition_id = prefs.condition_id ?? 1;
    currentSettings.severity = prefs.severity ?? 0.85;
    currentSettings.scope = prefs.scope ?? "images_only"; // Default to neural model image recoloring

    const newKey = `${currentSettings.condition_id}|${currentSettings.severity}|${currentSettings.scope}`;

    if (!currentSettings.enabled) {
      observer.disconnect();
      removePageFilter();
      restoreOriginalImages();

      // If warning banner is present, update to "turned off" safe state
      const banner = document.getElementById(BANNER_ID);
      if (banner) {
        showTurnedOffBanner("transaction / login");
      }
    } else {
      // Check for sensitive page context and show warning if needed
      checkSensitiveAndWarn();

      if (currentSettings.scope === "full_page") {
        await syncPageFilter();
      } else {
        removePageFilter();
      }

      // ALWAYS execute neural Restormer pixelwise model on eligible images
      observer.observe(document.documentElement, { childList: true, subtree: true });
      if (!wasEnabled || prevKey !== newKey) {
        document.querySelectorAll(`img[${STATUS_ATTR}]`).forEach(img => {
          img.removeAttribute(STATUS_ATTR);
          img.removeAttribute(PROFILE_KEY_ATTR);
        });
        scanImages();
      }
    }
  }

  // Listen for storage changes
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "local" && ("enabled" in changes || "condition_id" in changes || "severity" in changes || "scope" in changes)) {
      updateSettings();
    }
  });

  // Listen for explicit ping messages from popup
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "SETTINGS_UPDATED") {
      updateSettings().then(() => sendResponse({ ok: true }));
      return true;
    }

    if (message.action === "CHECK_PAGE_SENSITIVE") {
      const check = detectSensitiveContext();
      sendResponse({ sensitive: check.isSensitive, reason: check.reason });
      return true;
    }
  });

  // Listen for Ishihara test completion from test_page.html
  window.addEventListener("message", (event) => {
    if (event.data && event.data.source === "CVD_TEST_PAGE" && event.data.action === "SET_PROFILE_FROM_TEST") {
      const isNormal = Boolean(event.data.is_normal || event.data.condition_id === null || event.data.condition_id === undefined);
      if (isNormal) {
        chrome.storage.local.set({
          test_completed: true,
          auto_diagnosed: false,
        }).then(() => {
          updateSettings();
        });
      } else {
        const condId = Number(event.data.condition_id);
        const sev = Number(event.data.severity) || 0.85;

        chrome.storage.local.set({
          enabled: true,
          condition_id: condId,
          severity: sev,
          auto_diagnosed: true,
          test_completed: true,
        }).then(() => {
          updateSettings();
        });
      }
    }
  });

  // Initialize
  updateSettings();
  setTimeout(checkSensitiveAndWarn, 600);
})();
