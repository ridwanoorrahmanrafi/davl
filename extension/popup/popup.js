/**
 * Popup Script for Personalized CVD Assistant.
 * Handles profile configuration, health monitoring, and interactive Ishihara screening.
 * Features:
 * - Dynamic Ishihara test generation (randomized numerals, dynamic distractors, authentic CIE L*a*b* dot mosaics)
 * - 5-plate Quick Screening or 10-plate Comprehensive Thorough battery
 * - Re-rolling and shuffling of numerals & plates on demand
 * - Auto-detects CVD type and severity, then auto-configures personalization settings
 * - Syncs with FastAPI Restormer_RDB backend and updates content scripts
 */

// Global state for the Ishihara session
let session = null;            // {seed, plates, mode}
let responseTimes = [];        // ms per plate
let plateRenderStart = 0;      // timestamp when current plate rendered
let currentMode = "quick";     // 'quick' (5 plates) or 'comprehensive' (10 plates)

const API_BASE = "http://localhost:8000";
const $ = id => document.getElementById(id);

let plates = [];
let answers = [];
let busy = false;
let latestDiagnosis = null;
let redirectTimer = null;
let countdownInterval = null;
let plateTimerInterval = null;
let secondsRemaining = 12;

async function fetchApi(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 6000);

  try {
    const res = await fetch(url, {
      ...options,
      headers: { "Content-Type": "application/json", ...options.headers },
      signal: controller.signal,
    });
    clearTimeout(timeoutId);
    const data = await res.json();
    if (!res.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `Server error: ${res.status}`);
    }
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  }
}

async function checkHealth() {
  try {
    const data = await fetchApi("/health");
    if (data.trained_weights_loaded) {
      $("apiIndicator").className = "indicator online";
      $("apiStatusText").textContent = `${data.architecture} ready · ${data.device}`;
      return true;
    } else {
      $("apiIndicator").className = "indicator degraded";
      $("apiStatusText").textContent = "Backend ready (random weights active)";
      return true;
    }
  } catch (err) {
    $("apiIndicator").className = "indicator offline";
    $("apiStatusText").textContent = "Backend offline (start python run_server.py)";
    return false;
  }
}

async function loadSettings() {
  const state = await chrome.storage.local.get(["enabled", "condition_id", "severity", "scope", "auto_diagnosed"]);
  $("enableToggle").checked = state.enabled ?? false;
  $("toggleStatusLabel").textContent = state.enabled ? "ON" : "OFF";
  $("conditionSelect").value = String(state.condition_id !== undefined ? state.condition_id : 1);
  const sevPercent = Math.round((state.severity !== undefined ? state.severity : 0.85) * 100);
  $("severitySlider").value = sevPercent;
  $("severityValue").textContent = `${sevPercent}%`;

  const scope = state.scope || "images_only";
  if (scope === "images_only") {
    $("scopeImagesOnly").checked = true;
  } else {
    $("scopeFullPage").checked = true;
  }

  if (state.auto_diagnosed) {
    const condNames = ["Protanopia (Red-weak)", "Deuteranopia (Green-weak)", "Tritanopia (Blue-weak)"];
    const name = condNames[state.condition_id] || "Estimated profile";
    $("detectedBadge").textContent = `Estimated from test: ${name}`;
    $("detectedBadge").classList.remove("hidden");
  } else {
    $("detectedBadge").classList.add("hidden");
  }
}

async function saveSettings(autoDiagnosed = false) {
  const enabled = $("enableToggle").checked;
  const conditionId = Number($("conditionSelect").value);
  const severity = Number($("severitySlider").value) / 100.0;
  const scope = $("scopeImagesOnly").checked ? "images_only" : "full_page";

  const storageData = {
    enabled: enabled,
    condition_id: conditionId,
    severity: severity,
    scope: scope,
    auto_diagnosed: autoDiagnosed,
  };
  if (autoDiagnosed) {
    storageData.test_completed = true;
  }

  await chrome.storage.local.set(storageData);

  $("toggleStatusLabel").textContent = enabled ? "ON" : "OFF";

  // Notify active tab
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      await chrome.tabs.sendMessage(tab.id, { action: "SETTINGS_UPDATED" });
    }
  } catch (_) {
    // Content script might not be injected on special browser pages
  }
}

function switchTab(targetName) {
  if (redirectTimer) clearTimeout(redirectTimer);
  if (countdownInterval) clearInterval(countdownInterval);

  const isSettings = targetName === "Dashboard";
  $("tabDashboard").classList.toggle("active", isSettings);
  $("tabDiagnosis").classList.toggle("active", !isSettings);
  $("viewDashboard").classList.toggle("active-view", isSettings);
  $("viewDiagnosis").classList.toggle("active-view", !isSettings);

  // Widen popup when on Ishihara test tab
  document.body.classList.toggle("test-wide", !isSettings);
}

/** Render dynamic Ishihara plate previews in the intro card */
function renderIntroPreviews() {
  const container = $("introPlatesPreview");
  if (!container) return;
  container.replaceChildren();

  if (!session || session.mode !== currentMode) {
    session = window.generateDynamicSession ? window.generateDynamicSession(currentMode) : window.generateTestSession();
  }

  const sourcePlates = session?.plates || (typeof FALLBACK_PLATES !== "undefined" ? FALLBACK_PLATES : []);
  // Render up to 5 previews for clean visual row
  const previewPlates = sourcePlates.slice(0, 5);

  previewPlates.forEach(p => {
    const c = document.createElement("canvas");
    c.width = 80;
    c.height = 80;
    c.className = "intro-plate-thumb";
    c.title = `${p.title}: Numeral ${p.numeral}`;
    container.appendChild(c);
    try {
      const r = new IshiharaPlateRenderer(c);
      r.render(p.numeral, p.type, "medium");
    } catch (_) {}
  });
}

/** Update the dot step tracker to reflect the current plate index */
function updateDotTracker(currentIndex) {
  const track = $("plateDotProgress");
  if (!track) return;

  const total = plates.length || 5;
  if (track.children.length !== total) {
    track.replaceChildren();
    for (let i = 0; i < total; i++) {
      const span = document.createElement("span");
      span.className = "dot-step";
      span.dataset.step = String(i + 1);
      track.appendChild(span);
    }
  }

  const dots = track.querySelectorAll(".dot-step");
  dots.forEach((dot, i) => {
    dot.classList.remove("dot-active", "dot-done");
    if (i < currentIndex) {
      dot.classList.add("dot-done");
    } else if (i === currentIndex) {
      dot.classList.add("dot-active");
    }
  });
}

function goToPersonalizationWithHighlight() {
  if (redirectTimer) clearTimeout(redirectTimer);
  if (countdownInterval) clearInterval(countdownInterval);

  switchTab("Dashboard");

  // Pulse highlight on the automatically selected settings
  const select = $("conditionSelect");
  const badge = $("detectedBadge");
  const banner = $("autoAppliedBanner");

  if (select) select.classList.add("highlight-pulse");
  if (badge) badge.classList.add("highlight-pulse");
  if (banner) banner.classList.add("highlight-pulse");

  setTimeout(() => {
    if (select) select.classList.remove("highlight-pulse");
    if (badge) badge.classList.remove("highlight-pulse");
    if (banner) banner.classList.remove("highlight-pulse");
  }, 2500);
}

function selectPlateAnswer(userAnswer) {
  if (busy) return;
  busy = true;

  if (plateTimerInterval) clearInterval(plateTimerInterval);

  // Record response time for the just‑shown plate
  const elapsed = performance.now() - plateRenderStart;
  responseTimes.push(Math.round(elapsed));

  const currentIndex = answers.length;
  const plate = plates[currentIndex];
  answers.push({ plate_id: plate.id, user_answer: userAnswer });

  if (answers.length < plates.length) {
    renderCurrentPlate();
  } else {
    finishScreening();
  }
}

async function startScreeningTest() {
  if (redirectTimer) clearTimeout(redirectTimer);
  if (countdownInterval) clearInterval(countdownInterval);
  if (plateTimerInterval) clearInterval(plateTimerInterval);

  answers = [];
  responseTimes = [];
  latestDiagnosis = null;
  busy = false;

  $("testResultWrapper").classList.add("hidden");
  $("testIntroArea").classList.add("hidden");
  $("testContainer").classList.remove("hidden");

  // Generate a fresh dynamic session with randomized numerals
  session = window.generateDynamicSession ? window.generateDynamicSession(currentMode) : window.generateTestSession();
  plates = session.plates;

  // Initialize dot tracker with active plate count
  updateDotTracker(0);

  renderCurrentPlate();
}

function renderCurrentPlate() {
  busy = false;
  if (plateTimerInterval) clearInterval(plateTimerInterval);

  const currentIndex = answers.length;
  updateDotTracker(currentIndex);
  const plate = plates[currentIndex];

  // Header badges & progress
  const pct = Math.round((currentIndex / plates.length) * 100);
  if ($("plateNumberBadge")) $("plateNumberBadge").textContent = String(currentIndex + 1);
  if ($("plateQuestion")) $("plateQuestion").textContent = `Plate ${currentIndex + 1} of ${plates.length}`;
  if ($("progressPercentage")) $("progressPercentage").textContent = `${pct}%`;
  $("testProgress").style.width = `${pct}%`;
  $("platePrompt").textContent = plate.prompt || "What number do you see in the circle?";

  // Render authentic dynamic Ishihara dots on canvas
  plateRenderStart = performance.now();
  const canvas = $("plateCanvas");
  if (canvas) {
    const renderer = new IshiharaPlateRenderer(canvas);
    renderer.render(plate.numeral, plate.type, "medium");
  }

  // Active countdown timer (matching ishiharatest.com)
  secondsRemaining = plate.timeLimit || 12;
  const timerSecEl = $("testTimerSeconds");
  const timerPill = $("testTimerPill");
  if (timerSecEl) timerSecEl.textContent = String(secondsRemaining);
  if (timerPill) timerPill.classList.remove("timer-warning");

  plateTimerInterval = setInterval(() => {
    secondsRemaining--;
    if (timerSecEl) timerSecEl.textContent = String(secondsRemaining);
    if (secondsRemaining <= 3 && timerPill) {
      timerPill.classList.add("timer-warning");
    }
    if (secondsRemaining <= 0) {
      clearInterval(plateTimerInterval);
      selectPlateAnswer("Timeout / Unclear");
    }
  }, 1000);

  // Numerical options 2x2 grid (dynamically generated distractors)
  const optionsGrid = $("plateOptions");
  optionsGrid.replaceChildren();
  const currentOptions = plate.shuffledOptions || plate.options;
  for (const option of currentOptions) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "option-btn";
    btn.textContent = option;
    btn.onclick = () => selectPlateAnswer(option);
    optionsGrid.append(btn);
  }

  // Action buttons
  const cannotSeeBtn = $("btnCannotSee");
  if (cannotSeeBtn) {
    cannotSeeBtn.onclick = () => selectPlateAnswer("I cannot see any number");
  }
  const skipBtn = $("btnSkip");
  if (skipBtn) {
    skipBtn.onclick = () => selectPlateAnswer("Skip");
  }
}

async function finishScreening() {
  if (plateTimerInterval) clearInterval(plateTimerInterval);
  $("testContainer").classList.add("hidden");
  $("plateOptions").replaceChildren();
  $("testProgress").style.width = "100%";
  if ($("progressPercentage")) $("progressPercentage").textContent = "100%";

  const result = typeof evaluateScreeningLocally === "function"
    ? evaluateScreeningLocally(answers, plates)
    : {
        condition: "Evaluation Complete",
        condition_id: 1,
        severity: 0.85,
        errors: 0,
        correct: plates.length,
        accuracy: 100,
        total: plates.length,
        breakdown: []
      };

  latestDiagnosis = result;

  const condId = result.condition_id;
  const isDeficient = condId !== null && condId !== undefined;
  const total = result.total || plates.length || 5;
  const correct = result.correct !== undefined ? result.correct : (total - (result.errors || 0));
  const accuracy = result.accuracy !== undefined ? result.accuracy : Math.round((correct / total) * 100);
  const calibratedPct = result.severity ? Math.round(result.severity * 100) : (isDeficient ? 80 : 0);

  const conditionMap = {
    0: { name: "Protanopia (Red-weak)", class: "badge-protan", tag: "Red Deficiency Detected" },
    1: { name: "Deuteranopia (Green-weak)", class: "badge-deutan", tag: "Green Deficiency Detected" },
    2: { name: "Tritanopia (Blue-weak)", class: "badge-tritan", tag: "Blue Deficiency Detected" },
  };

  // Render Result Card
  const badgeCard = $("resultBadgeCard");
  badgeCard.className = "result-badge-card";

  if (isDeficient && conditionMap[condId]) {
    const info = conditionMap[condId];
    badgeCard.classList.add(info.class);
    $("diagnosisTitle").textContent = info.name;
    $("diagnosisTag").textContent = info.tag;
  } else {
    badgeCard.classList.add("badge-normal");
    $("diagnosisTitle").textContent = "Normal Color Vision";
    $("diagnosisTag").textContent = "No Deficiency Detected";
  }

  if ($("diagnosisAccuracy")) $("diagnosisAccuracy").textContent = `${accuracy}%`;
  if ($("diagnosisCorrect")) $("diagnosisCorrect").textContent = `${correct} / ${total}`;
  if ($("diagnosisSeverity")) $("diagnosisSeverity").textContent = isDeficient ? `${calibratedPct}%` : "0%";
  $("diagnosisDetail").textContent = result.message || "";

  // Render Itemized Breakdown Table
  const breakdownList = $("breakdownList");
  if (breakdownList && result.breakdown) {
    breakdownList.replaceChildren();
    result.breakdown.forEach((item, idx) => {
      const row = document.createElement("div");
      row.className = "breakdown-row";

      const left = document.createElement("div");
      left.className = "breakdown-left";

      const badge = document.createElement("span");
      badge.className = "breakdown-badge";
      badge.textContent = `P${idx + 1}`;

      const text = document.createElement("span");
      text.textContent = `Num: ${item.numeral} · Exp: ${item.expected} · You: ${item.userAnswer}`;
      left.append(badge, text);

      const status = document.createElement("span");
      status.className = item.isCorrect ? "breakdown-pass" : "breakdown-fail";
      status.textContent = item.isCorrect ? "✓ Pass" : "✗ CVD";

      row.append(left, status);
      breakdownList.append(row);
    });
  }

  $("testResultWrapper").classList.remove("hidden");

  // AUTOMATICALLY APPLY TO PERSONALIZATION PROFILE
  if (isDeficient) {
    const chosenName = conditionMap[condId].name;

    $("conditionSelect").value = String(condId);
    $("severitySlider").value = calibratedPct;
    $("severityValue").textContent = `${calibratedPct}%`;
    $("enableToggle").checked = true;
    $("toggleStatusLabel").textContent = "ON";

    $("detectedBadge").textContent = `Estimated from test: ${chosenName} (${calibratedPct}%)`;
    $("detectedBadge").classList.remove("hidden");

    $("autoBannerDesc").textContent = `${chosenName} (${calibratedPct}% severity) automatically configured based on your Ishihara test.`;
    $("autoAppliedBanner").classList.remove("hidden");

    await saveSettings(true);
  } else {
    $("detectedBadge").textContent = "Estimated from test: Normal Color Vision";
    $("detectedBadge").classList.remove("hidden");
    await chrome.storage.local.set({
      test_completed: true,
      auto_diagnosed: false,
    });
  }

  // Hide first-time onboarding banner once test has finished
  const firstTimeBanner = $("firstTimeWelcomeBanner");
  if (firstTimeBanner) firstTimeBanner.classList.add("hidden");

  // Refresh plate gallery with active session plates
  renderPlatesCatalog();

  // Auto-redirect countdown to Personalization view
  if (countdownInterval) clearInterval(countdownInterval);
  if (redirectTimer) clearTimeout(redirectTimer);

  let secondsLeft = 2;
  $("autoApplyStatusBox").classList.remove("hidden");
  $("autoApplyStatusText").textContent = isDeficient
    ? `✓ Automatically applied to Personalization! Switching tabs in ${secondsLeft}s...`
    : `✓ Test complete! Switching to Personalization in ${secondsLeft}s...`;

  countdownInterval = setInterval(() => {
    secondsLeft--;
    if (secondsLeft > 0) {
      $("autoApplyStatusText").textContent = isDeficient
        ? `✓ Automatically applied to Personalization! Switching tabs in ${secondsLeft}s...`
        : `✓ Test complete! Switching to Personalization in ${secondsLeft}s...`;
    } else {
      clearInterval(countdownInterval);
    }
  }, 1000);

  redirectTimer = setTimeout(() => {
    if (countdownInterval) clearInterval(countdownInterval);
    goToPersonalizationWithHighlight();
  }, 2200);
}

function renderPlatesCatalog() {
  const listEl = $("popupPlatesList");
  if (!listEl) return;

  const sourcePlates = (session && session.plates && session.plates.length > 0)
    ? session.plates
    : ((plates && plates.length > 0) ? plates : (typeof FALLBACK_PLATES !== "undefined" ? FALLBACK_PLATES : []));

  const titleEl = $("gallerySummaryTitle");
  if (titleEl) {
    titleEl.textContent = `🖼️ Browse Active Ishihara Test Plates (${sourcePlates.length} Plates)`;
  }

  listEl.replaceChildren();
  sourcePlates.forEach((p, idx) => {
    const card = document.createElement("div");
    card.className = "popup-plate-mini-card";

    const miniCanvas = document.createElement("canvas");
    miniCanvas.className = "popup-plate-mini-img";
    miniCanvas.width = 60;
    miniCanvas.height = 60;
    const miniRenderer = new IshiharaPlateRenderer(miniCanvas);
    miniRenderer.render(p.numeral || "12", p.type || "red-green", "medium");

    const info = document.createElement("div");
    info.className = "popup-plate-mini-info";

    const title = document.createElement("div");
    title.className = "popup-plate-mini-title";
    title.textContent = `Plate ${idx + 1} (${p.numeral}) · ${p.type}`;

    const normal = document.createElement("div");
    normal.className = "popup-plate-mini-normal";
    normal.textContent = `Normal: ${p.normal_answer}`;

    const cvd = document.createElement("div");
    cvd.className = "popup-plate-mini-cvd";
    cvd.textContent = `CVD: ${p.deutan_answer || p.protan_answer || p.tritan_answer || "Unclear"}`;

    info.append(title, normal, cvd);
    card.append(miniCanvas, info);
    listEl.append(card);
  });
}

// ── DOM INITIALIZATION & EVENT LISTENERS ────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  // Navigation tabs
  $("tabDashboard").onclick = () => switchTab("Dashboard");
  $("tabDiagnosis").onclick = () => switchTab("Diagnosis");

  // Personalization settings controls
  $("enableToggle").onchange = async () => {
    if ($("enableToggle").checked) {
      await checkHealth();
    }
    await saveSettings();
  };

  $("conditionSelect").onchange = () => {
    $("detectedBadge").classList.add("hidden");
    saveSettings(false);
  };

  $("scopeFullPage").onchange = () => saveSettings(false);
  $("scopeImagesOnly").onchange = () => saveSettings(false);

  $("severitySlider").oninput = () => {
    $("severityValue").textContent = `${$("severitySlider").value}%`;
  };
  $("severitySlider").onchange = () => saveSettings(false);

  $("applyNowBtn").onclick = async () => {
    await saveSettings();
    $("apiStatusText").textContent = "Settings saved & applied.";
  };

  $("resetProfileBtn").onclick = async () => {
    $("enableToggle").checked = false;
    $("conditionSelect").value = "1";
    $("severitySlider").value = "85";
    $("severityValue").textContent = "85%";
    $("scopeImagesOnly").checked = true;
    $("detectedBadge").classList.add("hidden");
    $("autoAppliedBanner").classList.add("hidden");
    await chrome.storage.local.remove(["test_completed", "auto_diagnosed"]);
    await saveSettings(false);
    $("apiStatusText").textContent = "Reset to default profile.";
  };

  if ($("dismissAutoBanner")) {
    $("dismissAutoBanner").onclick = () => {
      $("autoAppliedBanner").classList.add("hidden");
    };
  }

  // Mode Selection: Quick (5 plates) vs Comprehensive (10 plates)
  const btnModeQuick = $("btnModeQuick");
  const btnModeComp = $("btnModeComp");
  if (btnModeQuick && btnModeComp) {
    btnModeQuick.onclick = () => {
      currentMode = "quick";
      btnModeQuick.classList.add("active");
      btnModeComp.classList.remove("active");
      if ($("introStatPlates")) $("introStatPlates").textContent = "5";
      if ($("introStatDuration")) $("introStatDuration").textContent = "~60s";
      session = window.generateDynamicSession("quick");
      renderIntroPreviews();
      renderPlatesCatalog();
    };

    btnModeComp.onclick = () => {
      currentMode = "comprehensive";
      btnModeComp.classList.add("active");
      btnModeQuick.classList.remove("active");
      if ($("introStatPlates")) $("introStatPlates").textContent = "10";
      if ($("introStatDuration")) $("introStatDuration").textContent = "~120s";
      session = window.generateDynamicSession("comprehensive");
      renderIntroPreviews();
      renderPlatesCatalog();
    };
  }

  // Shuffle / Re-roll dynamic numbers
  const btnShuffleIntro = $("btnShuffleIntro");
  if (btnShuffleIntro) {
    btnShuffleIntro.onclick = () => {
      session = window.generateDynamicSession(currentMode);
      renderIntroPreviews();
      renderPlatesCatalog();
    };
  }

  const btnShuffleGallery = $("btnShuffleGallery");
  if (btnShuffleGallery) {
    btnShuffleGallery.onclick = () => {
      session = window.generateDynamicSession(currentMode);
      renderPlatesCatalog();
    };
  }

  // Ishihara Screening buttons
  if ($("startTestBtn")) {
    $("startTestBtn").onclick = startScreeningTest;
  }
  if ($("retryTestBtn")) {
    $("retryTestBtn").onclick = startScreeningTest;
  }
  if ($("goToPersonalizationBtn")) {
    $("goToPersonalizationBtn").onclick = goToPersonalizationWithHighlight;
  }

  // Keyboard shortcuts for answer selection
  document.addEventListener("keydown", e => {
    if (busy) return;
    const key = e.key;
    const currentPlate = plates[answers.length];
    if (!currentPlate) return;
    const opts = currentPlate.shuffledOptions || currentPlate.options;
    if (key >= "1" && key <= String(opts.length)) {
      const idx = Number(key) - 1;
      selectPlateAnswer(opts[idx]);
    } else if (key.toLowerCase() === "c") {
      selectPlateAnswer("I cannot see any number");
    } else if (key.toLowerCase() === "s") {
      selectPlateAnswer("Skip");
    }
  });

  // Check if current active page is a sensitive context (login, signup, transaction)
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id) {
      chrome.tabs.sendMessage(tab.id, { action: "CHECK_PAGE_SENSITIVE" }, response => {
        if (chrome.runtime.lastError) return;
        if (response && response.sensitive) {
          const notice = $("sensitivePageNotice");
          if (notice) {
            notice.classList.remove("hidden");
          }
        }
      });
    }
  } catch (_) {}

  // Initial load
  await loadSettings();
  await checkHealth();

  // Initialize dynamic session and render previews
  session = window.generateDynamicSession ? window.generateDynamicSession(currentMode) : window.generateTestSession();
  renderIntroPreviews();
  renderPlatesCatalog();

  // First-time open: automatically show test tab (let user click start)
  const testState = await chrome.storage.local.get(["test_completed", "auto_diagnosed"]);
  if (!testState.test_completed && !testState.auto_diagnosed) {
    const welcomeBanner = $("firstTimeWelcomeBanner");
    if (welcomeBanner) welcomeBanner.classList.remove("hidden");
    switchTab("Diagnosis");
  }
});
