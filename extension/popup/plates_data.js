/**
 * Authentic Dynamic Ishihara Plate Engine & Palettes from ishiharatest.com.
 * Uses exact CIE L*a*b* to sRGB conversion, multi-scale Poisson dot layout,
 * dynamic numeral generation, and clinical diagnostic pairings.
 */

// Seeded pseudo‑random generator (Mulberry32) – deterministic for reproducible sessions
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6d2b79f5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = (t + Math.imul(t ^ t >>> 7, 61 | t)) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// Shuffle an array in‑place using RNG (Fisher‑Yates)
function shuffleArray(arr, rand = Math.random) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

// ── CLINICAL NUMERAL POOLS FOR DYNAMIC PLATES ──────────────────────────────
const CLINICAL_NUMERALS = {
  control: ["12", "16", "29", "8", "5", "74", "45", "97", "26", "35"],
  "red-green": ["74", "29", "57", "45", "73", "16", "35", "96", "5", "8", "3", "7"],
  protan: ["6", "8", "9", "5", "3", "2", "26", "42", "35", "96"],
  deutan: ["2", "3", "5", "7", "8", "6", "21", "74", "57", "97"],
  "blue-yellow": ["15", "13", "16", "75", "73", "29", "8", "7", "5"]
};

/**
 * Generate 4 realistic distractors for any 1-digit or 2-digit numeral.
 */
function generateDynamicDistractors(numeral, rand = Math.random) {
  const isSingle = numeral.length === 1;
  const num = parseInt(numeral, 10);
  const options = new Set([numeral]);

  if (isSingle) {
    const candidates = ["2", "3", "5", "6", "7", "8", "9", "0"];
    let attempts = 0;
    while (options.size < 4 && attempts < 50) {
      attempts++;
      const pick = candidates[Math.floor(rand() * candidates.length)];
      options.add(pick);
    }
  } else {
    const d1 = numeral[0];
    const d2 = numeral[1];
    
    // 1. Swapped digits
    if (d1 !== d2) options.add(d2 + d1);
    
    // 2. Visually confusing digits
    const similar = {
      "1": ["7", "4"], "2": ["3", "7"], "3": ["8", "5"], "4": ["1", "9"],
      "5": ["6", "3"], "6": ["5", "8"], "7": ["1", "2"], "8": ["3", "9"],
      "9": ["0", "6"], "0": ["8", "9"]
    };
    if (similar[d1]) options.add(similar[d1][0] + d2);
    if (similar[d2]) options.add(d1 + similar[d2][0]);

    // 3. Mathematical proximity
    [-10, 10, -1, 1, -2, 2, -11, 11].forEach(delta => {
      const val = num + delta;
      if (val >= 10 && val <= 99) options.add(String(val));
    });

    // 4. Random fallback
    let attempts = 0;
    while (options.size < 4 && attempts < 50) {
      attempts++;
      const rnd = String(Math.floor(rand() * 90) + 10);
      options.add(rnd);
    }
  }

  const arr = Array.from(options).slice(0, 4);
  shuffleArray(arr, rand);
  return arr;
}

/**
 * Generate a single dynamic plate object.
 */
function generateDynamicPlate(type, specificNumeral = null, rand = Math.random) {
  const pool = CLINICAL_NUMERALS[type] || CLINICAL_NUMERALS["red-green"];
  const numeral = specificNumeral || pool[Math.floor(rand() * pool.length)];
  const options = generateDynamicDistractors(numeral, rand);

  let title = "Ishihara Screening Plate";
  let prompt = "What number do you see in the circle?";
  let protanAns = "I cannot see any number";
  let deutanAns = "I cannot see any number";
  let tritanAns = numeral;

  if (type === "control") {
    title = "Plate · Demonstration (Control)";
    protanAns = numeral;
    deutanAns = numeral;
    tritanAns = numeral;
  } else if (type === "red-green") {
    title = "Plate · Red-Green Screening";
    protanAns = "I cannot see any number";
    deutanAns = "I cannot see any number";
    tritanAns = numeral;
  } else if (type === "protan") {
    title = "Plate · Protan (Red-Weak) Diagnostic";
    protanAns = "I cannot see any number";
    deutanAns = numeral;
    tritanAns = numeral;
  } else if (type === "deutan") {
    title = "Plate · Deutan (Green-Weak) Diagnostic";
    protanAns = numeral;
    deutanAns = "I cannot see any number";
    tritanAns = numeral;
  } else if (type === "blue-yellow" || type === "tritan") {
    title = "Plate · Tritan (Blue-Yellow) Diagnostic";
    protanAns = numeral;
    deutanAns = numeral;
    tritanAns = "I cannot see any number";
  }

  return {
    id: `dyn-plate-${type}-${numeral}-${Date.now().toString(36)}-${Math.floor(rand() * 1000)}`,
    numeral,
    type,
    title,
    prompt,
    timeLimit: 12,
    options,
    shuffledOptions: options.slice(),
    normal_answer: numeral,
    protan_answer: protanAns,
    deutan_answer: deutanAns,
    tritan_answer: tritanAns
  };
}

/**
 * Generate a fully dynamic test session with randomized numerals and plates.
 * Supports 'quick' (5 plates) or 'comprehensive' (10 plates).
 */
function generateDynamicSession(mode = "quick", seed = Date.now()) {
  const rand = mulberry32(seed);

  const configs = mode === "comprehensive"
    ? [
        { type: "control", title: "Plate 1 · Demonstration (Control)" },
        { type: "red-green", title: "Plate 2 · Red-Green Screening (Vanishing)" },
        { type: "protan", title: "Plate 3 · Protan Diagnostic (Red-Weak)" },
        { type: "deutan", title: "Plate 4 · Deutan Diagnostic (Green-Weak)" },
        { type: "blue-yellow", title: "Plate 5 · Tritan Diagnostic (Blue-Yellow)" },
        { type: "red-green", title: "Plate 6 · Red-Green High-Sensitivity" },
        { type: "protan", title: "Plate 7 · Protan Secondary Verification" },
        { type: "deutan", title: "Plate 8 · Deutan Secondary Verification" },
        { type: "red-green", title: "Plate 9 · Red-Green Precision Vanishing" },
        { type: "blue-yellow", title: "Plate 10 · Tritan Secondary Verification" }
      ]
    : [
        { type: "control", title: "Plate 1 · Demonstration (Control)" },
        { type: "red-green", title: "Plate 2 · Red-Green Screening" },
        { type: "protan", title: "Plate 3 · Protan (Red-Weak) Diagnostic" },
        { type: "deutan", title: "Plate 4 · Deutan (Green-Weak) Diagnostic" },
        { type: "blue-yellow", title: "Plate 5 · Tritan (Blue-Yellow) Diagnostic" }
      ];

  const plates = configs.map((cfg, idx) => {
    const plate = generateDynamicPlate(cfg.type, null, rand);
    plate.index = idx + 1;
    plate.title = cfg.title;
    return plate;
  });

  return { seed, plates, mode };
}

/**
 * Standard generateTestSession function - now dynamically generated by default!
 */
function generateTestSession(seed = Date.now(), mode = "quick") {
  return generateDynamicSession(mode, seed);
}

/** Feature‑vector schema for test result */
function evaluateIshiharaTest(sessionPlates, userAnswers, responseTimes) {
  const result = {
    plates: [],
    totalCorrect: 0,
    accuracy: 0,
    estimatedType: 3, // 0=protan,1=deutan,2=tritan,3=normal
    severity: 0,
    responseTimes: [],
  };

  const typeMistakes = { protan: 0, deutan: 0, tritan: 0 };
  sessionPlates.forEach((p, idx) => {
    const answer = userAnswers[idx];
    const correctAnswer = p.normal_answer;
    const isCorrect = answer === correctAnswer;
    result.plates.push({
      id: p.id,
      answer,
      correct: isCorrect,
      responseTime: responseTimes[idx] || null,
      expected: correctAnswer,
    });
    if (isCorrect) {
      result.totalCorrect++;
    } else {
      if (answer === p.protan_answer) typeMistakes.protan += 2;
      if (answer === p.deutan_answer) typeMistakes.deutan += 2;
      if (answer === p.tritan_answer) typeMistakes.tritan += 2;

      if (p.type === "protan") typeMistakes.protan += 3;
      else if (p.type === "deutan") typeMistakes.deutan += 3;
      else if (p.type === "blue-yellow" || p.type === "tritan") typeMistakes.tritan += 4;
      else if (p.type === "red-green") {
        typeMistakes.protan += 1;
        typeMistakes.deutan += 1;
      }
    }
    result.responseTimes.push(responseTimes[idx] || null);
  });

  result.accuracy = Math.round((result.totalCorrect / sessionPlates.length) * 100);

  const maxMistake = Math.max(typeMistakes.protan, typeMistakes.deutan, typeMistakes.tritan);
  if (maxMistake > 0 && result.totalCorrect < sessionPlates.length) {
    if (maxMistake === typeMistakes.protan) result.estimatedType = 0;
    else if (maxMistake === typeMistakes.deutan) result.estimatedType = 1;
    else result.estimatedType = 2;
    result.severity = Math.round(Math.min(1.0, Math.max(0.5, 1 - result.totalCorrect / sessionPlates.length)) * 100) / 100;
  } else {
    result.estimatedType = 3; // Normal vision
    result.severity = 0;
  }

  return result;
}

// ── COLOR CONVERSION & CIE L*a*b* ENGINE ────────────────────────────────────
const ISHIHARA_WEIGHTS = { easy: 26, medium: 17, hard: 10 };
const KE = 0.03, SE = 0.04, JE = 62, TE = 10, NE = 28, AE = 6, ZE = 3;
const PE = 0.95047, IE = 1.0, CE = 1.08883;

function labF(e) {
  return e * e * e > 0.008856 ? e * e * e : (e - 16 / 116) / 7.787;
}

function srgbTransfer(e) {
  const t = e <= 0.0031308 ? 12.92 * e : 1.055 * Math.pow(e, 0.4166666666666667) - 0.055;
  return Math.min(1, Math.max(0, t));
}

function labToHex(L, a, b) {
  const i = (L + 16) / 116;
  const r = i + a / 500;
  const s = i - b / 200;
  const X = PE * labF(r);
  const Y = IE * labF(i);
  const Z = CE * labF(s);
  const R = 3.2406 * X - 1.5372 * Y - 0.4986 * Z;
  const G = -0.9689 * X + 1.8758 * Y + 0.0415 * Z;
  const B = 0.0557 * X - 0.204 * Y + 1.057 * Z;
  const toHex = p => Math.round(srgbTransfer(p) * 255).toString(16).padStart(2, "0");
  return `#${toHex(R)}${toHex(G)}${toHex(B)}`;
}

/**
 * Generates dot color with natural organic stochastic variation.
 */
function getIshiharaColor(difficulty, target, type = "red-green", rand = Math.random) {
  if (type === "control") {
    // Demonstration plate (High contrast orange-red on cyan-teal)
    if (target === "figure") {
      return `hsl(${14 + rand() * 16}, ${82 + rand() * 15}%, ${50 + rand() * 8}%)`;
    } else {
      return `hsl(${165 + rand() * 25}, ${65 + rand() * 18}%, ${44 + rand() * 12}%)`;
    }
  }

  if (type === "tritan" || type === "blue-yellow") {
    // Tritan plate (Violet / magenta figure on golden-yellow background)
    if (target === "figure") {
      return `hsl(${270 + rand() * 25}, ${62 + rand() * 18}%, ${58 + rand() * 10}%)`;
    } else {
      return `hsl(${46 + rand() * 16}, ${75 + rand() * 18}%, ${54 + rand() * 10}%)`;
    }
  }

  if (type === "protan") {
    // Protan diagnostic (Dusty pink/crimson figure on sage/olive background)
    if (target === "figure") {
      return `hsl(${348 + rand() * 20}, ${58 + rand() * 16}%, ${54 + rand() * 8}%)`;
    } else {
      return `hsl(${96 + rand() * 22}, ${34 + rand() * 16}%, ${50 + rand() * 8}%)`;
    }
  }

  if (type === "deutan") {
    // Deutan diagnostic (Soft emerald/grass green figure on warm sand/ochre background)
    if (target === "figure") {
      return `hsl(${138 + rand() * 20}, ${50 + rand() * 18}%, ${50 + rand() * 8}%)`;
    } else {
      return `hsl(${32 + rand() * 18}, ${56 + rand() * 18}%, ${52 + rand() * 8}%)`;
    }
  }

  // Clinical Red-Green isoluminant CIE Lab palette (figure warm peach, background sage/olive)
  const i = ISHIHARA_WEIGHTS[difficulty] || 17;
  const r = target === "figure" ? 1 : -1;
  const s = JE + r * KE * i + (rand() * 2 - 1) * TE;
  const o = NE + r * SE * i + (rand() * 2 - 1) * AE;
  const l = r * i + (rand() * 2 - 1) * ZE;
  return labToHex(s, l, o);
}

/**
 * High-performance HTML5 Canvas dynamic Ishihara dot-matrix renderer.
 * Features:
 * - Spatial grid acceleration for sub-5ms dot placement
 * - Multi-scale natural dot radii (small, medium, large)
 * - Circular paper disc clipping with warm authentic tone
 * - Drop shadows for realistic printed lithograph depth
 */
class IshiharaPlateRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.width = canvas.width;
    this.height = canvas.height;
    this.dots = [];
    this.cellSize = 12;
    this.cols = Math.ceil(this.width / this.cellSize);
    this.rows = Math.ceil(this.height / this.cellSize);
    this.grid = [];
  }

  initGrid() {
    this.grid = Array.from({ length: this.cols * this.rows }, () => []);
  }

  canPlaceDotGrid(x, y, r) {
    const cx = this.width / 2, cy = this.height / 2, maxR = this.width / 2 - 3;
    if (Math.hypot(x - cx, y - cy) + r > maxR) return false;

    const gx = Math.floor(x / this.cellSize);
    const gy = Math.floor(y / this.cellSize);
    const minGx = Math.max(0, gx - 1), maxGx = Math.min(this.cols - 1, gx + 1);
    const minGy = Math.max(0, gy - 1), maxGy = Math.min(this.rows - 1, gy + 1);

    for (let c = minGx; c <= maxGx; c++) {
      for (let rIdx = minGy; rIdx <= maxGy; rIdx++) {
        const cellDots = this.grid[rIdx * this.cols + c];
        for (let i = 0; i < cellDots.length; i++) {
          const d = cellDots[i];
          const minDist = r + d.r + 1.2;
          const dx = x - d.x;
          const dy = y - d.y;
          if (Math.abs(dx) >= minDist || Math.abs(dy) >= minDist) continue;
          if (dx * dx + dy * dy < minDist * minDist) return false;
        }
      }
    }
    return true;
  }

  addDot(dot) {
    this.dots.push(dot);
    const gx = Math.floor(dot.x / this.cellSize);
    const gy = Math.floor(dot.y / this.cellSize);
    if (gx >= 0 && gx < this.cols && gy >= 0 && gy < this.rows) {
      this.grid[gy * this.cols + gx].push(dot);
    }
  }

  render(numeral, type = "red-green", difficulty = "medium") {
    this.dots = [];
    this.initGrid();

    const cx = this.width / 2;
    const cy = this.height / 2;
    const maxR = this.width / 2 - 2;

    // Clear and draw authentic circular paper disc
    this.ctx.clearRect(0, 0, this.width, this.height);
    this.ctx.save();
    this.ctx.beginPath();
    this.ctx.arc(cx, cy, maxR, 0, 2 * Math.PI);
    this.ctx.fillStyle = "#fbfaf5"; // Warm lithographic paper base
    this.ctx.fill();
    this.ctx.clip(); // Clip so dots never bleed outside circular disc

    // Render offscreen text mask for numeral
    const mask = document.createElement("canvas");
    mask.width = this.width;
    mask.height = this.height;
    const mctx = mask.getContext("2d");

    const fontSize = numeral.length === 1
      ? Math.round(this.width * 0.56)
      : (numeral.length === 2 ? Math.round(this.width * 0.44) : Math.round(this.width * 0.36));

    mctx.font = `bold ${fontSize}px system-ui, Arial, sans-serif`;
    mctx.textAlign = "center";
    mctx.textBaseline = "middle";
    mctx.fillStyle = "#000000";
    mctx.fillText(numeral, cx, cy + 2);

    const imgData = mctx.getImageData(0, 0, this.width, this.height).data;

    // Pass 1: Figure dots (multi-scale dots inside numeral strokes)
    const step = 3;
    for (let y = 0; y < this.height; y += step) {
      for (let x = 0; x < this.width; x += step) {
        const idx = (y * this.width + x) * 4;
        if (imgData[idx + 3] > 160 && Math.random() < 0.65) {
          const jx = x + (Math.random() - 0.5) * 4;
          const jy = y + (Math.random() - 0.5) * 4;
          const rad = 2.8 + Math.random() * 4.6;
          if (this.canPlaceDotGrid(jx, jy, rad)) {
            this.addDot({
              x: jx,
              y: jy,
              r: rad,
              color: getIshiharaColor(difficulty, "figure", type),
              isFig: true
            });
          }
        }
      }
    }

    // Pass 2: Fill figure gaps with smaller dots
    for (let tries = 0; tries < 1000 && this.dots.filter(d => d.isFig).length < 160; tries++) {
      const rx = Math.floor(Math.random() * this.width);
      const ry = Math.floor(Math.random() * this.height);
      if (imgData[(ry * this.width + rx) * 4 + 3] > 160) {
        const rad = 2.2 + Math.random() * 3.5;
        if (this.canPlaceDotGrid(rx, ry, rad)) {
          this.addDot({
            x: rx,
            y: ry,
            r: rad,
            color: getIshiharaColor(difficulty, "figure", type),
            isFig: true
          });
        }
      }
    }

    // Pass 3: Fill background dots (Poisson-disc packing with multi-scale dots)
    const targetTotal = this.width < 100 ? 120 : 650;
    for (let i = 0; i < 5000 && this.dots.length < targetTotal; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const dist = Math.sqrt(Math.random()) * maxR;
      const bx = cx + dist * Math.cos(angle);
      const by = cy + dist * Math.sin(angle);
      const rad = 2.4 + Math.random() * 4.4;
      if (this.canPlaceDotGrid(bx, by, rad)) {
        this.addDot({
          x: bx,
          y: by,
          r: rad,
          color: getIshiharaColor(difficulty, "background", type),
          isFig: false
        });
      }
    }

    // Draw dots with subtle soft shadow for realistic lithographic texture
    this.ctx.shadowColor = "rgba(0, 0, 0, 0.18)";
    this.ctx.shadowBlur = 1.2;
    this.ctx.shadowOffsetX = 0.5;
    this.ctx.shadowOffsetY = 0.5;

    // Draw background dots first
    for (const d of this.dots.filter(d => !d.isFig)) {
      this.ctx.fillStyle = d.color;
      this.ctx.beginPath();
      this.ctx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
      this.ctx.fill();
    }

    // Draw figure dots on top
    for (const d of this.dots.filter(d => d.isFig)) {
      this.ctx.fillStyle = d.color;
      this.ctx.beginPath();
      this.ctx.arc(d.x, d.y, d.r, 0, 2 * Math.PI);
      this.ctx.fill();
    }

    this.ctx.shadowColor = "transparent";
    this.ctx.restore();
  }

  toDataURL() {
    return this.canvas.toDataURL("image/png");
  }
}

/**
 * Standard Ishihara screening reference plates.
 */
const FALLBACK_PLATES = [
  {
    id: "plate-demo-01",
    numeral: "12",
    type: "control",
    title: "Plate 1 · Demonstration (Control)",
    prompt: "What number do you see in the circle?",
    timeLimit: 12,
    options: ["12", "15", "17", "13"],
    normal_answer: "12",
    protan_answer: "12",
    deutan_answer: "12",
    tritan_answer: "12",
  },
  {
    id: "plate-rg-02",
    numeral: "74",
    type: "red-green",
    title: "Plate 2 · Red-Green Screening",
    prompt: "What number do you see in the circle?",
    timeLimit: 12,
    options: ["74", "21", "71", "79"],
    normal_answer: "74",
    protan_answer: "21",
    deutan_answer: "21",
    tritan_answer: "74",
  },
  {
    id: "plate-protan-03",
    numeral: "6",
    type: "protan",
    title: "Plate 3 · Protan (Red) Diagnostic",
    prompt: "What number do you see in the circle?",
    timeLimit: 12,
    options: ["6", "8", "9", "0"],
    normal_answer: "6",
    protan_answer: "I cannot see any number",
    deutan_answer: "6",
    tritan_answer: "6",
  },
  {
    id: "plate-deutan-04",
    numeral: "2",
    type: "deutan",
    title: "Plate 4 · Deutan (Green) Diagnostic",
    prompt: "What number do you see in the circle?",
    timeLimit: 12,
    options: ["2", "3", "5", "7"],
    normal_answer: "2",
    protan_answer: "2",
    deutan_answer: "I cannot see any number",
    tritan_answer: "2",
  },
  {
    id: "plate-tritan-05",
    numeral: "15",
    type: "blue-yellow",
    title: "Plate 5 · Tritan (Blue-Yellow) Diagnostic",
    prompt: "What number do you see in the circle?",
    timeLimit: 12,
    options: ["15", "13", "16", "75"],
    normal_answer: "15",
    protan_answer: "15",
    deutan_answer: "15",
    tritan_answer: "I cannot see any number",
  },
];

/**
 * Client-side scoring algorithm supporting both dynamic sessions and static fallback.
 */
function evaluateScreeningLocally(answers, activePlates = null) {
  const sourcePlates = (activePlates && activePlates.length > 0)
    ? activePlates
    : (typeof session !== "undefined" && session?.plates ? session.plates : FALLBACK_PLATES);

  const platesMap = {};
  sourcePlates.forEach(p => {
    platesMap[p.id] = p;
  });

  let protanVotes = 0;
  let deutanVotes = 0;
  let tritanVotes = 0;
  let errors = 0;
  let correctCount = 0;

  const breakdown = [];

  for (const a of answers) {
    const plate = platesMap[a.plate_id];
    if (!plate) continue;

    const userAns = a.user_answer;
    const normalAns = plate.normal_answer;
    const isCorrect = userAns === normalAns;

    if (isCorrect) {
      correctCount++;
    } else {
      errors++;
      if (userAns === plate.protan_answer) protanVotes += 3;
      if (userAns === plate.deutan_answer) deutanVotes += 3;
      if (userAns === plate.tritan_answer) tritanVotes += 3;

      if (plate.type === "protan") protanVotes += 4;
      else if (plate.type === "deutan") deutanVotes += 4;
      else if (plate.type === "blue-yellow" || plate.type === "tritan") tritanVotes += 5;
      else if (plate.type === "red-green") {
        protanVotes += 2;
        deutanVotes += 2;
      }
    }

    breakdown.push({
      plate_id: plate.id,
      title: plate.title,
      numeral: plate.numeral,
      expected: normalAns,
      userAnswer: userAns || "Timeout / Unclear",
      isCorrect: isCorrect,
    });
  }

  const total = answers.length || sourcePlates.length || 5;
  const accuracy = Math.round((correctCount / total) * 100);

  if (errors === 0) {
    return {
      diagnosed: false,
      condition: "Normal Color Vision",
      condition_id: null,
      severity: 0.0,
      errors: 0,
      correct: correctCount,
      accuracy: 100,
      total: total,
      confidence: "High",
      message: "No color confusion detected across test plates. Standard color vision verified.",
      breakdown: breakdown,
    };
  }

  const scores = [
    { votes: protanVotes, id: 0, name: "Protanopia (Red-weak)" },
    { votes: deutanVotes, id: 1, name: "Deuteranopia (Green-weak)" },
    { votes: tritanVotes, id: 2, name: "Tritanopia (Blue-weak)" },
  ];
  scores.sort((a, b) => b.votes - a.votes);
  const top = scores[0];

  const estSeverity = Math.round(Math.min(1.0, Math.max(0.5, errors / total)) * 100) / 100;

  return {
    diagnosed: false,
    condition: top.name,
    condition_id: top.id,
    severity: estSeverity,
    errors: errors,
    correct: correctCount,
    accuracy: accuracy,
    total: total,
    confidence: top.votes >= 4 ? "High" : "Moderate",
    message: "Color vision deficiency detected based on dynamic Ishihara plate confusion. Automatically configured in your Personalization profile.",
    breakdown: breakdown,
  };
}

// ── Export all symbols AFTER every class/const/function is fully defined ──────
window.mulberry32               = mulberry32;
window.shuffleArray             = shuffleArray;
window.CLINICAL_NUMERALS        = CLINICAL_NUMERALS;
window.generateDynamicDistractors = generateDynamicDistractors;
window.generateDynamicPlate     = generateDynamicPlate;
window.generateDynamicSession   = generateDynamicSession;
window.generateTestSession      = generateTestSession;
window.evaluateIshiharaTest     = evaluateIshiharaTest;
window.IshiharaPlateRenderer    = IshiharaPlateRenderer;
window.FALLBACK_PLATES          = FALLBACK_PLATES;
window.evaluateScreeningLocally = evaluateScreeningLocally;
