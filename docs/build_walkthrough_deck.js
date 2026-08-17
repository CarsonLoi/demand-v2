/**
 * build_walkthrough_deck.js — the "explain it to someone brand new" deck.
 *
 * Separate from docs/build_deck.js (the executive summary deck) and writes to
 * a DIFFERENT file, so the existing deck is not touched.
 *
 *   node docs/build_walkthrough_deck.js
 *   -> docs/Demand_Forecast_Walkthrough.pptx
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const D = JSON.parse(fs.readFileSync(path.join(__dirname, "deck_data.json"), "utf8"));

// ── palette: navy dominant, teal support, gold accent (same colours used in
//    every analysis chart in this project, so the deck matches the artefacts)
const NAVY = "0F2942", NAVY_2 = "1B3D5E", TEAL = "2E8BA8", GOLD = "E8B547";
const RED = "C74B4B", GREEN = "2FA87C", GREY = "5F7183", LIGHT = "F4F6F8", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

const W = 13.3, H = 7.5;
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Demand Forecast";
pres.title = "Demand Forecast Model — Complete Walkthrough";

let sectionNo = 0;

/* ───────────────────────── helpers ───────────────────────── */

// Dark section divider
function section(title, subtitle, notes) {
  sectionNo += 1;
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.ShapeType.ellipse, {
    x: 0.9, y: 2.5, w: 1.15, h: 1.15, fill: { color: GOLD },
  });
  s.addText(String(sectionNo), {
    x: 0.9, y: 2.5, w: 1.15, h: 1.15, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 40, bold: true, color: NAVY, margin: 0,
  });
  s.addText(title, {
    x: 2.45, y: 2.45, w: 9.8, h: 0.95, fontFace: HEAD, fontSize: 40,
    bold: true, color: WHITE, valign: "middle", margin: 0,
  });
  s.addText(subtitle, {
    x: 2.5, y: 3.45, w: 9.6, h: 0.8, fontFace: BODY, fontSize: 17,
    color: "AEC3D6", valign: "top", margin: 0,
  });
  if (notes) s.addNotes(notes);
  return s;
}

// Light content slide with a title
function slide(title, notes) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText(title, {
    x: 0.6, y: 0.4, w: 12.1, h: 0.75, fontFace: HEAD, fontSize: 32,
    bold: true, color: NAVY, valign: "middle", margin: 0,
  });
  if (notes) s.addNotes(notes);
  return s;
}

// Sub-heading line under the title
function lede(s, text, y) {
  s.addText(text, {
    x: 0.6, y: y === undefined ? 1.18 : y, w: 12.1, h: 0.45,
    fontFace: BODY, fontSize: 15, color: GREY, italic: true, margin: 0,
  });
}

// Rounded card
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || LIGHT },
    line: { color: "E2E8EE", width: 1 },
  });
}

// Numbered / icon circle
function circle(s, x, y, d, text, fill, txtColor, size) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(text, {
    x, y, w: d, h: d, align: "center", valign: "middle", margin: 0,
    fontFace: HEAD, fontSize: size || 18, bold: true, color: txtColor || WHITE,
  });
}

// Big number callout
function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 1.0, fontFace: HEAD, fontSize: 54, bold: true,
    color: color || NAVY, align: "center", valign: "middle", margin: 0,
  });
  s.addText(label, {
    x, y: y + 1.0, w, h: 0.6, fontFace: BODY, fontSize: 13, color: GREY,
    align: "center", valign: "top", margin: 0,
  });
}

const CHART_BASE = () => ({
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BODY,
  catAxisLabelColor: GREY, valAxisLabelColor: GREY,
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
  valGridLine: { color: "E8EDF2", size: 1 },
  catGridLine: { style: "none" },
  chartColors: [NAVY, TEAL, GOLD, RED],
});

/* ═════════════════════ 1. TITLE ═════════════════════ */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Demand Forecast Model", {
    x: 0.9, y: 2.25, w: 11.5, h: 1.0, fontFace: HEAD, fontSize: 50,
    bold: true, color: WHITE, margin: 0,
  });
  s.addText("A complete walkthrough — how it works, and why", {
    x: 0.95, y: 3.3, w: 11.5, h: 0.6, fontFace: BODY, fontSize: 21,
    color: GOLD, margin: 0,
  });
  s.addText("Written for someone who has never seen this system before.\nNo statistics background needed.", {
    x: 0.95, y: 4.05, w: 10, h: 0.9, fontFace: BODY, fontSize: 14,
    color: "AEC3D6", margin: 0, lineSpacingMultiple: 1.3,
  });
  s.addNotes("This deck walks through the entire forecasting system from first principles. It assumes no prior knowledge.");
}

/* ═════════════════════ 2. WHAT IT DOES ═════════════════════ */
{
  const s = slide("What this system does");
  lede(s, "One job: guess how busy the casino floor will be, every day, for the next four weeks.");

  s.addText([
    { text: "It predicts ", options: { color: NAVY } },
    { text: "patron hours", options: { color: TEAL, bold: true } },
    { text: " — roughly, how many people are on the floor multiplied by how long they stay.", options: { color: NAVY } },
  ], { x: 0.6, y: 1.85, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 16, margin: 0 });

  s.addText("That number drives staff scheduling. Too low and the floor is understaffed; too high and you pay for labour you did not need.", {
    x: 0.6, y: 2.35, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 14, color: GREY, margin: 0,
  });

  card(s, 0.6, 3.15, 3.85, 2.35);
  card(s, 4.72, 3.15, 3.85, 2.35);
  card(s, 8.85, 3.15, 3.85, 2.35);
  stat(s, 0.6, 3.5, 3.85, "28", "days ahead, every run", NAVY);
  stat(s, 4.72, 3.5, 3.85, "~4.3%", "typical average miss", TEAL);
  stat(s, 8.85, 3.5, 3.85, "3", "numbers per day", GOLD);

  s.addText("For each of the 28 days it gives three numbers: a most-likely figure (P50), a low case (P10) and a high case (P90). Plan against P50.", {
    x: 0.6, y: 5.9, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 14, color: NAVY, margin: 0,
  });
  s.addText("Read \"4.3% miss\" as: if the real answer was 7,000 patron hours, a typical forecast lands about 300 either side.", {
    x: 0.6, y: 6.45, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 13, color: GREY, italic: true, margin: 0,
  });
  s.addNotes("MAPE = mean absolute percentage error. 4.3% across a full year. Ordinary days ~3.8%, Chinese New Year ~7.3%.");
}

/* ═════════════════════ 3. MENTAL MODEL ═════════════════════ */
{
  const s = slide("The whole system, in one line");
  lede(s, "There is no database, no stored model, no training pipeline to babysit.");

  const boxes = [
    ["One data file", "data/raw/rawdata.csv\nOne row per day", TEAL],
    ["One script", "forecast.py\nRun it when you want a forecast", NAVY],
    ["One output folder", "predictions.csv\nplus a chart", GOLD],
  ];
  let x = 0.75;
  boxes.forEach(([t, b, c], i) => {
    card(s, x, 2.1, 3.5, 2.3, LIGHT);
    circle(s, x + 0.25, 2.35, 0.55, String(i + 1), c, WHITE, 18);
    s.addText(t, { x: x + 0.25, y: 3.05, w: 3.0, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.25, y: 3.5, w: 3.0, h: 0.85, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });
    if (i < 2) {
      s.addText("→", { x: x + 3.55, y: 3.0, w: 0.55, h: 0.6, fontFace: BODY, fontSize: 26, color: GOLD, align: "center", valign: "middle", margin: 0 });
    }
    x += 4.1;
  });

  card(s, 0.75, 4.8, 11.8, 2.05, "FBF6E9");
  s.addText("Every run starts from scratch", {
    x: 1.05, y: 5.05, w: 11.2, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0,
  });
  s.addText("There is no \"training step\" you run separately. Each time you ask for a forecast, the model is rebuilt from all the data available at that moment, makes its predictions, and is thrown away. Nothing is saved between runs — so nothing can quietly go stale or drift out of date without anyone noticing.", {
    x: 1.05, y: 5.5, w: 11.2, h: 1.15, fontFace: BODY, fontSize: 13.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.25,
  });
  s.addNotes("This is a deliberate design choice: statelessness removes an entire category of production bugs.");
}

/* ═════════════════════ SECTION 1 ═════════════════════ */
section("The one rule that shapes everything",
  "Almost every design decision in this system exists to obey a single constraint.",
  "If you remember one thing from this deck, make it this section.");

/* ═════════════════════ 4. NO PEEKING ═════════════════════ */
{
  const s = slide("No peeking");
  lede(s, "A forecast may only use information that genuinely existed at the time it was made.");

  s.addText("Suppose today is 1 March and you want to predict 15 March — that is 14 days ahead.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

  // timeline
  const tY = 2.55, tX = 0.9, tW = 11.5;
  s.addShape(pres.ShapeType.rect, { x: tX, y: tY + 0.62, w: tW, h: 0.035, fill: { color: "C9D4DE" } });
  s.addShape(pres.ShapeType.rect, { x: tX, y: tY, w: tW * 0.46, h: 0.62, fill: { color: "E4EFF3" } });
  s.addShape(pres.ShapeType.rect, { x: tX + tW * 0.46, y: tY, w: tW * 0.35, h: 0.62, fill: { color: "F7E3E3" } });
  s.addShape(pres.ShapeType.rect, { x: tX + tW * 0.81, y: tY, w: tW * 0.19, h: 0.62, fill: { color: "FBF0D6" } });

  s.addText("data you MAY use", { x: tX, y: tY, w: tW * 0.46, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: 12.5, bold: true, color: NAVY, margin: 0 });
  s.addText("the last 13 days — FORBIDDEN", { x: tX + tW * 0.46, y: tY, w: tW * 0.35, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: 12.5, bold: true, color: RED, margin: 0 });
  s.addText("the day you predict", { x: tX + tW * 0.81, y: tY, w: tW * 0.19, h: 0.62, align: "center", valign: "middle", fontFace: BODY, fontSize: 12.5, bold: true, color: "8A6A1F", margin: 0 });

  s.addText("… up to 28 Feb", { x: tX, y: tY + 0.7, w: tW * 0.46, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: GREY, margin: 0 });
  s.addText("1 – 14 March", { x: tX + tW * 0.46, y: tY + 0.7, w: tW * 0.35, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: GREY, margin: 0 });
  s.addText("15 March", { x: tX + tW * 0.81, y: tY + 0.7, w: tW * 0.19, h: 0.35, align: "center", fontFace: BODY, fontSize: 11, color: GREY, margin: 0 });

  card(s, 0.6, 4.15, 5.9, 2.6);
  circle(s, 0.9, 4.42, 0.5, "!", RED);
  s.addText("Why this matters so much", { x: 1.55, y: 4.45, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("On 1 March, the demand for 5 March has not happened yet. If the model were allowed to look at it while learning, it would score brilliantly in testing and fail completely in real use.\n\nThis mistake is called leakage. It is the single most common way forecasting projects fail.", {
    x: 0.9, y: 5.05, w: 5.3, h: 1.55, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 6.75, 4.15, 5.95, 2.6);
  circle(s, 7.05, 4.42, 0.5, "✓", GREEN);
  s.addText("How it is enforced", { x: 7.7, y: 4.45, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Every backward-looking number carries a rule: it may not read anything newer than (days ahead + 1).\n\nThis was also tested, not just assumed — a fake value was planted in future data and confirmed never to reach the predictions.", {
    x: 7.05, y: 5.05, w: 5.35, h: 1.55, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("The technical term is horizon gating: min_safe = horizon + 1. Verified empirically with a sentinel-value injection test.");
}

/* ═════════════════════ 5. 28 MODELS ═════════════════════ */
{
  const s = slide("Why there are 28 models, not one");
  lede(s, "A direct consequence of the no-peeking rule.");

  s.addText("A forecast for tomorrow may use yesterday's numbers. A forecast for four weeks out may not. They are genuinely different problems, so each gets its own model.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.55, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

  const rows = [
    ["Model 1", "predicts tomorrow", "may use data up to yesterday", "the most information"],
    ["Model 7", "predicts one week out", "nothing from the last 7 days", "less information"],
    ["Model 28", "predicts four weeks out", "nothing from the last 28 days", "the least information"],
  ];
  let y = 2.5;
  rows.forEach(([a, b, c, d], i) => {
    const tint = i === 0 ? "EAF2F5" : i === 1 ? "F1F4F7" : "F7F3EA";
    card(s, 0.6, y, 12.1, 1.0, tint);
    s.addText(a, { x: 0.95, y: y + 0.28, w: 1.6, h: 0.45, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 2.65, y: y + 0.3, w: 2.9, h: 0.4, fontFace: BODY, fontSize: 14, color: NAVY, margin: 0 });
    s.addText(c, { x: 5.7, y: y + 0.3, w: 4.2, h: 0.4, fontFace: BODY, fontSize: 13.5, color: GREY, margin: 0 });
    s.addText(d, { x: 10.0, y: y + 0.3, w: 2.4, h: 0.4, fontFace: BODY, fontSize: 13, color: TEAL, italic: true, margin: 0 });
    y += 1.12;
  });

  s.addText("This is also why accuracy gets worse the further out you look — there is simply less to go on. Any accuracy number is meaningless unless you say how far ahead it was measured.", {
    x: 0.6, y: 6.0, w: 12.1, h: 0.6, fontFace: BODY, fontSize: 14, color: NAVY, margin: 0,
  });
  s.addNotes("28 separate LightGBM models are trained per run, one per forecast distance.");
}

/* ═════════════════════ SECTION 2 ═════════════════════ */
section("What the model looks at",
  "About 170 pieces of information per day — and they are not equally important.",
  "Feature importance measured by LightGBM gain.");

/* ═════════════════════ 6. FEATURE DONUT ═════════════════════ */
{
  const s = slide("Half the model is holidays");
  lede(s, "Share of the model's decision-making, measured — not assumed.");

  s.addChart(pres.ChartType.doughnut, [{
    name: "Share",
    labels: ["Holidays", "Recent history", "Calendar", "Everything else"],
    values: [47, 15, 12, 26],
  }], {
    x: 0.5, y: 1.7, w: 5.6, h: 5.1, holeSize: 52,
    chartColors: [GOLD, TEAL, NAVY, "C3CDD6"],
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: BODY,
    showValue: true, dataLabelFontSize: 12, dataLabelColor: "FFFFFF",
    dataLabelFormatCode: '0"%"',
    showTitle: false,
  });

  const items = [
    ["Holidays — 47%", "Chinese New Year, Golden Week, Labour Day, Mid-Autumn and five more. Not single days but windows: CNY disrupts trade for about three weeks.", GOLD],
    ["Recent history — 15%", "What demand did 2, 7, 28, 365 days ago; rolling averages; the last few same-weekdays.", TEAL],
    ["Calendar — 12%", "Day of week, month, position in the year. Day of week matters enormously — Saturday is a different business from Tuesday.", NAVY],
    ["Everything else — 26%", "Tables open on the floor, the Mainland-China working calendar, and combinations of the above.", "8A94A0"],
  ];
  let y = 1.95;
  items.forEach(([t, b, c]) => {
    s.addShape(pres.ShapeType.ellipse, { x: 6.55, y: y + 0.06, w: 0.22, h: 0.22, fill: { color: c } });
    s.addText(t, { x: 6.95, y, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 6.95, y: y + 0.36, w: 5.75, h: 0.78, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.28;
  });
  s.addNotes("Holidays being ~47% of total gain is the single most important fact about this model's structure.");
}

/* ═════════════════════ 7. RECENCY WEIGHTING ═════════════════════ */
{
  const s = slide("Not every day in history counts equally");
  lede(s, "Older days are deliberately given less say. How much less is a setting called half-life.");

  s.addChart(pres.ChartType.line, [
    { name: "240-day half-life (production)", labels: D.weights.labels, values: D.weights.hl240 },
    { name: "1095-day half-life (long-history version)", labels: D.weights.labels, values: D.weights.hl1095 },
  ], Object.assign(CHART_BASE(), {
    x: 0.5, y: 1.75, w: 7.4, h: 4.75,
    chartColors: [TEAL, GOLD], lineSize: 3, lineSmooth: true,
    catAxisTitle: "days in the past", showCatAxisTitle: true,
    valAxisTitle: "how much that day counts (%)", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  }));

  card(s, 8.2, 1.85, 4.5, 2.0, "FBF6E9");
  s.addText("Reading the chart", { x: 8.5, y: 2.05, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("With a 240-day half-life, a day from 240 days ago counts half as much as today. From 480 days ago, a quarter. And so on — it falls away fast.", {
    x: 8.5, y: 2.42, w: 3.9, h: 1.3, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 8.2, 4.2, 4.5, 2.5, "F7E9E9");
  s.addText("Why this trips people up", { x: 8.5, y: 4.42, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("At a 240-day half-life, a day from 2016 counts about 1/35,000th of today. So loading ten years of history and changing nothing else does literally nothing.\n\nThe half-life is not a follow-up to adding history — it IS the change.", {
    x: 8.5, y: 4.82, w: 3.9, h: 1.75, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addText("Holiday rows get an extra 3× boost on top — they are rare but high-stakes.", {
    x: 0.6, y: 6.65, w: 7.3, h: 0.4, fontFace: BODY, fontSize: 12.5, color: GREY, italic: true, margin: 0,
  });
  s.addNotes("weight = 0.5 ^ (days_back / half_life). Production uses 240; the long-history pipeline defaults to 1095.");
}

/* ═════════════════════ 8. THE MODEL ═════════════════════ */
{
  const s = slide("The model itself: many small, weak guesses");
  lede(s, "LightGBM — gradient-boosted decision trees. The name is scarier than the idea.");

  card(s, 0.6, 1.75, 7.5, 2.6, LIGHT);
  s.addText("How boosting works", { x: 0.95, y: 1.95, w: 6.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Build a small, crude rule. See where it is wrong. Build a second rule that fixes those mistakes. Then a third that fixes what is still wrong — and repeat 500 times.\n", options: { breakLine: true } },
    { text: "No single rule is clever. Added together, they are.", options: { bold: true, color: NAVY } },
  ], { x: 0.95, y: 2.4, w: 6.85, h: 1.7, fontFace: BODY, fontSize: 13.5, color: GREY, margin: 0, lineSpacingMultiple: 1.25 });

  card(s, 8.35, 1.75, 4.35, 2.6, "EAF2F5");
  s.addText("Why this model", { x: 8.65, y: 1.95, w: 3.8, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Finds combinations on its own", options: { bullet: true, breakLine: true } },
    { text: "Copes with missing values — essential here, since long-range models have data deliberately withheld", options: { bullet: true, breakLine: true } },
    { text: "Trains in minutes on a laptop", options: { bullet: true } },
  ], { x: 8.75, y: 2.4, w: 3.7, h: 1.8, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, paraSpaceAfter: 6 });

  s.addText("What was rejected, and why", { x: 0.6, y: 4.6, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  const rej = [
    ["Classic time-series (ARIMA)", "cannot absorb 170 different inputs"],
    ["Neural networks", "need far more than a decade of daily data"],
    ["GAM (a smoother statistical model)", "tested this project — worse at all 9 forecast distances"],
  ];
  let x2 = 0.6;
  rej.forEach(([t, b]) => {
    card(s, x2, 5.25, 3.93, 1.6, "F7E9E9");
    s.addText(t, { x: x2 + 0.25, y: 5.45, w: 3.45, h: 0.5, fontFace: BODY, fontSize: 13, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x2 + 0.25, y: 5.95, w: 3.45, h: 0.75, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0 });
    x2 += 4.08;
  });
  s.addNotes("500 trees, learning rate 0.02, depth 5. Deliberately small trees to avoid overfitting on limited data.");
}

/* ═════════════════════ SECTION 3 ═════════════════════ */
section("Chinese New Year — the hard part",
  "The single largest source of error, and the one place the model gets deliberate help.",
  "CNY is the defining difficulty of this problem.");

/* ═════════════════════ 9. THE CNY COLLAPSE ═════════════════════ */
{
  const s = slide("What actually happens at Chinese New Year");
  lede(s, "Real demand around CNY 2025 (29 January). Trade does not dip — it collapses, then rebounds.");

  s.addChart(pres.ChartType.line, [
    { name: "actual demand (patron hours)", labels: D.cny.labels, values: D.cny.values },
  ], Object.assign(CHART_BASE(), {
    x: 0.5, y: 1.7, w: 8.3, h: 5.0,
    chartColors: [NAVY], lineSize: 3, lineSmooth: false,
    showLegend: false,
    valAxisMinVal: 0,
    catAxisLabelRotate: 45,
  }));

  card(s, 9.05, 1.85, 3.65, 1.85, "F7E9E9");
  s.addText("The collapse", { x: 9.3, y: 2.02, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Demand on the eve of CNY falls to roughly HALF of normal — 3,331 against a typical 7,300.", {
    x: 9.3, y: 2.4, w: 3.15, h: 1.2, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 9.05, 4.0, 3.65, 2.7, "FBF6E9");
  s.addText("Why the model struggles", { x: 9.3, y: 4.2, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Two reasons at once:\n\n1.  It has seen only one or two previous CNYs to learn from.\n\n2.  Its \"how busy are we lately\" inputs are themselves dragged down by the same collapse — right when it needs them most.", {
    x: 9.3, y: 4.68, w: 3.15, h: 1.95, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addNotes("Eve-of-CNY ratios: 0.43x in 2024, 0.47x in 2025, 0.49x in 2026 — the shape is remarkably stable year to year, which is what makes the anchor work.");
}

/* ═════════════════════ 10. THE ANCHOR ═════════════════════ */
{
  const s = slide("The fix: stop guessing, use last year's shape");
  lede(s, "Rather than hoping the model discovers the pattern, the answer is corrected afterwards.");

  s.addText("The collapse is dramatic, but it is also remarkably consistent year to year. That makes it something you can simply look up instead of learn.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

  const steps = [
    ["1", "Find the matching day", "Not the same date — the same POSITION. \"Three days before CNY\" this year matches \"three days before CNY\" last year."],
    ["2", "Work out the shape", "Last year that day ran at 0.47× normal trade. That ratio is the pattern."],
    ["3", "Apply to today's level", "Multiply this year's normal trade by that ratio. Correct for day of week."],
    ["4", "Blend it in", "Final answer = 20% model + 80% this lookup."],
  ];
  let x = 0.6;
  steps.forEach(([n, t, b]) => {
    card(s, x, 2.45, 2.95, 2.6, LIGHT);
    circle(s, x + 0.3, 2.7, 0.5, n, GOLD, NAVY, 17);
    s.addText(t, { x: x + 0.3, y: 3.32, w: 2.4, h: 0.55, fontFace: HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 3.88, w: 2.4, h: 1.05, fontFace: BODY, fontSize: 11, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    x += 3.1;
  });

  card(s, 0.6, 5.35, 6.0, 1.65, "E9F4EF");
  s.addText("It works", { x: 0.9, y: 5.52, w: 5.4, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("CNY error roughly halved: 11.9% → 6.9% at short range, 16.1% → 7.4% at long range. Calibrated on 2025 and applied blind to 2026.", {
    x: 0.9, y: 5.9, w: 5.4, h: 1.0, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 6.75, 5.35, 5.95, 1.65, "F7E9E9");
  s.addText("Chinese New Year only", { x: 7.05, y: 5.52, w: 5.35, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("The same trick was tried on Easter, Dragon Boat and Mid-Autumn and made all three WORSE. Their swings are mild, so a lookup built from one or two past years is mostly noise.", {
    x: 7.05, y: 5.9, w: 5.35, h: 1.0, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("blended = 0.2 * model + 0.8 * profile. The alpha curve is flat between 0.75 and 0.95, so the result does not depend on hitting it precisely.");
}

/* ═════════════════════ 11. FIXED VS MOVING ═════════════════════ */
{
  const s = slide("Two kinds of holiday, handled differently");
  lede(s, "This distinction causes more subtle bugs than anything else in the system.");

  card(s, 0.6, 1.8, 5.95, 2.5, "EAF2F5");
  circle(s, 0.9, 2.05, 0.5, "F", TEAL);
  s.addText("Fixed holidays", { x: 1.55, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Christmas, New Year, Labour Day, Golden Week, Ching Ming.\n\nSame date every year. To find last year's version, count back 364 days — exactly 52 weeks, which also lands on the same day of the week.", {
    x: 0.9, y: 2.65, w: 5.35, h: 1.5, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 6.75, 1.8, 5.95, 2.5, "FBF6E9");
  circle(s, 7.05, 2.05, 0.5, "M", GOLD, NAVY);
  s.addText("Moving holidays", { x: 7.7, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Chinese New Year, Mid-Autumn, Dragon Boat, Easter.\n\nThey follow the lunar calendar and can shift by up to three weeks. Counting back 364 days lands on completely the wrong part of the holiday.", {
    x: 7.05, y: 2.65, w: 5.35, h: 1.5, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 0.6, 4.7, 12.1, 2.35, "F7E9E9");
  s.addText("A real bug this caused", { x: 0.95, y: 4.95, w: 11.4, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "CNY 2025 fell on 29 January. CNY 2026 falls on 17 February — 19 days later. ", options: { color: NAVY } },
    { text: "So for a date in late January 2026, \"the same date last year\" landed deep inside the 2025 collapse — comparing an ordinary Tuesday against a day running at half normal trade.", options: { color: NAVY } },
  ], { x: 0.95, y: 5.4, w: 11.4, h: 0.6, fontFace: BODY, fontSize: 13, margin: 0, lineSpacingMultiple: 1.2 });
  s.addText("The model was handed a number far outside anything it had seen, and missed 2 February 2026 by 27.7% — predicting 4,964 when the answer was 6,868.", {
    x: 0.95, y: 6.15, w: 11.4, h: 0.7, fontFace: BODY, fontSize: 13, bold: true, color: RED, margin: 0,
  });
  s.addNotes("Moving holidays are matched by POSITION within the holiday, not by calendar date. Ching Ming is also a solar term — production hardcodes 4 April, but it falls on 5 April in 2018, 2019, 2022, 2023.");
}

/* ═════════════════════ SECTION 4 ═════════════════════ */
section("How good is it, really?",
  "Measured on held-out data — days the model had never seen when it made the call.",
  "Every number here comes from honest backtests, not from fitting the training data.");

/* ═════════════════════ 12. ACCURACY BY LEAD ═════════════════════ */
{
  const s = slide("Accuracy depends on how far ahead you ask");
  lede(s, "A single accuracy figure means nothing without the forecast distance attached.");

  s.addChart(pres.ChartType.line, [
    { name: "ordinary days", labels: D.lead.labels, values: D.lead.ex },
    { name: "all days", labels: D.lead.labels, values: D.lead.all },
    { name: "Chinese New Year days", labels: D.lead.labels, values: D.lead.cny },
  ], Object.assign(CHART_BASE(), {
    x: 0.5, y: 1.75, w: 8.2, h: 4.9,
    chartColors: [TEAL, NAVY, GOLD], lineSize: 3, lineSmooth: true,
    catAxisTitle: "days ahead the forecast was made", showCatAxisTitle: true,
    valAxisTitle: "average % miss", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
    valAxisMinVal: 0,
  }));

  card(s, 8.95, 1.9, 3.75, 1.7, "EAF2F5");
  s.addText("Ordinary days", { x: 9.25, y: 2.05, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("About 2.9% one day out, rising to roughly 4.9% at four weeks. Predictable, and good enough to schedule against.", {
    x: 9.25, y: 2.42, w: 3.15, h: 1.05, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15,
  });

  card(s, 8.95, 3.75, 3.75, 1.55, "FBF6E9");
  s.addText("Holiday days", { x: 9.25, y: 3.9, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Roughly double the error, and it stays high at every distance. This is where the remaining difficulty lives.", {
    x: 9.25, y: 4.27, w: 3.15, h: 0.95, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15,
  });

  card(s, 8.95, 5.45, 3.75, 0.85, "F7E9E9");
  s.addText("Never quote a MAPE without its lead time.", {
    x: 9.25, y: 5.58, w: 3.15, h: 0.6, fontFace: BODY, fontSize: 12.5, bold: true, color: NAVY, margin: 0, valign: "middle",
  });
  s.addNotes("Measured on the current production configuration. n=72 days per lead, 6 of them CNY-window days.");
}

/* ═════════════════════ 13. ACTUAL VS FORECAST ═════════════════════ */
{
  const s = slide("What it looks like in practice");
  lede(s, "May 2026, forecast seven days ahead, re-forecasting weekly. The model had not seen any of these days.");

  s.addChart(pres.ChartType.line, [
    { name: "what actually happened", labels: D.avf.labels, values: D.avf.actual },
    { name: "what the model predicted", labels: D.avf.labels, values: D.avf.forecast },
  ], Object.assign(CHART_BASE(), {
    x: 0.5, y: 1.75, w: 8.5, h: 5.0,
    chartColors: [NAVY, TEAL], lineSize: 3, lineSmooth: true,
    catAxisLabelRotate: 45,
  }));

  stat(s, 9.2, 1.95, 3.5, "3.18%", "average miss over the month", TEAL);
  card(s, 9.2, 3.5, 3.5, 3.2, LIGHT);
  s.addText("Reading this chart", { x: 9.5, y: 3.68, w: 2.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("The model tracks the weekly rhythm well — the weekend peaks and midweek troughs line up.\n\nWhere it struggles is unusual single days. That is the honest limit: it learns patterns, and a one-off event is not a pattern.", {
    x: 9.5, y: 4.05, w: 2.9, h: 2.5, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("Produced by long_history/evaluate_long.py in rolling_window mode, 7-day lead. Bias was -30 patron hours, i.e. very slightly under-forecasting.");
}

/* ═════════════════════ 14. LIMITATIONS ═════════════════════ */
{
  const s = slide("What it cannot do — read this before relying on it");
  lede(s, "Stated plainly, because knowing the limits is part of using it safely.");

  const lims = [
    ["The high/low range is too narrow", "The P10–P90 band is supposed to contain the real answer 80% of the time. Measured, it manages 46–54%. Treat it as a rough confidence signal, not a guarantee.", RED],
    ["It cannot see one-off events", "A concert, a competitor closing, a policy change. If it is not in the calendar and not in the past, the model does not know.", GOLD],
    ["Typhoons are not predicted", "Deliberately — see the next slide.", GOLD],
    ["Holiday calendar ends in 2026", "The underlying calendar library has no 2027 dates. Forecasts made late in 2026 will reach past it. A known, dated deadline.", RED],
  ];
  let y = 1.8;
  lims.forEach(([t, b, c]) => {
    card(s, 0.6, y, 12.1, 1.12, c === RED ? "F7E9E9" : "FBF6E9");
    circle(s, 0.9, y + 0.3, 0.52, "!", c, WHITE, 17);
    s.addText(t, { x: 1.62, y: y + 0.16, w: 10.8, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 1.62, y: y + 0.54, w: 10.8, h: 0.5, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0 });
    y += 1.22;
  });
  s.addNotes("The interval-coverage problem is a correctness issue, not an accuracy one, but people may be sizing risk off that band.");
}

/* ═════════════════════ 15. TYPHOONS ═════════════════════ */
{
  const s = slide("Typhoons: a decision, not a pattern");
  lede(s, "A good example of knowing when NOT to use a model.");

  s.addText("The obvious idea is to feed typhoon warnings in as an input. It was tried, and it cannot work — because the storm is not what closes the casino. A management decision is.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.55, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

  card(s, 0.6, 2.55, 5.95, 1.9, "F7E9E9");
  s.addText("Typhoon RAGASA", { x: 0.95, y: 2.75, w: 5.35, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Operations suspended.\nDemand went to effectively zero.", {
    x: 0.95, y: 3.2, w: 5.35, h: 0.9, fontFace: BODY, fontSize: 13.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 6.75, 2.55, 5.95, 1.9, "E9F4EF");
  s.addText("Typhoon TORAJI", { x: 7.05, y: 2.75, w: 5.35, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Operations stayed open.\nDemand was close to a normal day.", {
    x: 7.05, y: 3.2, w: 5.35, h: 0.9, fontFace: BODY, fontSize: 13.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  s.addText("Similar storms. Opposite outcomes. No amount of weather data separates them, because the deciding factor is human.", {
    x: 0.6, y: 4.65, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 14.5, bold: true, color: NAVY, align: "center", margin: 0,
  });

  card(s, 0.6, 5.45, 12.1, 1.5, LIGHT);
  s.addText("What is done instead", { x: 0.95, y: 5.65, w: 11.4, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("A separate manual tool. When management decides to suspend operations, a person applies that to the forecast directly. The model is never asked to guess a decision it cannot see.", {
    x: 0.95, y: 6.05, w: 11.4, h: 0.7, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0,
  });
  s.addNotes("typhoon_override.py. Closure days are also detected in the data via floortables == 0, which correctly finds Mangkhut 2018 and both COVID closures.");
}

/* ═════════════════════ SECTION 5 ═════════════════════ */
section("The long-history version",
  "A second, complete copy of the system trained on 11 years instead of 2.",
  "Lives in long_history/ and cannot affect the live model.");

/* ═════════════════════ 16. THE 27.7% STORY ═════════════════════ */
{
  const s = slide("Why more history matters — one concrete day");
  lede(s, "2 February 2026. Same day, same inputs, two models.");

  s.addChart(pres.ChartType.bar, [{
    name: "% miss on 2 Feb 2026",
    labels: ["Production model\n(2 years of data)", "Long-history model\n(11 years of data)"],
    values: [27.7, 3.2],
  }], Object.assign(CHART_BASE(), {
    x: 0.5, y: 1.8, w: 6.4, h: 4.8,
    barDir: "col", chartColors: [RED, GREEN], showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 15,
    dataLabelColor: NAVY, dataLabelFormatCode: '0.0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 32,
    valAxisTitle: "how far off it was (%)", showValAxisTitle: true,
    valAxisTitleColor: GREY, valAxisTitleFontSize: 11,
  }));

  card(s, 7.15, 1.9, 5.55, 2.35, "F7E9E9");
  s.addText("What went wrong", { x: 7.45, y: 2.08, w: 4.95, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("The model compares today against \"the same day last year\". That date landed two days before CNY 2025 — when trade was at half normal.\n\nIt was handed a comparison far outside anything in its experience, and had no idea what to do with it.", {
    x: 7.45, y: 2.45, w: 4.95, h: 1.7, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 7.15, 4.55, 5.55, 2.15, "E9F4EF");
  s.addText("Why the longer version coped", { x: 7.45, y: 4.63, w: 4.95, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Nothing was fixed or patched. It had simply watched about ten Chinese New Years drift around the calendar, so the pattern was familiar rather than shocking.", {
    x: 7.45, y: 5.0, w: 4.95, h: 1.2, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("The real argument for long history is not the average (only about -0.3pp) but the removal of catastrophic single-day failures.");
}

/* ═════════════════════ 17. HONEST STATUS ═════════════════════ */
{
  const s = slide("Is the long-history version better? Honestly: partly");
  lede(s, "Reported as measured, including the parts that did not work.");

  card(s, 0.6, 1.8, 5.95, 2.3, "E9F4EF");
  circle(s, 0.9, 2.02, 0.5, "✓", GREEN);
  s.addText("What it clearly wins on", { x: 1.55, y: 2.05, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Removes catastrophic single-day errors (27.7% → 3.2%)", options: { bullet: true, breakLine: true } },
    { text: "Has seen ~10 Chinese New Years, not 2", options: { bullet: true, breakLine: true } },
    { text: "Fixes a real calendar bug in Ching Ming dates", options: { bullet: true } },
  ], { x: 0.95, y: 2.6, w: 5.3, h: 1.35, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, paraSpaceAfter: 5 });

  card(s, 6.75, 1.8, 5.95, 2.3, "FBF6E9");
  circle(s, 7.05, 2.02, 0.5, "~", GOLD, NAVY);
  s.addText("What it does not win on", { x: 7.7, y: 2.05, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Only about 0.3 percentage points better on average", options: { bullet: true, breakLine: true } },
    { text: "Improves most test windows but makes two worse", options: { bullet: true, breakLine: true } },
    { text: "Does not pass the strict \"nothing may get worse\" bar", options: { bullet: true } },
  ], { x: 7.1, y: 2.6, w: 5.3, h: 1.35, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, paraSpaceAfter: 5 });

  s.addText("Ideas tested on it and rejected", { x: 0.6, y: 4.35, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  const rej = [
    ["Blocking the contaminated year-ago comparison", "Looked convincing on one Chinese New Year. Tested against six, it failed — and the long-history model had already solved the problem by itself."],
    ["Swapping in a GAM (smoother model)", "Worse at all nine forecast distances tested. Also no better on holidays, so a holiday-only hybrid was ruled out too."],
  ];
  let x = 0.6;
  rej.forEach(([t, b]) => {
    card(s, x, 4.95, 5.95, 1.95, "F7E9E9");
    s.addText(t, { x: x + 0.3, y: 5.15, w: 5.35, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 5.45, w: 5.35, h: 1.3, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    x += 6.15;
  });
  s.addNotes("The single-event-versus-six-events lesson is the most transferable thing in this deck.");
}

/* ═════════════════════ 18. RUNNING IT ═════════════════════ */
{
  const s = slide("Running it");
  lede(s, "Two systems, two commands. Both check the data before doing anything.");

  card(s, 0.6, 1.8, 5.95, 2.55, LIGHT);
  s.addText("The live forecast", { x: 0.95, y: 1.98, w: 5.35, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("1.  Add yesterday's actual demand to the data file\n2.  Run the forecast\n3.  Read predictions.csv and the chart", {
    x: 0.95, y: 2.42, w: 5.35, h: 1.0, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.3,
  });
  s.addShape(pres.ShapeType.roundRect, { x: 0.95, y: 3.5, w: 5.35, h: 0.62, rectRadius: 0.06, fill: { color: NAVY } });
  s.addText("uv run python forecast.py", {
    x: 1.1, y: 3.5, w: 5.05, h: 0.62, fontFace: "Courier New", fontSize: 12.5, color: "9FE0C8", valign: "middle", margin: 0,
  });

  card(s, 6.75, 1.8, 5.95, 2.55, LIGHT);
  s.addText("The long-history version", { x: 7.05, y: 1.98, w: 5.35, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("One command runs three steps in order and stops at the first failure:\ncheck data  →  measure accuracy  →  forecast", {
    x: 7.05, y: 2.42, w: 5.35, h: 1.0, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.3,
  });
  s.addShape(pres.ShapeType.roundRect, { x: 7.05, y: 3.5, w: 5.35, h: 0.62, rectRadius: 0.06, fill: { color: NAVY } });
  s.addText("uv run python long_history/run_pipeline.py", {
    x: 7.2, y: 3.5, w: 5.05, h: 0.62, fontFace: "Courier New", fontSize: 11, color: "9FE0C8", valign: "middle", margin: 0,
  });

  card(s, 0.6, 4.75, 12.1, 2.15, "F7E9E9");
  circle(s, 0.95, 5.05, 0.55, "!", RED);
  s.addText("The one thing that will silently break everything", { x: 1.75, y: 5.02, w: 10.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("A missing day in the data file. Every \"what happened N days ago\" calculation counts backwards through the calendar, so one absent row quietly corrupts a large number of inputs — with no error message. Both systems check for this before anything else, and refuse to run if they find a gap. Never skip that check.", {
    x: 1.75, y: 5.48, w: 10.6, h: 1.3, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("Gaps must be filled, never worked around by deleting surrounding rows.");
}

/* ═════════════════════ 19. TESTED AND REJECTED ═════════════════════ */
{
  const s = slide("What has already been tried and rejected");
  lede(s, "Recorded so nobody spends a week rediscovering it. Every one was measured on held-out data.");

  const rows = [
    ["Sharing training data between nearby forecast distances", "worse on all 6 test windows"],
    ["Feeding typhoon warnings in as an input", "unlearnable — the driver is a closure decision"],
    ["Removing closure days from the history", "worse (6.28% → 6.75%)"],
    ["Cutting the feature list down to 40 or 60", "worse than keeping all 170"],
    ["Using the CNY trick on other holidays", "worse in 7 of 12 cases"],
    ["Matching CNY by day of week instead of last year", "13.0% vs 7.1%"],
    ["Averaging all past CNYs instead of using last year", "9.4% vs 7.1%"],
    ["Replacing the model with a GAM", "worse at all 9 forecast distances"],
  ];
  let y = 1.85;
  rows.forEach(([a, b], i) => {
    const tint = i % 2 === 0 ? LIGHT : WHITE;
    s.addShape(pres.ShapeType.rect, { x: 0.6, y, w: 12.1, h: 0.55, fill: { color: tint } });
    s.addText(a, { x: 0.95, y, w: 8.0, h: 0.55, fontFace: BODY, fontSize: 13, color: NAVY, valign: "middle", margin: 0 });
    s.addText(b, { x: 9.05, y, w: 3.4, h: 0.55, fontFace: BODY, fontSize: 12.5, color: RED, valign: "middle", margin: 0 });
    y += 0.55;
  });

  card(s, 0.6, 6.4, 12.1, 0.75, "FBF6E9");
  s.addText("The pattern: tinkering with individual inputs does not work on this problem. Assume any idea is neutral until it is measured.", {
    x: 0.95, y: 6.4, w: 11.4, h: 0.75, fontFace: BODY, fontSize: 13, bold: true, color: NAVY, valign: "middle", margin: 0,
  });
  s.addNotes("This table is the most valuable page in the deck for anyone planning to improve the model.");
}

/* ═════════════════════ 20. CLOSING ═════════════════════ */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Five things to remember", {
    x: 0.9, y: 0.65, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, margin: 0,
  });

  const pts = [
    ["One file, one script, rebuilt every run", "Nothing is stored between runs, so nothing can go stale unnoticed."],
    ["No peeking", "There are 28 separate models because each forecast distance may legally see a different amount of history."],
    ["Half the model is holidays", "And Chinese New Year gets its own correction, because it cannot be learned from one or two examples."],
    ["Typhoons are a human decision", "The model is never asked to guess something it cannot see. That is a feature, not a gap."],
    ["Measure before believing", "Most sensible-sounding improvements tested here made things worse. A result from a single event is not a result."],
  ];
  let y = 1.7;
  pts.forEach(([t, b], i) => {
    circle(s, 0.9, y + 0.05, 0.55, String(i + 1), GOLD, NAVY, 18);
    s.addText(t, { x: 1.75, y, w: 10.6, h: 0.38, fontFace: HEAD, fontSize: 18, bold: true, color: WHITE, margin: 0 });
    s.addText(b, { x: 1.75, y: y + 0.38, w: 10.6, h: 0.42, fontFace: BODY, fontSize: 13, color: "AEC3D6", margin: 0 });
    y += 1.02;
  });
  s.addNotes("Close on the measurement discipline point — it is what keeps this model honest.");
}

pres.writeFile({ fileName: path.join(__dirname, "Demand_Forecast_Walkthrough.pptx") })
  .then(f => console.log("wrote", f));
