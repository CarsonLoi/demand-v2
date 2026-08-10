// build_deck.js — generates the Demand Forecast model presentation.
// Run:  node docs/build_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const OUT = path.join(__dirname, "Demand_Forecast_Model.pptx");

// ── Palette ───────────────────────────────────────────────────────────────
const NAVY = "0F2942";
const NAVY_MID = "1A3D5C";
const TEAL = "2E8BA8";
const TEAL_LT = "8FC4D6";
const GOLD = "E8B547";
const OFFWHITE = "F4F7FA";
const WHITE = "FFFFFF";
const INK = "23313F";
const GREY = "5F7183";
const GREEN = "2FA87C";
const RED = "C74B4B";

const HFONT = "Cambria";
const BFONT = "Calibri";

const W = 13.333, H = 7.5;
const M = 0.62;                    // page margin

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Analytics";
pres.title = "Daily Casino Demand Forecast";

// ── Helpers ───────────────────────────────────────────────────────────────
function shadow() {          // fresh object every call (pptxgenjs mutates)
  return { type: "outer", angle: 90, blur: 8, offset: 2, color: "0A1A28", opacity: 0.16 };
}

function darkSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  return s;
}
function lightSlide() {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  return s;
}

// Standard content-slide title (light background)
function title(s, text, kicker) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: M, y: 0.34, w: 11, h: 0.26, margin: 0,
      fontFace: BFONT, fontSize: 11.5, bold: true, color: TEAL, charSpacing: 2,
    });
  }
  s.addText(text, {
    x: M, y: kicker ? 0.62 : 0.5, w: W - 2 * M, h: 0.7, margin: 0,
    fontFace: HFONT, fontSize: 30, bold: true, color: NAVY,
  });
}

// Icon-style numbered circle
function circle(s, x, y, d, fill, label, labelColor) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: d, h: d, fill: { color: fill }, line: { color: fill },
  });
  s.addText(label, {
    x, y, w: d, h: d, margin: 0, align: "center", valign: "middle",
    fontFace: BFONT, fontSize: d > 0.5 ? 15 : 12, bold: true,
    color: labelColor || WHITE,
  });
}

// Soft card (no edge stripes)
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.06,
    fill: { color: fill || OFFWHITE }, line: { color: "E2E9F0", width: 1 },
    shadow: shadow(),
  });
}

// Big number callout
function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.82, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 42, bold: true, color: color || NAVY,
  });
  s.addText(label, {
    x, y: y + 0.84, w, h: 0.5, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 11.5, color: GREY,
  });
}

function bullets(s, items, x, y, w, h, size) {
  s.addText(
    items.map((t, i) => ({
      text: t, options: { bullet: true, breakLine: i !== items.length - 1 },
    })),
    {
      x, y, w, h, margin: 0, fontFace: BFONT, fontSize: size || 14,
      color: INK, lineSpacing: (size || 14) * 1.45, paraSpaceAfter: 7,
    }
  );
}

function sectionDivider(num, kicker, headline, sub) {
  const s = darkSlide();
  s.addText(num, {
    x: M, y: 2.25, w: 2, h: 1.1, margin: 0,
    fontFace: HFONT, fontSize: 68, bold: true, color: GOLD,
  });
  s.addText(kicker.toUpperCase(), {
    x: M + 1.65, y: 2.45, w: 9, h: 0.3, margin: 0,
    fontFace: BFONT, fontSize: 12, bold: true, color: TEAL_LT, charSpacing: 2.5,
  });
  s.addText(headline, {
    x: M + 1.6, y: 2.75, w: 10.5, h: 0.9, margin: 0,
    fontFace: HFONT, fontSize: 34, bold: true, color: WHITE,
  });
  if (sub) {
    s.addText(sub, {
      x: M + 1.65, y: 3.68, w: 9.6, h: 0.6, margin: 0,
      fontFace: BFONT, fontSize: 14, color: TEAL_LT,
    });
  }
  return s;
}

function footer(s, txt) {
  s.addText(txt, {
    x: M, y: H - 0.46, w: W - 2 * M, h: 0.26, margin: 0,
    fontFace: BFONT, fontSize: 9.5, color: "9AA9B8", italic: true,
  });
}

// ══════════════════════════════════════════════════════════════════════════
// 1 — TITLE
// ══════════════════════════════════════════════════════════════════════════
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: 9.5, y: -1.7, w: 6.2, h: 6.2, fill: { color: NAVY_MID }, line: { color: NAVY_MID },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.1, y: 3.6, w: 3.4, h: 3.4, fill: { color: TEAL }, line: { color: TEAL },
  });
  s.addText("PREDICTIVE ANALYTICS  ·  GAMING OPERATIONS", {
    x: M, y: 1.75, w: 9, h: 0.3, margin: 0,
    fontFace: BFONT, fontSize: 12, bold: true, color: GOLD, charSpacing: 2.5,
  });
  s.addText("Daily Casino Demand Forecast", {
    x: M, y: 2.2, w: 9.2, h: 1.5, margin: 0,
    fontFace: HFONT, fontSize: 46, bold: true, color: WHITE,
  });
  s.addText(
    "A 28-day, machine-learned demand forecast — and the planning pipeline it drives, " +
    "from patron hours to dealer shifts.",
    { x: M, y: 3.75, w: 8.3, h: 0.9, margin: 0, fontFace: BFONT, fontSize: 15.5, color: TEAL_LT }
  );
  s.addText("Model · Methodology · Validation · Operations", {
    x: M, y: 5.15, w: 8, h: 0.32, margin: 0,
    fontFace: BFONT, fontSize: 12.5, color: "8FA6BA",
  });
  s.addNotes(
    "Purpose: explain the full demand-forecast system to both management and technical " +
    "stakeholders. Management: slides 2-7 and 21-25. Technical: slides 8-20."
  );
}

// ══════════════════════════════════════════════════════════════════════════
// 2 — EXECUTIVE SUMMARY
// ══════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  title(s, "Executive summary", "At a glance");

  const items = [
    { v: "28", l: "days forecast\nahead, daily", c: NAVY },
    { v: "4.3%", l: "forecast error\n(2026 YTD)", c: GREEN },
    { v: "-53%", l: "error vs. naive\nbaseline", c: TEAL },
    { v: "170", l: "engineered\nfeatures", c: NAVY },
  ];
  items.forEach((it, i) => {
    const x = M + i * 3.08;
    card(s, x, 1.62, 2.85, 1.72);
    stat(s, x, 1.82, 2.85, it.v, it.l, it.c);
  });

  card(s, M, 3.66, 5.95, 3.16, OFFWHITE);
  s.addText("What it does", {
    x: M + 0.3, y: 3.9, w: 5.4, h: 0.35, margin: 0,
    fontFace: HFONT, fontSize: 17, bold: true, color: NAVY,
  });
  bullets(s, [
    "Predicts daily patron hours 28 days ahead, refreshed every day",
    "Outputs a median (P50) plus a P10–P90 planning band",
    "Feeds table-count, dealer-headcount and shift-assignment planning",
    "Runs unattended in ~3 minutes; ~35 min for the highest-accuracy mode",
  ], M + 0.3, 4.32, 5.4, 2.3, 13);

  card(s, M + 6.25, 3.66, 5.95, 3.16, OFFWHITE);
  s.addText("Why it matters", {
    x: M + 6.55, y: 3.9, w: 5.4, h: 0.35, margin: 0,
    fontFace: HFONT, fontSize: 17, bold: true, color: NAVY,
  });
  bullets(s, [
    "Replaces judgement-based planning with an auditable, repeatable number",
    "Cuts average daily mis-allocation from ~630 to ~300 patron hours",
    "Halves Chinese New Year error — the year's single costliest planning week",
    "Every run is archived by date — forecasts are traceable after the fact",
  ], M + 6.55, 4.32, 5.4, 2.3, 13);

  footer(s, "Accuracy measured across 2026 year to date (1 Jan – 27 May, 147 days), re-forecasting weekly — the cadence the model is actually run at.");
  s.addNotes("Lead with -53% vs naive. 4.3% is the full-year operational number and includes Chinese New Year. A single CNY-free month scores ~3.4% - that better number is real but not representative, so it is not the headline.");
}

// ══════════════════════════════════════════════════════════════════════════
// 2b — HOW IT WORKS, IN ONE PICTURE  (executive overview)
// ══════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  title(s, "How it works, in one picture", "Overview");

  const steps = [
    { n: "1", t: "Demand history", d: "879 days of daily patron\nhours and table counts", c: NAVY },
    { n: "2", t: "Feature engineering", d: "170 signals: weekday, holiday\nwindows, trends, momentum", c: TEAL },
    { n: "3", t: "28 models", d: "One per forecast day, so each\nuses only data it would have had", c: TEAL },
    { n: "4", t: "Holiday correction", d: "Chinese New Year adjusted\nagainst last year's pattern", c: GOLD },
    { n: "5", t: "28-day forecast", d: "Median plus a planning band,\nrefreshed every run", c: GREEN },
  ];

  const cw = 2.22, gap = 0.16;
  steps.forEach((st, i) => {
    const x = M + i * (cw + gap);
    card(s, x, 1.68, cw, 2.62);
    circle(s, x + cw / 2 - 0.22, 1.92, 0.44, st.c, st.n);
    s.addText(st.t, {
      x: x + 0.12, y: 2.52, w: cw - 0.24, h: 0.42, margin: 0, align: "center",
      fontFace: HFONT, fontSize: 13, bold: true, color: NAVY,
    });
    s.addText(st.d, {
      x: x + 0.12, y: 2.96, w: cw - 0.24, h: 1.1, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 10.3, color: GREY,
    });
    if (i < steps.length - 1) {
      s.addText("→", {
        x: x + cw + 0.005, y: 2.78, w: gap, h: 0.3, margin: 0, align: "center",
        fontFace: BFONT, fontSize: 15, bold: true, color: TEAL_LT,
      });
    }
  });

  card(s, M, 4.62, 5.95, 2.14, OFFWHITE);
  s.addText("What makes it trustworthy", {
    x: M + 0.3, y: 4.84, w: 5.4, h: 0.35, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY,
  });
  bullets(s, [
    "A forecast never sees data that did not exist yet",
    "Retrains from scratch each run — nothing to drift",
    "Every claim in this deck is a measured backtest",
  ], M + 0.3, 5.24, 5.4, 1.4, 12);

  card(s, M + 6.25, 4.62, 5.95, 2.14, OFFWHITE);
  s.addText("What it does not do", {
    x: M + 6.55, y: 4.84, w: 5.4, h: 0.35, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY,
  });
  bullets(s, [
    "Predict typhoon closures — a human applies that call",
    "Replace judgement on Chinese New Year staffing",
    "Guarantee its uncertainty band; treat it as relative",
  ], M + 6.55, 5.24, 5.4, 1.4, 12);

  footer(s, "Sections 1–6 that follow expand each of these five stages in detail.");
  s.addNotes("This is the slide to leave up if you only have five minutes. Everything after it is elaboration.");
}

// ══════════════════════════════════════════════════════════════════════════
// 2c — WHERE THE ERROR IS  (executive overview)
// ══════════════════════════════════════════════════════════════════════════
{
  const s = lightSlide();
  title(s, "Almost all remaining error sits in one week", "Where we stand");
  s.addText(
    "Ordinary trading days are forecast tightly. Chinese New Year is the outlier — and this year it was cut by half.",
    { x: M, y: 1.3, w: 11.6, h: 0.3, margin: 0, fontFace: BFONT, fontSize: 13.5, color: GREY });

  const cols = [
    { t: "Ordinary days", n: "3.8%", sub: "129 of 147 days", c: GREEN,
      d: "Well inside the tolerance that floor and roster planning needs." },
    { t: "Chinese New Year", n: "7.3%", sub: "18 of 147 days", c: GOLD,
      d: "Was 14.2% before this year's correction — the single biggest improvement made." },
    { t: "All days combined", n: "4.3%", sub: "147 days, 2026 YTD", c: NAVY,
      d: "The number to quote. It includes the hard week rather than excluding it." },
  ];
  cols.forEach((c2, i) => {
    const x = M + i * 4.02;
    card(s, x, 1.86, 3.78, 2.5);
    s.addText(c2.t, { x: x + 0.22, y: 2.06, w: 3.34, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY });
    s.addText(c2.n, { x: x + 0.22, y: 2.44, w: 3.34, h: 0.86, margin: 0,
      fontFace: HFONT, fontSize: 40, bold: true, color: c2.c });
    s.addText(c2.sub, { x: x + 0.22, y: 3.3, w: 3.34, h: 0.3, margin: 0,
      fontFace: BFONT, fontSize: 11, color: GREY });
    s.addText(c2.d, { x: x + 0.22, y: 3.62, w: 3.34, h: 0.7, margin: 0,
      fontFace: BFONT, fontSize: 10.5, color: INK });
  });

  // CNY before/after bar
  card(s, M, 4.62, 11.58, 2.14, "FBF6EA");
  s.addText("Chinese New Year, before and after the correction", {
    x: M + 0.32, y: 4.82, w: 6.6, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });

  // barW sized so the longest bar + its value label clears the explanatory
  // text block at x = M + 7.3 (see geometry note: 14.16% is the max value).
  const barX = M + 0.32, barW = 5.0;
  [{ l: "Before", v: 14.16, c: RED, y: 5.3 },
   { l: "After", v: 7.29, c: GREEN, y: 5.86 }].forEach((b) => {
    s.addText(b.l, { x: barX, y: b.y, w: 0.9, h: 0.32, margin: 0,
      fontFace: BFONT, fontSize: 11.5, bold: true, color: INK });
    s.addShape(pres.ShapeType.roundRect, {
      x: barX + 0.95, y: b.y + 0.04, w: barW * (b.v / 16), h: 0.26, rectRadius: 0.03,
      fill: { color: b.c }, line: { color: b.c },
    });
    s.addText(b.v.toFixed(2) + "%", {
      x: barX + 1.02 + barW * (b.v / 16), y: b.y, w: 1.0, h: 0.32, margin: 0,
      fontFace: BFONT, fontSize: 11.5, bold: true, color: b.c });
  });

  s.addText(
    "The correction matches each CNY day to the same position in last year's CNY — " +
    "not the same calendar date, because the holiday moves by up to three weeks. " +
    "It touches only CNY days; every other day is left exactly as the model produced it.",
    { x: M + 7.3, y: 5.16, w: 4.1, h: 1.4, margin: 0,
      fontFace: BFONT, fontSize: 10.8, color: GREY });

  footer(s, "2026 year to date, re-forecast weekly. The correction weight was calibrated on a prior year and applied unchanged — it was not tuned on the period being scored.");
  s.addNotes("If asked 'why is 4.3% worse than the 2.5% we saw before' - the earlier figure was a single month with no Chinese New Year in it. This slide is the like-for-like full-year view.");
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 1
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("01", "The business problem", "Staffing a floor against\nan unknown tomorrow",
  "Why a forecast is the foundation of every downstream planning decision.");

// ── 4 — The decision we are supporting
{
  const s = lightSlide();
  title(s, "Every day, the floor makes an irreversible bet", "The decision");

  s.addText(
    "Dealer rosters are locked weeks in advance. Tables are opened and closed on the day. " +
    "Both decisions are made before demand is known.",
    { x: M, y: 1.5, w: 11.6, h: 0.55, margin: 0, fontFace: BFONT, fontSize: 15, color: GREY }
  );

  const cols = [
    { t: "Too few tables", c: RED, pts: [
      "Patrons wait, or walk to a competitor",
      "Revenue lost that cannot be recovered",
      "Service perception damaged on peak days",
    ]},
    { t: "Too many tables", c: GOLD, pts: [
      "Dealers idle on paid shifts",
      "Wage cost with no matching revenue",
      "Floor looks empty — a second-order deterrent",
    ]},
    { t: "The right number", c: GREEN, pts: [
      "Capacity matched to expected demand",
      "Uncertainty made explicit via P10–P90",
      "Consistent, auditable, repeatable",
    ]},
  ];
  cols.forEach((col, i) => {
    const x = M + i * 4.0;
    card(s, x, 2.25, 3.72, 3.5);
    circle(s, x + 0.28, 2.55, 0.42, col.c, String(i + 1));
    s.addText(col.t, {
      x: x + 0.82, y: 2.57, w: 2.7, h: 0.4, margin: 0,
      fontFace: HFONT, fontSize: 16, bold: true, color: NAVY,
    });
    bullets(s, col.pts, x + 0.28, 3.2, 3.2, 2.3, 12.5);
  });

  s.addText(
    "Demand swings 30–60% around major holidays — the periods where mis-staffing is most expensive.",
    { x: M, y: 6.0, w: 11.6, h: 0.4, margin: 0, fontFace: BFONT, fontSize: 13.5,
      italic: true, color: TEAL, align: "center" }
  );
  s.addNotes("Frame the problem as a bet made under uncertainty. The model does not remove the uncertainty — it measures it.");
}

// ── 5 — Cost of error
{
  const s = lightSlide();
  title(s, "What forecast error costs", "Quantifying the gap");

  s.addText(
    "On an average day of ~6,900 patron hours, every percentage point of forecast error " +
    "is ~69 patron hours of capacity placed in the wrong hour or not placed at all.",
    { x: M, y: 1.48, w: 11.6, h: 0.6, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY }
  );

  // Comparison: naive vs model
  card(s, M, 2.28, 5.6, 2.35, OFFWHITE);
  s.addText("Naive planning (same weekday, last year)", {
    x: M + 0.32, y: 2.5, w: 5, h: 0.32, margin: 0,
    fontFace: BFONT, fontSize: 12.5, bold: true, color: GREY });
  s.addText("9.14%", { x: M + 0.32, y: 2.84, w: 2.4, h: 0.72, margin: 0,
    fontFace: HFONT, fontSize: 40, bold: true, color: RED });
  s.addText("≈ 630 patron hours\nmis-allocated per day", {
    x: M + 2.75, y: 2.92, w: 2.6, h: 0.7, margin: 0,
    fontFace: BFONT, fontSize: 12.5, color: INK });

  card(s, M + 6.0, 2.28, 5.6, 2.35, "EAF6F1");
  s.addText("Model (with Chinese New Year correction)", {
    x: M + 6.32, y: 2.5, w: 5, h: 0.32, margin: 0,
    fontFace: BFONT, fontSize: 12.5, bold: true, color: GREY });
  s.addText("4.26%", { x: M + 6.32, y: 2.84, w: 2.4, h: 0.72, margin: 0,
    fontFace: HFONT, fontSize: 40, bold: true, color: GREEN });
  s.addText("≈ 300 patron hours\nmis-allocated per day", {
    x: M + 8.75, y: 2.92, w: 2.6, h: 0.7, margin: 0,
    fontFace: BFONT, fontSize: 12.5, color: INK });

  card(s, M, 4.85, 11.6, 1.55, NAVY);
  s.addText("≈ 330 patron hours per day of mis-allocation avoided", {
    x: M + 0.4, y: 5.08, w: 11, h: 0.5, margin: 0,
    fontFace: HFONT, fontSize: 24, bold: true, color: WHITE });
  s.addText(
    "Equivalent to roughly 2–4 table-shifts per day, depending on seats per table and utilisation — " +
    "recovered every day, at no marginal cost once the model is running.",
    { x: M + 0.4, y: 5.62, w: 10.8, h: 0.55, margin: 0,
      fontFace: BFONT, fontSize: 13, color: TEAL_LT });

  footer(s, "Patron-hour figures derived from mean daily demand × MAPE. Table-shift conversion depends on local seats-per-table and utilisation assumptions.");
  s.addNotes("Be explicit that the table-shift conversion is an assumption, not a measured figure. The patron-hour numbers are direct from the model.");
}

// ── 6 — Pipeline overview
{
  const s = lightSlide();
  title(s, "From history to headcount", "The end-to-end pipeline");

  const steps = [
    { n: "1", t: "Daily demand\nforecast", d: "28-day patron\nhours, P10/P50/P90", c: TEAL, built: true },
    { n: "2", t: "Hourly\nsplit", d: "Distribute the day\nacross 24 hours", c: TEAL, built: true },
    { n: "3", t: "Table count\nper hour", d: "Erlang-B + newsvendor\nservice level", c: GOLD, built: false },
    { n: "4", t: "Dealer\nheadcount", d: "Relief factor ×\nabsence buffer", c: GOLD, built: false },
    { n: "5", t: "Shift\nassignment", d: "CP-SAT optimiser\nper table", c: GOLD, built: false },
  ];
  const bw = 2.16, gap = 0.28;
  steps.forEach((st, i) => {
    const x = M + i * (bw + gap);
    card(s, x, 2.1, bw, 2.55, st.built ? "EAF3F7" : OFFWHITE);
    circle(s, x + bw / 2 - 0.24, 2.32, 0.48, st.c, st.n);
    s.addText(st.t, {
      x: x + 0.12, y: 2.95, w: bw - 0.24, h: 0.62, margin: 0, align: "center",
      fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY });
    s.addText(st.d, {
      x: x + 0.12, y: 3.62, w: bw - 0.24, h: 0.7, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 11, color: GREY });
    s.addText(st.built ? "IN PRODUCTION" : "DESIGNED", {
      x: x + 0.12, y: 4.3, w: bw - 0.24, h: 0.24, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 8.5, bold: true,
      color: st.built ? GREEN : GOLD, charSpacing: 1 });
    if (i < steps.length - 1) {
      s.addShape(pres.ShapeType.rightArrow, {
        x: x + bw + 0.03, y: 3.2, w: 0.22, h: 0.2,
        fill: { color: "C3D2DE" }, line: { color: "C3D2DE" } });
    }
  });

  card(s, M, 5.05, 11.6, 1.32, OFFWHITE);
  s.addText("This deck covers all five stages — stages 1 and 2 are built and running; stages 3–5 are designed and specified, awaiting operational data.", {
    x: M + 0.35, y: 5.32, w: 11, h: 0.8, margin: 0,
    fontFace: BFONT, fontSize: 13.5, color: INK });
  s.addNotes("Set expectations early: the forecast is live, the downstream optimisation is designed but needs per-table operational data.");
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 2
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("02", "Data & feature engineering", "What the model sees",
  "171 engineered features from four data sources — and the leakage rule that shapes them all.");

// ── 8 — Data sources
{
  const s = lightSlide();
  title(s, "Four inputs, one contract", "Data sources");

  const rows = [
    { n: "Demand history", f: "rawdata.csv", d: "Daily patron hours + floor table count. 879 days (Jan 2024 – May 2026).", st: "Core", c: TEAL },
    { n: "Holiday calendar", f: "_shared.py", d: "Anchor dates + demand windows for 9 holidays, plus Mainland China working-day calendar.", st: "Core", c: TEAL },
    { n: "Typhoon signals", f: "typhoons.csv", d: "HKO Signal 8+ events, scraped automatically. Forward-looking (from weather forecast).", st: "Active", c: GREEN },
    { n: "Hotel pacing", f: "reservations.csv", d: "Rooms on the books by stay date and lead time. Wired and leakage-guarded.", st: "Optional", c: GOLD },
  ];
  rows.forEach((r, i) => {
    const y = 1.55 + i * 1.24;
    card(s, M, y, 11.6, 1.12);
    circle(s, M + 0.3, y + 0.33, 0.46, r.c, String(i + 1));
    s.addText(r.n, { x: M + 0.92, y: y + 0.18, w: 2.6, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 15.5, bold: true, color: NAVY });
    s.addText(r.f, { x: M + 0.92, y: y + 0.55, w: 2.6, h: 0.3, margin: 0,
      fontFace: "Courier New", fontSize: 10.5, color: TEAL });
    s.addText(r.d, { x: M + 3.7, y: y + 0.3, w: 6.5, h: 0.6, margin: 0,
      fontFace: BFONT, fontSize: 12.5, color: INK });
    s.addText(r.st, { x: M + 10.3, y: y + 0.38, w: 1.1, h: 0.32, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 10.5, bold: true, color: r.c });
  });

  footer(s, "Demand data is the only mandatory input. The model degrades gracefully — a missing optional source becomes a neutral feature, never an error.");
}

// ── 9 — Feature families (donut infographic)
{
  const s = lightSlide();
  title(s, "Half the model is about holidays", "Feature engineering");

  const donut = path.join(ROOT, "docs", "assets", "feature_donut.png");
  if (fs.existsSync(donut)) {
    s.addImage({ path: donut, x: M - 0.2, y: 1.5, w: 7.2, h: 5.0 });
  }

  const pts = [
    { n: "87", t: "Holiday features", d: "Flags, day-offsets, distance to next holiday, holiday × weekday", c: GOLD },
    { n: "46", t: "History features", d: "Lags, rolling statistics, moving averages — all horizon-gated", c: TEAL },
    { n: "19", t: "Calendar features", d: "Weekday, month, cyclical encodings", c: NAVY },
    { n: "19", t: "Everything else", d: "Mainland blocks, interactions, typhoon", c: GREY },
  ];
  pts.forEach((p, i) => {
    const y = 1.62 + i * 1.24;
    card(s, M + 7.2, y, 4.4, 1.1);
    s.addText(p.n, { x: M + 7.36, y: y + 0.24, w: 0.9, h: 0.6, margin: 0, align: "center",
      fontFace: HFONT, fontSize: 26, bold: true, color: p.c });
    s.addText(p.t, { x: M + 8.32, y: y + 0.2, w: 3.1, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 14, bold: true, color: NAVY });
    s.addText(p.d, { x: M + 8.32, y: y + 0.54, w: 3.1, h: 0.46, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
  });

  footer(s, "Holiday periods are a small share of the calendar but the largest share of forecast risk — the feature set reflects that.");
  s.addNotes("Let the donut do the talking. The one number to say aloud: half the feature set exists to handle holidays.");
}

// ── 10 — Horizon-aware lags
{
  const s = lightSlide();
  title(s, "The rule that shapes everything: no peeking", "Horizon-aware features");

  s.addText(
    "A forecast made 14 days out cannot use data from the last 13 days — that data does not exist yet. " +
    "Every lag, rolling window and moving average is therefore gated by the forecast horizon.",
    { x: M, y: 1.5, w: 11.6, h: 0.62, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY }
  );

  const ex = [
    { h: "1 day ahead", ok: "lag_2, lag_3, lag_7, rolling_mean_7 …", no: "lag_1 (today is not closed yet)" },
    { h: "7 days ahead", ok: "lag_8, lag_14, rolling means from day 8 back", no: "lag_2 … lag_7" },
    { h: "28 days ahead", ok: "lag_28, lag_35, lag_365, lag_728", no: "everything more recent than 28 days" },
  ];
  ex.forEach((e, i) => {
    const y = 2.32 + i * 1.24;
    card(s, M, y, 11.6, 1.12);
    s.addText(e.h, { x: M + 0.3, y: y + 0.36, w: 2.0, h: 0.4, margin: 0,
      fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
    circle(s, M + 2.45, y + 0.36, 0.36, GREEN, "✓");
    s.addText(e.ok, { x: M + 2.95, y: y + 0.2, w: 4.35, h: 0.36, margin: 0,
      fontFace: "Courier New", fontSize: 10.5, color: INK, valign: "middle" });
    circle(s, M + 2.45, y + 0.62, 0.36, RED, "✕");
    s.addText(e.no, { x: M + 2.95, y: y + 0.6, w: 4.35, h: 0.36, margin: 0,
      fontFace: BFONT, fontSize: 11, color: GREY, valign: "middle" });
    s.addShape(pres.ShapeType.roundRect, {
      x: M + 7.65, y: y + 0.24, w: 3.6, h: 0.64, rectRadius: 0.05,
      fill: { color: i === 2 ? "FBF1DC" : "EAF3F7" }, line: { color: "FFFFFF" } });
    s.addText(i === 2 ? "Long-memory signals only" : "Recent momentum available", {
      x: M + 7.7, y: y + 0.24, w: 3.5, h: 0.64, margin: 0, align: "center", valign: "middle",
      fontFace: BFONT, fontSize: 11.5, bold: true, color: i === 2 ? "8A6516" : TEAL });
  });

  card(s, M, 6.12, 11.6, 0.72, NAVY);
  s.addText("Consequence: we train 28 separate models — one per forecast horizon — each with its own legal feature set.", {
    x: M + 0.35, y: 6.28, w: 11, h: 0.4, margin: 0,
    fontFace: BFONT, fontSize: 13.5, bold: true, color: WHITE });
  s.addNotes("This is the single most important technical slide. It explains both the architecture and why accuracy decays with horizon.");
}

// ── 11 — Holiday modelling
{
  const s = lightSlide();
  title(s, "Holidays get special treatment", "Feature deep-dive");

  s.addText("Holiday periods carry the largest demand swings and the largest forecast risk. Three mechanisms handle them.",
    { x: M, y: 1.5, w: 11.6, h: 0.4, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  const mech = [
    { t: "Windows, not days", d: "Each holiday has a demand window, not a single date — Chinese New Year runs −7 to +10 days; Dragon Boat −2 to +1 to capture the soft pre-holiday lull.", c: TEAL },
    { t: "Per-offset flags", d: "Within a window, each day-offset gets its own feature, so the model learns that day +1 behaves differently from day +4 rather than averaging them.", c: TEAL },
    { t: "Fixed vs lunar split", d: "Fixed-date holidays (Labour Day, Golden Week) align year-on-year, so last year's value is informative. Lunar holidays move by weeks — Chinese New Year gets a dedicated correction that matches position within the holiday, not calendar date.", c: GOLD },
  ];
  mech.forEach((m, i) => {
    const y = 2.06 + i * 1.44;
    card(s, M, y, 11.6, 1.3);
    circle(s, M + 0.32, y + 0.42, 0.46, m.c, String(i + 1));
    s.addText(m.t, { x: M + 0.98, y: y + 0.2, w: 3.1, h: 0.38, margin: 0,
      fontFace: HFONT, fontSize: 15.5, bold: true, color: NAVY });
    s.addText(m.d, { x: M + 0.98, y: y + 0.58, w: 10.3, h: 0.62, margin: 0,
      fontFace: BFONT, fontSize: 12.5, color: INK });
  });

  card(s, M, 6.4, 11.6, 0.6, OFFWHITE);
  s.addText("Modelled holidays: Chinese New Year · Golden Week · Labour Day · Mid-Autumn · Dragon Boat · Ching Ming · Christmas · New Year · Easter", {
    x: M + 0.3, y: 6.52, w: 11, h: 0.36, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 11.5, color: GREY });
}

// ── 12 — External signals
{
  const s = lightSlide();
  title(s, "Two external signals — and why one was removed", "Weather & hotel pacing");

  card(s, M, 1.55, 5.7, 4.5);
  circle(s, M + 0.35, 1.85, 0.5, GREY, "T");
  s.addText("Typhoon Signal 8+", { x: M + 1.0, y: 1.9, w: 4.4, h: 0.4, margin: 0,
    fontFace: HFONT, fontSize: 18, bold: true, color: NAVY });
  s.addText("MANUAL", { x: M + 4.45, y: 1.62, w: 1.05, h: 0.28, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 9.5, bold: true, color: RED, charSpacing: 1 });
  bullets(s, [
    "Removed from the model in 2026 — it could not be learned reliably",
    "Nine typhoon days on record; impact ranged from +10% to −87%",
    "The driver is whether operations SUSPEND, a business decision the storm signal does not predict",
    "Now applied by a person, as an explicit override, once the closure call is known",
  ], M + 0.35, 2.55, 5.0, 3.3, 12);

  card(s, M + 5.95, 1.55, 5.65, 4.5);
  circle(s, M + 6.3, 1.85, 0.5, GOLD, "H");
  s.addText("Hotel on-the-books", { x: M + 6.95, y: 1.9, w: 4.4, h: 0.4, margin: 0,
    fontFace: HFONT, fontSize: 18, bold: true, color: NAVY });
  s.addText("OPTIONAL", { x: M + 10.4, y: 1.62, w: 1.1, h: 0.28, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 9.5, bold: true, color: GOLD, charSpacing: 1 });
  bullets(s, [
    "Rooms booked for each future stay date, by lead time",
    "Three features: current pace, 7-day pick-up, and pace vs. same-weekday norm",
    "Strongest signal at 1–14 days out; naturally fades at longer horizons",
    "Snapshot filtering ensures a run never sees bookings made after its run date",
  ], M + 6.3, 2.55, 5.0, 3.3, 12);

  card(s, M, 6.25, 11.6, 0.72, NAVY);
  s.addText("Design principle: every feature must earn its place in backtests. Fewer, stronger signals beat many weak ones.", {
    x: M + 0.35, y: 6.42, w: 11, h: 0.4, margin: 0,
    fontFace: BFONT, fontSize: 13.5, bold: true, color: WHITE });
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 3
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("03", "Methodology", "How the forecast\nis actually produced",
  "Per-horizon models, a six-model ensemble, and a validation design that cannot cheat.");

// ── 14 — Architecture
{
  const s = lightSlide();
  title(s, "Twenty-eight models, not one", "Core architecture");

  s.addText("Because each horizon has a different legal feature set, each horizon gets its own model — trained, tuned and evaluated independently.",
    { x: M, y: 1.5, w: 11.6, h: 0.45, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  // Horizon strip
  const n = 14, bw2 = 0.76, gp = 0.06;
  for (let i = 0; i < n; i++) {
    const x = M + i * (bw2 + gp);
    const shade = i < 4 ? "1E6E8C" : i < 9 ? "4E9CB5" : "9CC6D6";
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.2, w: bw2, h: 1.0, rectRadius: 0.05,
      fill: { color: shade }, line: { color: shade } });
    s.addText("h" + (i + 1), { x, y: 2.2, w: bw2, h: 1.0, margin: 0,
      align: "center", valign: "middle", fontFace: BFONT, fontSize: 11, bold: true, color: WHITE });
  }
  s.addText("…  through  h28", { x: M + n * (bw2 + gp) + 0.05, y: 2.2, w: 1.4, h: 1.0, margin: 0,
    valign: "middle", fontFace: BFONT, fontSize: 12, bold: true, color: GREY });

  s.addText("Short horizons — rich recent history", { x: M, y: 3.3, w: 3.4, h: 0.3, margin: 0,
    fontFace: BFONT, fontSize: 11, color: TEAL, bold: true });
  s.addText("Long horizons — long-memory signals only", { x: M + 7.0, y: 3.3, w: 4.4, h: 0.3, margin: 0,
    fontFace: BFONT, fontSize: 11, color: GREY, bold: true, align: "right" });

  const feats = [
    { t: "Sample weighting", d: "Recent days count more (240-day half-life). Holiday-window days carry 3× weight so rare, high-stakes periods are not drowned out." },
    { t: "Trained from scratch, every run", d: "No stored model artefacts to drift or go stale — each run retrains on all data available at its run date." },
    { t: "Non-negative, quantile-consistent output", d: "Predictions are floored at zero and ordered so P10 ≤ P50 ≤ P90 always holds." },
  ];
  feats.forEach((f, i) => {
    const y = 3.82 + i * 1.02;
    card(s, M, y, 11.6, 0.9);
    circle(s, M + 0.28, y + 0.22, 0.44, NAVY_MID, String(i + 1));
    s.addText(f.t, { x: M + 0.9, y: y + 0.12, w: 3.3, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 14, bold: true, color: NAVY });
    s.addText(f.d, { x: M + 0.9, y: y + 0.44, w: 10.4, h: 0.38, margin: 0,
      fontFace: BFONT, fontSize: 11.8, color: GREY });
  });
}

// ── NEW — What is gradient boosting
{
  const s = lightSlide();
  title(s, "What the model actually is", "Gradient boosting, in plain terms");

  s.addText("Every model in the ensemble belongs to one family: gradient-boosted decision trees. The idea is to build many small, deliberately weak models in sequence — each one correcting what the previous ones got wrong.",
    { x: M, y: 1.46, w: 11.6, h: 0.58, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  const bimg = path.join(ROOT, "docs", "assets", "boosting_stages.png");
  if (fs.existsSync(bimg)) {
    s.addImage({ path: bimg, x: M, y: 2.05, w: 11.6, h: 2.87 });
  }

  card(s, M, 5.08, 11.6, 0.72, NAVY);
  s.addText("Each tree is trained on what the previous ones got wrong. Hundreds of small corrections add up to an accurate forecast.", {
    x: M, y: 5.08, w: 11.6, h: 0.72, margin: 0, align: "center", valign: "middle",
    fontFace: BFONT, fontSize: 13.5, bold: true, color: WHITE });

  const why = [
    { n: "✓", t: "Finds interactions itself", d: "Saturday-inside-Golden-Week is discovered, not coded" },
    { n: "✓", t: "Handles gaps natively", d: "Long horizons have lags deliberately withheld" },
    { n: "✓", t: "Built for small tabular data", d: "Hundreds of rows, hundreds of columns" },
  ];
  why.forEach((w2, i) => {
    const x = M + i * 3.95;
    card(s, x, 5.98, 3.68, 0.95, OFFWHITE);
    circle(s, x + 0.22, 6.2, 0.36, GREEN, w2.n);
    s.addText(w2.t, { x: x + 0.66, y: 6.14, w: 2.85, h: 0.3, margin: 0,
      fontFace: HFONT, fontSize: 12.5, bold: true, color: NAVY });
    s.addText(w2.d, { x: x + 0.66, y: 6.46, w: 2.85, h: 0.36, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
  });
  s.addNotes("Point at the shaded red area shrinking across the three panels — that is the whole idea of boosting, visible.");
}

// ── NEW — The three libraries
{
  const s = lightSlide();
  title(s, "Three implementations, three ways to grow a tree", "LightGBM · XGBoost · CatBoost");

  s.addText("All three do gradient boosting, but they build their trees differently — which is exactly why their mistakes differ, and why blending them helps.",
    { x: M, y: 1.46, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  const libs = [
    { n: "LightGBM", tag: "PRIMARY", c: TEAL,
      how: "Leaf-wise growth — always splits the leaf that reduces error most, producing deep, asymmetric trees.",
      pro: "Fastest to train; handles many features well",
      con: "Can overfit small data if left unconstrained",
      use: "Four of the six ensemble members, and the entire fast daily mode." },
    { n: "XGBoost", tag: "DIVERSITY", c: NAVY_MID,
      how: "Level-wise growth — expands every node one depth at a time, giving balanced, symmetric trees.",
      pro: "Strong, well-tested regularisation",
      con: "Slower; wastes splits on unhelpful branches",
      use: "One ensemble member, using a median (quantile) objective." },
    { n: "CatBoost", tag: "DIVERSITY", c: GOLD,
      how: "Oblivious trees — the same split is applied across a whole level, plus ordered boosting to curb bias.",
      pro: "Resists a subtle overfitting bias others share",
      con: "Weakest of the three on its own here",
      use: "One member, absolute-error loss — valuable for its different failure mode." },
  ];
  libs.forEach((l, i) => {
    const x = M + i * 3.95;
    card(s, x, 2.0, 3.68, 4.35, i === 0 ? "EAF3F7" : OFFWHITE);
    s.addText(l.n, { x: x + 0.24, y: 2.18, w: 2.3, h: 0.4, margin: 0,
      fontFace: HFONT, fontSize: 19, bold: true, color: NAVY });
    s.addText(l.tag, { x: x + 2.4, y: 2.26, w: 1.1, h: 0.26, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 8.5, bold: true, color: l.c, charSpacing: 1 });
    s.addText("How it grows trees", { x: x + 0.24, y: 2.68, w: 3.2, h: 0.24, margin: 0,
      fontFace: BFONT, fontSize: 9.5, bold: true, color: l.c, charSpacing: 0.8 });
    s.addText(l.how, { x: x + 0.24, y: 2.92, w: 3.2, h: 0.78, margin: 0,
      fontFace: BFONT, fontSize: 10.5, color: INK });
    circle(s, x + 0.24, 3.78, 0.28, GREEN, "+");
    s.addText(l.pro, { x: x + 0.6, y: 3.76, w: 2.85, h: 0.34, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
    circle(s, x + 0.24, 4.22, 0.28, RED, "−");
    s.addText(l.con, { x: x + 0.6, y: 4.2, w: 2.85, h: 0.34, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
    s.addShape(pres.ShapeType.roundRect, {
      x: x + 0.24, y: 4.72, w: 3.2, h: 1.42, rectRadius: 0.05,
      fill: { color: WHITE }, line: { color: "E2E9F0", width: 1 } });
    s.addText("Role here", { x: x + 0.38, y: 4.82, w: 2.9, h: 0.24, margin: 0,
      fontFace: BFONT, fontSize: 9.5, bold: true, color: NAVY, charSpacing: 0.8 });
    s.addText(l.use, { x: x + 0.38, y: 5.06, w: 2.92, h: 1.0, margin: 0,
      fontFace: BFONT, fontSize: 10.2, color: INK });
  });

  footer(s, "Hyperparameters are held deliberately conservative — shallow trees, low learning rate, strong regularisation — because each horizon trains on only a few hundred rows.");
}

// ── NEW — Why this family
{
  const s = lightSlide();
  title(s, "Why boosted trees, and not something else", "Model selection");

  s.addText("Tested, not assumed — every alternative was run on this data.",
    { x: M, y: 1.42, w: 11.6, h: 0.34, margin: 0, fontFace: BFONT, fontSize: 14, color: GREY });

  const fimg = path.join(ROOT, "docs", "assets", "family_bars.png");
  if (fs.existsSync(fimg)) {
    s.addImage({ path: fimg, x: M, y: 1.88, w: 7.5, h: 3.05 });
  }

  const notRun = [
    { t: "Classical time series", d: "ARIMA, ETS — cannot absorb 171 features or holiday windows", c: GREY },
    { t: "Neural networks", d: "LSTM, transformers — need far more than 880 days of history", c: GREY },
  ];
  notRun.forEach((nr, i) => {
    const x = M + i * 3.95;
    card(s, x, 5.15, 3.68, 1.0, OFFWHITE);
    circle(s, x + 0.24, 5.36, 0.38, GREY, "—");
    s.addText(nr.t, { x: x + 0.7, y: 5.3, w: 2.8, h: 0.3, margin: 0,
      fontFace: HFONT, fontSize: 13, bold: true, color: NAVY });
    s.addText(nr.d, { x: x + 0.7, y: 5.62, w: 2.8, h: 0.42, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
    s.addText("NOT VIABLE HERE", { x: x + 0.7, y: 5.98, w: 2.8, h: 0.2, margin: 0,
      fontFace: BFONT, fontSize: 8, bold: true, color: GREY, charSpacing: 1 });
  });

  card(s, M + 7.9, 1.88, 3.7, 3.05, "EAF6F1");
  s.addText("Why trees win here", { x: M + 8.14, y: 2.08, w: 3.2, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  bullets(s, [
    "Best measured accuracy",
    "Native missing-value handling",
    "Trains in minutes on a laptop",
    "Six variants disagree usefully — so they blend well",
  ], M + 8.14, 2.5, 3.2, 2.2, 11.5);

  card(s, M + 7.9, 5.15, 3.7, 1.0, NAVY);
  s.addText("2.52%", { x: M + 8.14, y: 5.28, w: 1.5, h: 0.5, margin: 0,
    fontFace: HFONT, fontSize: 26, bold: true, color: WHITE });
  s.addText("best on the\nMay 2026 window", { x: M + 9.7, y: 5.32, w: 1.7, h: 0.5, margin: 0,
    fontFace: BFONT, fontSize: 10.5, color: TEAL_LT });

  footer(s, "MAPE from held-out backtests on this dataset, not published benchmarks. 2.52% is a single CNY-free month; the full-year figure is 4.26% — see Results.");
  s.addNotes("The NeuralProphet number is real - it was run as part of the v2 model comparison. Useful for pre-empting 'why not deep learning'.");
}

// ── 15 — Ensemble
{
  const s = lightSlide();
  title(s, "Six models, six different biases", "The ensemble");

  s.addText("Different algorithms fail differently. Blending them cancels individual quirks — the classic ensemble argument.",
    { x: M, y: 1.5, w: 11.6, h: 0.4, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  const models = [
    { t: "LightGBM L2", d: "Squared-error objective. The workhorse, and the default fast mode.", c: TEAL },
    { t: "LightGBM quantile", d: "Median objective — naturally unbiased, robust to outliers.", c: TEAL },
    { t: "XGBoost", d: "Different split-finding and regularisation path.", c: TEAL },
    { t: "CatBoost (MAE)", d: "Ordered boosting, absolute-error loss. Weakest alone, valuable for diversity.", c: TEAL },
    { t: "Bagged LightGBM", d: "Three seeds averaged — variance reduction.", c: TEAL },
    { t: "Two-stage holiday specialist", d: "A baseline model plus a second model trained only on holiday-window residuals.", c: GOLD },
  ];
  models.forEach((m, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 3.95, y = 2.05 + row * 1.62;
    card(s, x, y, 3.68, 1.44, i === 5 ? "FBF4E4" : OFFWHITE);
    circle(s, x + 0.24, y + 0.24, 0.42, m.c, String(i + 1));
    s.addText(m.t, { x: x + 0.76, y: y + 0.22, w: 2.8, h: 0.46, margin: 0,
      fontFace: HFONT, fontSize: 13.5, bold: true, color: NAVY });
    s.addText(m.d, { x: x + 0.24, y: y + 0.78, w: 3.2, h: 0.58, margin: 0,
      fontFace: BFONT, fontSize: 11, color: GREY });
  });

  card(s, M, 5.42, 11.6, 1.1, NAVY);
  s.addText("The two-stage specialist is the one component built specifically for this problem — it isolates “what is different about holidays” instead of asking one model to learn normal days and holidays at once.", {
    x: M + 0.38, y: 5.62, w: 10.9, h: 0.7, margin: 0,
    fontFace: BFONT, fontSize: 13, color: TEAL_LT });
}

// ── 16 — Blending
{
  const s = lightSlide();
  title(s, "Not all models are best everywhere", "Blending strategy");

  s.addText("Two refinements on top of a plain average. Both are shown here on one real day: Friday 1 May 2026.",
    { x: M, y: 1.42, w: 11.6, h: 0.34, margin: 0, fontFace: BFONT, fontSize: 14, color: GREY });

  // ---- Step 1 : selection, with a concrete lookup table
  card(s, M, 1.88, 5.6, 2.55);
  circle(s, M + 0.28, 2.08, 0.44, TEAL, "1");
  s.addText("Pick the best models per cell", { x: M + 0.9, y: 2.12, w: 4.5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 15.5, bold: true, color: NAVY });
  s.addText("Instead of averaging all six models everywhere, use the best two for each weekday-and-lead-time combination.", {
    x: M + 0.3, y: 2.58, w: 5.05, h: 0.5, margin: 0, fontFace: BFONT, fontSize: 11.5, color: GREY });

  const lookup = [
    ["Weekday · lead", "Models used"],
    ["Friday · 1 day", "XGBoost + CatBoost"],
    ["Friday · 14 days", "LightGBM-bagged + 2-stage"],
    ["Tuesday · 7 days", "LightGBM-L2 + XGBoost"],
  ];
  lookup.forEach((row, r) => {
    const y = 3.14 + r * 0.3;
    if (r === 0) {
      s.addShape(pres.ShapeType.rect, { x: M + 0.3, y: y - 0.02, w: 5.0, h: 0.3,
        fill: { color: "E4EDF3" }, line: { color: "E4EDF3" } });
    }
    s.addText(row[0], { x: M + 0.4, y, w: 1.85, h: 0.28, margin: 0,
      fontFace: BFONT, fontSize: 10.5, bold: r === 0, color: r === 0 ? NAVY : INK });
    s.addText(row[1], { x: M + 2.3, y, w: 2.95, h: 0.28, margin: 0,
      fontFace: BFONT, fontSize: 10.5, bold: r === 0,
      color: r === 0 ? NAVY : TEAL });
  });

  // ---- Step 2 : holiday anchor, worked example
  card(s, M + 6.0, 1.88, 5.6, 2.55);
  circle(s, M + 6.28, 2.08, 0.44, GOLD, "2");
  s.addText("Nudge holidays toward last year", { x: M + 6.9, y: 2.12, w: 4.5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 15.5, bold: true, color: NAVY });
  s.addText("Every model under-shoots holiday peaks. On holiday days we blend in last year's value, grown forward.", {
    x: M + 6.3, y: 2.58, w: 5.05, h: 0.5, margin: 0, fontFace: BFONT, fontSize: 11.5, color: GREY });

  const calc = [
    ["Last year — 1 May 2025", "8,349", GREY],
    ["× demand growth since", "× 1.02", GREY],
    ["= anchor estimate", "8,517", GOLD],
    ["Models alone said", "7,939", GREY],
    ["Blended 80 / 20 →", "8,055", GREEN],
  ];
  calc.forEach((row, r) => {
    const y = 3.10 + r * 0.25;
    s.addText(row[0], { x: M + 6.4, y, w: 3.3, h: 0.24, margin: 0,
      fontFace: BFONT, fontSize: 10.5, color: INK });
    s.addText(row[1], { x: M + 9.75, y, w: 1.5, h: 0.24, margin: 0, align: "right",
      fontFace: BFONT, fontSize: 11, bold: r >= 2, color: row[2] });
  });

  // ---- Outcome strip
  card(s, M, 4.62, 11.6, 1.05, NAVY);
  s.addText("Actual demand on 1 May 2026:  8,653", {
    x: M + 0.4, y: 4.78, w: 5.6, h: 0.38, margin: 0,
    fontFace: HFONT, fontSize: 17, bold: true, color: WHITE });
  s.addText("Models alone were 8.3% low.  With the anchor, 6.9% low — the same correction applied across every holiday.", {
    x: M + 0.4, y: 5.18, w: 10.8, h: 0.34, margin: 0,
    fontFace: BFONT, fontSize: 12, color: TEAL_LT });

  // ---- Progression
  const prog = [
    { t: "Plain average of 6 models", v: "3.30%", c: GREY },
    { t: "+ best-two selection", v: "3.02%", c: TEAL },
    { t: "+ holiday anchor", v: "2.52%", c: GREEN },
  ];
  prog.forEach((p, i) => {
    const x = M + i * 3.95;
    card(s, x, 5.88, 3.68, 0.92, i === 2 ? "EAF6F1" : OFFWHITE);
    s.addText(p.v, { x: x + 0.24, y: 6.0, w: 1.5, h: 0.5, margin: 0,
      fontFace: HFONT, fontSize: 22, bold: true, color: p.c });
    s.addText(p.t, { x: x + 1.8, y: 6.12, w: 1.75, h: 0.42, margin: 0,
      fontFace: BFONT, fontSize: 10.5, color: GREY });
    if (i < 2) {
      s.addShape(pres.ShapeType.rightArrow, { x: x + 3.74, y: 6.24, w: 0.16, h: 0.18,
        fill: { color: "C3D2DE" }, line: { color: "C3D2DE" } });
    }
  });

  footer(s, "This progression is measured on the May 2026 window (no Chinese New Year). Chinese New Year is corrected by a separate mechanism — see the overview and Results sections.");
  s.addNotes("Walk the 1 May 2026 example left to right. The point: selection picks WHICH models, the anchor corrects holiday under-shoot. Both learned from past data only.");
}

// ── 17 — Honest validation
{
  const s = lightSlide();
  title(s, "A validation design that cannot cheat", "Guarding against leakage");

  s.addText("The easiest way to produce an impressive but worthless forecast is to let the model see the answers. Three guards prevent it.",
    { x: M, y: 1.5, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  // two-round diagram
  card(s, M, 2.05, 5.6, 2.65, "EAF3F7");
  s.addText("Round 1 — Calibration", { x: M + 0.3, y: 2.25, w: 5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  bullets(s, [
    "Train on data before the last 180 days",
    "Predict those 180 days — unseen during training",
    "Learn the selection table and anchor weight from these honest errors",
  ], M + 0.3, 2.68, 5.0, 1.8, 12);

  card(s, M + 6.0, 2.05, 5.6, 2.65, "EAF6F1");
  s.addText("Round 2 — Production", { x: M + 6.3, y: 2.25, w: 5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  bullets(s, [
    "Retrain on all available data",
    "Forecast the true future",
    "Apply the recipe learned in Round 1, unchanged",
  ], M + 6.3, 2.68, 5.0, 1.8, 12);

  const guards = [
    { t: "Demand", d: "Training is restricted to dates on or before the run date, even when newer actuals sit in the file." },
    { t: "Hotel pacing", d: "Booking snapshots taken after the run date are dropped before any feature is computed." },
    { t: "Typhoon", d: "Kept for future dates by design — warnings are genuinely available ahead of time. A strict backtest switch exists." },
  ];
  guards.forEach((g, i) => {
    const x = M + i * 3.95;
    card(s, x, 4.95, 3.68, 1.55, OFFWHITE);
    s.addText(g.t, { x: x + 0.25, y: 5.12, w: 3.2, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY });
    s.addText(g.d, { x: x + 0.25, y: 5.48, w: 3.2, h: 0.92, margin: 0,
      fontFace: BFONT, fontSize: 11, color: GREY });
  });

  footer(s, "Because of this design, reported accuracy is what the model would genuinely have achieved in production on those dates.");
  s.addNotes("For a technical audience this is the credibility slide. For management, the message is simply: the numbers are not flattered.");
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 4 — RESULTS
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("04", "Results", "How accurate is it,\nand against what?",
  "Measured against three naive baselines on the same held-out window.");

// ── 19 — Baseline comparison (native chart)
{
  const s = lightSlide();
  title(s, "Every step earns its complexity", "Baseline comparison");

  s.addChart(pres.ChartType.bar, [{
    name: "MAPE (%)",
    labels: ["Naive\n(same weekday,\nlast year)", "Machine-learned\nmodel",
             "+ Chinese New Year\ncorrection"],
    values: [9.14, 5.10, 4.26],
  }], {
    x: M, y: 1.5, w: 8.3, h: 4.55,
    barDir: "col",
    chartColors: [RED, TEAL, GREEN],
    varyColors: true,
    showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: '0.00"%"',
    dataLabelFontSize: 13, dataLabelFontBold: true, dataLabelColor: INK,
    showLegend: false, showTitle: false,
    catAxisLabelColor: INK, catAxisLabelFontSize: 11.5,
    valAxisLabelColor: GREY, valAxisLabelFontSize: 10,
    valAxisTitle: "MAPE (%)  — lower is better", showValAxisTitle: true,
    valAxisTitleColor: GREY, valAxisTitleFontSize: 11,
    valGridLine: { color: "E6ECF2", size: 1 },
    catGridLine: { style: "none" },
    valAxisMinVal: 0, valAxisMaxVal: 10.5,
  });

  card(s, M + 8.7, 1.5, 2.9, 2.15, "EAF6F1");
  s.addText("53%", { x: M + 8.7, y: 1.72, w: 2.9, h: 0.8, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 44, bold: true, color: GREEN });
  s.addText("lower error than\nthe naive baseline", { x: M + 8.7, y: 2.52, w: 2.9, h: 0.6, margin: 0,
    align: "center", fontFace: BFONT, fontSize: 12, color: INK });
  s.addText("9.14% → 4.26%", { x: M + 8.7, y: 3.12, w: 2.9, h: 0.36, margin: 0,
    align: "center", fontFace: BFONT, fontSize: 12.5, bold: true, color: GREY });

  card(s, M + 8.7, 3.85, 2.9, 2.2, OFFWHITE);
  s.addText("Read it as a ladder", { x: M + 8.9, y: 4.02, w: 2.5, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 13.5, bold: true, color: NAVY });
  s.addText(
    "Machine learning closes most of the gap. The Chinese New Year correction closes " +
    "a further 0.84pp — and it is worth far more than that on CNY days themselves.",
    { x: M + 8.9, y: 4.38, w: 2.5, h: 1.5, margin: 0, fontFace: BFONT, fontSize: 10.8, color: GREY });

  footer(s, "2026 year to date: 1 Jan – 27 May, 147 days, re-forecast weekly. Every forecast trained only on data available at that point in time.");
  s.addNotes("This ladder is measured across the full year to date including Chinese New Year, not a single favourable month. Each origin retrains from scratch on data available at that date only.");
}

// ── 19b — Reconciling the two headline accuracy numbers
{
  const s = lightSlide();
  title(s, "Why you will see two accuracy numbers in this deck", "Reading the numbers");
  s.addText(
    "Both are correct. They measure different periods, with different models, over different forecast distances.",
    { x: M, y: 1.3, w: 11.6, h: 0.3, margin: 0, fontFace: BFONT, fontSize: 13.5, color: GREY });

  const colX = [M + 3.5, M + 7.75];
  // headers
  card(s, colX[0], 1.72, 4.0, 0.62, "EAF0F6");
  s.addText("2.52%", { x: colX[0], y: 1.8, w: 4.0, h: 0.46, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 22, bold: true, color: TEAL });
  card(s, colX[1], 1.72, 4.0, 0.62, "EAF6F1");
  s.addText("4.26%", { x: colX[1], y: 1.8, w: 4.0, h: 0.46, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 22, bold: true, color: GREEN });

  s.addText("appears on the model-selection slide", {
    x: colX[0], y: 2.36, w: 4.0, h: 0.26, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 10, italic: true, color: GREY });
  s.addText("appears on the results slide — the one to quote", {
    x: colX[1], y: 2.36, w: 4.0, h: 0.26, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 10, italic: true, color: GREY });

  const rows = [
    ["Period measured", "1–28 May 2026\n(28 days)", "1 Jan – 27 May 2026\n(147 days)"],
    ["Chinese New Year included?", "No — that month has none", "Yes — 18 of the 147 days"],
    ["Model configuration", "6-model ensemble\n+ selection + holiday anchor", "Single fast model\n+ CNY correction"],
    ["Forecast distance", "One 28-day forecast", "Re-forecast weekly,\ndays 1–7 used"],
    ["What it tells you", "Best case on a clean month", "What to expect operationally"],
  ];
  let y = 2.78;
  rows.forEach((r, i) => {
    const h = 0.66;
    if (i % 2 === 0) {
      s.addShape(pres.ShapeType.rect, { x: M, y, w: 11.6, h,
        fill: { color: "F7FAFC" }, line: { color: "F7FAFC" } });
    }
    s.addText(r[0], { x: M + 0.18, y: y + 0.06, w: 3.2, h: h - 0.12, margin: 0, valign: "middle",
      fontFace: BFONT, fontSize: 11.5, bold: true, color: NAVY });
    s.addText(r[1], { x: colX[0], y: y + 0.06, w: 4.0, h: h - 0.12, margin: 0,
      align: "center", valign: "middle", fontFace: BFONT, fontSize: 11, color: INK });
    s.addText(r[2], { x: colX[1], y: y + 0.06, w: 4.0, h: h - 0.12, margin: 0,
      align: "center", valign: "middle", fontFace: BFONT, fontSize: 11, color: INK });
    y += h;
  });

  card(s, M, 6.14, 11.6, 0.72, NAVY);
  s.addText(
    "On the SAME May window, today's production model scores 3.41% — the gap to 4.26% is Chinese New Year, not a regression.",
    { x: M + 0.35, y: 6.31, w: 11, h: 0.4, margin: 0,
      fontFace: BFONT, fontSize: 13, bold: true, color: WHITE });

  s.addNotes("Use this slide if anyone challenges the numbers. The short answer: 2.52% is a best-case benchmark on a month with no Chinese New Year; 4.26% is the realistic full-year figure and is the one to plan against.");
}

// ── 20 — Accuracy by lead time (chart injected if data available)
{
  const s = lightSlide();
  title(s, "Long-range error is concentrated in one week of the year", "Forecast horizon");
  s.addText("Accuracy decays with lead time — but almost all of that decay comes from Chinese New Year.",
    { x: M, y: 1.3, w: 11.6, h: 0.3, margin: 0, fontFace: BFONT, fontSize: 13.5, color: GREY });

  const img2 = path.join(ROOT, "forecasts", "chart_accuracy_by_lead.png");
  if (fs.existsSync(img2)) {
    s.addImage({ path: img2, x: M, y: 1.62, w: 7.9, h: 3.95 });
  } else {
    card(s, M, 1.62, 7.9, 3.95, OFFWHITE);
  }

  // Split table
  card(s, M, 5.72, 7.9, 1.12, OFFWHITE);
  const tblRows = [
    ["", "1 day", "14 days", "28 days"],
    ["Chinese New Year (10 days)", "6.2%", "22.5%", "19.8%"],
    ["Rest of year (62 days)", "3.4%", "6.0%", "4.4%"],
  ];
  tblRows.forEach((row, r) => {
    const y = 5.84 + r * 0.3;
    row.forEach((cell, c2) => {
      const x = M + 0.22 + (c2 === 0 ? 0 : 3.5 + (c2 - 1) * 1.35);
      s.addText(cell, {
        x, y, w: c2 === 0 ? 3.4 : 1.3, h: 0.26, margin: 0,
        align: c2 === 0 ? "left" : "right",
        fontFace: BFONT, fontSize: 10.5, bold: r === 0,
        color: r === 0 ? GREY : (r === 1 ? RED : GREEN) });
    });
  });

  const notes = [
    { t: "CNY is the hard case", d: "At two weeks out the model misses the CNY collapse by 22%. The event's size and its shifting lunar date make it genuinely hard to anticipate.", c: RED },
    { t: "Normal weeks hold up well", d: "Outside CNY, error rises only from 3.4% to about 4.4% between one day and four weeks out — long-lead rostering is well supported.", c: GREEN },
    { t: "Plan CNY differently", d: "Do not commit CNY staffing on a 14–28 day forecast. Use short-lead updates, prior-year patterns and explicit management judgement.", c: GOLD },
  ];
  notes.forEach((n2, i) => {
    const y = 1.62 + i * 1.78;
    card(s, M + 8.3, y, 3.3, 1.62, OFFWHITE);
    circle(s, M + 8.5, y + 0.18, 0.36, n2.c, String(i + 1));
    s.addText(n2.t, { x: M + 8.94, y: y + 0.16, w: 2.5, h: 0.4, margin: 0,
      fontFace: HFONT, fontSize: 12.5, bold: true, color: NAVY });
    s.addText(n2.d, { x: M + 8.5, y: y + 0.64, w: 2.9, h: 0.9, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
  });

  footer(s, "Rolling-origin backtest, Jan–May 2026, 72 target dates per lead. Single LightGBM model; each origin retrained on data available only at that lead time.");
  s.addNotes("Important nuance: an earlier run on a March-May window showed a flat curve, because that window excluded CNY. Including CNY reveals the real horizon effect. Both results are consistent once you split the periods - this is the honest presentation.");
}

// ── 21 — Actual vs forecast
{
  const s = lightSlide();
  title(s, "Five months, tracked day by day", "2026 year to date");

  const img = path.join(ROOT, "forecasts", "chart_actual_vs_forecast.png");
  if (fs.existsSync(img)) {
    s.addImage({ path: img, x: M, y: 1.28, w: 12.1, h: 4.62 });
  } else {
    card(s, M, 1.28, 12.1, 4.62, OFFWHITE);
  }

  const obs = [
    { n: "3.98%", t: "MAPE over 148 days", c: NAVY },
    { n: "−12", t: "average bias (patron hours)", c: GREEN },
    { n: "50%", t: "demand drop at CNY, tracked", c: GOLD },
  ];
  obs.forEach((o, i) => {
    const x = M + i * 4.0;
    card(s, x, 6.0, 3.72, 0.82, OFFWHITE);
    s.addText(o.n, { x: x + 0.2, y: 6.08, w: 1.4, h: 0.62, margin: 0,
      fontFace: HFONT, fontSize: 24, bold: true, color: o.c });
    s.addText(o.t, { x: x + 1.7, y: 6.22, w: 1.85, h: 0.4, margin: 0,
      fontFace: BFONT, fontSize: 10.5, color: GREY });
  });

  footer(s, "Single LightGBM at 1-day lead. The 2.52% quoted earlier is the full ensemble on the 28-day forecast — a different configuration and window.");
  s.addNotes("Be explicit if asked: 3.98% here vs 2.52% earlier is not a contradiction — this is a single model over a harder window that includes CNY, evaluated at 1-day lead. Both numbers are honest, they measure different things.");
}

// ── 22 — What did not work
{
  const s = lightSlide();
  title(s, "What we tried that did not work", "Honest negative results");

  // Scoreboard
  card(s, M, 1.42, 3.6, 1.05, "FBEDED");
  s.addText("11", { x: M + 0.3, y: 1.54, w: 1.0, h: 0.7, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 34, bold: true, color: NAVY });
  s.addText("approaches tested", { x: M + 1.35, y: 1.76, w: 2.1, h: 0.34, margin: 0,
    fontFace: BFONT, fontSize: 12, color: INK });
  card(s, M + 3.95, 1.42, 3.6, 1.05, "FBEDED");
  s.addText("9", { x: M + 4.25, y: 1.54, w: 1.0, h: 0.7, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 34, bold: true, color: RED });
  s.addText("rejected", { x: M + 5.3, y: 1.76, w: 2.1, h: 0.34, margin: 0,
    fontFace: BFONT, fontSize: 12, color: INK });
  card(s, M + 7.9, 1.42, 3.7, 1.05, "EAF6F1");
  s.addText("2", { x: M + 8.2, y: 1.54, w: 1.0, h: 0.7, margin: 0, align: "center",
    fontFace: HFONT, fontSize: 34, bold: true, color: GREEN });
  s.addText("in production", { x: M + 9.25, y: 1.76, w: 2.1, h: 0.34, margin: 0,
    fontFace: BFONT, fontSize: 12, color: INK });

  const fails = [
    { t: "Log-transformed target", d: "Compressed the holiday spikes" },
    { t: "Ratio-to-baseline target", d: "Amplified noise — 14.6% on one window" },
    { t: "Learned blend weights", d: "Overfitted the validation period" },
    { t: "Per-weekday bias scaling", d: "Bias inverted on the next window" },
    { t: "Ridge / ElasticNet", d: "Unstable on thin history — 10.7%" },
    { t: "ExtraTrees in the pool", d: "Won one window, lost another" },
  ];
  fails.forEach((f, i) => {
    const col = i % 3, row = Math.floor(i / 3);
    const x = M + col * 3.95, y = 2.7 + row * 1.2;
    card(s, x, y, 3.68, 1.06);
    circle(s, x + 0.24, y + 0.22, 0.38, RED, "✕");
    s.addText(f.t, { x: x + 0.7, y: y + 0.16, w: 2.85, h: 0.32, margin: 0,
      fontFace: HFONT, fontSize: 12.5, bold: true, color: NAVY });
    s.addText(f.d, { x: x + 0.7, y: y + 0.5, w: 2.85, h: 0.4, margin: 0,
      fontFace: BFONT, fontSize: 10, color: GREY });
  });

  const wins = [
    { t: "Weekday × horizon selection", d: "3.30% → 3.02%" },
    { t: "Fixed-date holiday anchor", d: "3.02% → 2.52%" },
  ];
  wins.forEach((w2, i) => {
    const x = M + i * 5.95;
    card(s, x, 5.2, 5.65, 1.0, "EAF6F1");
    circle(s, x + 0.26, 5.42, 0.4, GREEN, "✓");
    s.addText(w2.t, { x: x + 0.72, y: 5.32, w: 2.8, h: 0.32, margin: 0,
      fontFace: HFONT, fontSize: 12.5, bold: true, color: NAVY });
    s.addText(w2.d, { x: x + 0.72, y: 5.66, w: 2.8, h: 0.3, margin: 0,
      fontFace: HFONT, fontSize: 13, bold: true, color: GREEN });
  });

  footer(s, "Documenting the failures prevents the next analyst re-treading the same ground.");
  s.addNotes("This slide buys credibility with a technical audience faster than any positive result.");
}

// ── 23 — Limitations
{
  const s = lightSlide();
  title(s, "Known limitations", "Where the model is weak");

  const lims = [
    { t: "Accuracy varies by period", d: "2.5–3% on a typical month; 5%+ on months with thin history or overlapping lunar holidays. A single headline number would be misleading.", c: GOLD },
    { t: "Data volume is the binding constraint", d: "Measured: 15 months of history → 8.4% error; 22 months → 4.7%; 28 months → 3.0%. Accuracy improves as history accumulates, at no development cost.", c: TEAL },
    { t: "Prediction intervals are optimistic", d: "The P10–P90 band currently captures roughly 55–79% of outcomes against an 80% design target. Treat the band as a relative confidence signal, not a guarantee.", c: RED },
    { t: "Limited headroom from blending alone", d: "On one window, even a hypothetical perfect model-picker reached only 2.20%. Further gains must come from better data, not cleverer combination.", c: GREY },
  ];
  lims.forEach((l, i) => {
    const y = 1.5 + i * 1.32;
    card(s, M, y, 11.6, 1.2);
    circle(s, M + 0.3, y + 0.38, 0.44, l.c, String(i + 1));
    s.addText(l.t, { x: M + 0.92, y: y + 0.18, w: 4.3, h: 0.36, margin: 0,
      fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY });
    s.addText(l.d, { x: M + 0.92, y: y + 0.54, w: 10.3, h: 0.58, margin: 0,
      fontFace: BFONT, fontSize: 11.8, color: GREY });
  });

  footer(s, "Stating limitations explicitly is deliberate: a forecast presented without them invites misuse.");
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 5 — DOWNSTREAM
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("05", "From forecast to roster", "Turning patron hours\ninto people",
  "Four downstream stages: hourly shape, table count, headcount, and shift assignment.");

// ── 25 — Hourly split
{
  const s = lightSlide();
  title(s, "Splitting the day into 24 hours", "Stage 2 — hourly shape");

  s.addText("The daily total is multiplied by a 24-hour shape that sums to one. The question is which shape to use for each day.",
    { x: M, y: 1.5, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  card(s, M, 2.05, 11.6, 0.86, NAVY);
  s.addText("hourly demand  =  daily forecast  ×  hourly share", {
    x: M, y: 2.05, w: 11.6, h: 0.86, margin: 0, align: "center", valign: "middle",
    fontFace: "Courier New", fontSize: 18, bold: true, color: WHITE });

  const steps = [
    { t: "Local weekday shapes", d: "Built from the previous 90 non-holiday days, recomputed every run so the shape tracks current behaviour rather than a stale annual average.", c: TEAL },
    { t: "Data decides for holidays", d: "For each holiday-day we measure whether its hourly shape genuinely differs from a normal weekday, using Total Variation Distance against contemporary weekday baselines.", c: GOLD },
    { t: "Inherit or specialise", d: "If the difference is small, the day inherits the weekday shape — more samples, more stable. If it is large, a recency-weighted holiday-specific shape is used instead.", c: GOLD },
  ];
  steps.forEach((st, i) => {
    const x = M + i * 3.95;
    card(s, x, 3.2, 3.68, 2.55, OFFWHITE);
    circle(s, x + 0.25, 3.42, 0.44, st.c, String(i + 1));
    s.addText(st.t, { x: x + 0.25, y: 3.98, w: 3.2, h: 0.4, margin: 0,
      fontFace: HFONT, fontSize: 14, bold: true, color: NAVY });
    s.addText(st.d, { x: x + 0.25, y: 4.42, w: 3.2, h: 1.2, margin: 0,
      fontFace: BFONT, fontSize: 11, color: GREY });
  });

  card(s, M, 6.05, 11.6, 0.78, "EAF3F7");
  s.addText("Why it matters: on Chinese New Year day 1 the hourly rhythm is driven by the holiday, not by whether it falls on a Monday. The method detects this from data rather than assuming it.", {
    x: M + 0.35, y: 6.2, w: 11, h: 0.5, margin: 0,
    fontFace: BFONT, fontSize: 12.5, color: INK });
}

// ── 26 — Table count
{
  const s = lightSlide();
  title(s, "How many tables should be open?", "Stage 3 — capacity sizing");

  s.addText("Hourly patron hours convert directly to concurrent players. From there, two established methods size the floor.",
    { x: M, y: 1.5, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  card(s, M, 2.05, 5.7, 2.5, OFFWHITE);
  circle(s, M + 0.3, 2.3, 0.46, TEAL, "A");
  s.addText("Queueing (Erlang-B)", { x: M + 0.92, y: 2.34, w: 4.4, h: 0.38, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  s.addText(
    "Tables behave as a loss system — a patron who finds no seat leaves rather than queues. " +
    "Erlang-B gives the minimum seat count that keeps the probability of turning someone away below a chosen target.",
    { x: M + 0.3, y: 2.86, w: 5.1, h: 1.5, margin: 0, fontFace: BFONT, fontSize: 11.8, color: GREY });

  card(s, M + 5.95, 2.05, 5.65, 2.5, OFFWHITE);
  circle(s, M + 6.25, 2.3, 0.46, GOLD, "B");
  s.addText("Service level (newsvendor)", { x: M + 6.87, y: 2.34, w: 4.4, h: 0.38, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  s.addText(
    "Whether to staff to P50 or P90 is an economic choice, not a statistical one. " +
    "The critical ratio — cost of a lost patron against the cost of an idle dealer — sets the quantile to plan to.",
    { x: M + 6.25, y: 2.86, w: 5.1, h: 1.5, margin: 0, fontFace: BFONT, fontSize: 11.8, color: GREY });

  card(s, M, 4.75, 11.6, 1.15, NAVY);
  s.addText("open table c  if   marginal expected revenue of table c   ≥   cost per table-hour", {
    x: M, y: 4.75, w: 11.6, h: 1.15, margin: 0, align: "center", valign: "middle",
    fontFace: "Courier New", fontSize: 15, bold: true, color: WHITE });

  s.addText("The profit-maximising stopping rule: keep opening tables while the next one still pays for itself. Requires revenue-per-patron-hour and cost-per-table-hour by tier.", {
    x: M, y: 6.08, w: 11.6, h: 0.6, margin: 0, align: "center",
    fontFace: BFONT, fontSize: 12.5, italic: true, color: GREY });
}

// ── 27 — Dealer headcount
{
  const s = lightSlide();
  title(s, "From tables to dealers", "Stage 4 — headcount");

  s.addText("Continuous table coverage needs more dealers than tables — because dealers rotate off for breaks, and some are absent.",
    { x: M, y: 1.5, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  card(s, M, 2.1, 11.6, 0.95, NAVY);
  s.addText("dealers per table  =  relief factor  ×  absence buffer", {
    x: M, y: 2.1, w: 11.6, h: 0.95, margin: 0, align: "center", valign: "middle",
    fontFace: "Courier New", fontSize: 18, bold: true, color: WHITE });

  card(s, M, 3.3, 5.6, 1.9, OFFWHITE);
  s.addText("Relief factor", { x: M + 0.3, y: 3.5, w: 5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  s.addText("(deal minutes + break minutes) ÷ deal minutes", {
    x: M + 0.3, y: 3.88, w: 5.1, h: 0.34, margin: 0,
    fontFace: "Courier New", fontSize: 11, color: TEAL });
  s.addText("A 60-on / 20-off rotation gives 1.33. Shorter deal periods raise it.", {
    x: M + 0.3, y: 4.28, w: 5.1, h: 0.7, margin: 0, fontFace: BFONT, fontSize: 11.8, color: GREY });

  card(s, M + 6.0, 3.3, 5.6, 1.9, OFFWHITE);
  s.addText("Absence buffer", { x: M + 6.3, y: 3.5, w: 5, h: 0.36, margin: 0,
    fontFace: HFONT, fontSize: 16, bold: true, color: NAVY });
  s.addText("1 ÷ (1 − expected absence rate)", {
    x: M + 6.3, y: 3.88, w: 5.1, h: 0.34, margin: 0,
    fontFace: "Courier New", fontSize: 11, color: TEAL });
  s.addText("A 5% absence rate gives 1.05. A sick-leave forecast can replace the flat assumption.", {
    x: M + 6.3, y: 4.28, w: 5.1, h: 0.7, margin: 0, fontFace: BFONT, fontSize: 11.8, color: GREY });

  // worked example
  card(s, M, 5.45, 11.6, 1.5, "EAF6F1");
  s.addText("Worked example", { x: M + 0.35, y: 5.6, w: 3, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 14.5, bold: true, color: NAVY });
  s.addText("1.33  ×  1.05  =  1.40 dealers per open table   →   10 tables require 14 dealers, plus ~3 supervisors at a span of five.", {
    x: M + 0.35, y: 5.98, w: 11, h: 0.7, margin: 0,
    fontFace: BFONT, fontSize: 14, color: INK });
}

// ── 28 — Shift assignment
{
  const s = lightSlide();
  title(s, "Which table opens, and when", "Stage 5 — shift assignment");

  s.addText("A constraint-programming optimiser assigns each table a shift pattern so that hourly coverage matches the forecast.",
    { x: M, y: 1.5, w: 11.6, h: 0.42, margin: 0, fontFace: BFONT, fontSize: 14.5, color: GREY });

  card(s, M, 2.05, 3.68, 3.5, OFFWHITE);
  s.addText("Decision", { x: M + 0.25, y: 2.24, w: 3.2, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  bullets(s, [
    "One shift pattern per table",
    "24-hour, 16-hour or 8-hour",
    "Start times chosen by the optimiser from a candidate list",
  ], M + 0.25, 2.66, 3.2, 2.6, 11.8);

  card(s, M + 3.95, 2.05, 3.68, 3.5, OFFWHITE);
  s.addText("Objective", { x: M + 4.2, y: 2.24, w: 3.2, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  bullets(s, [
    "Penalise under-coverage heavily",
    "Penalise surplus lightly",
    "Respect each table's preferred opening length",
  ], M + 4.2, 2.66, 3.2, 2.6, 11.8);

  card(s, M + 7.9, 2.05, 3.7, 3.5, "FBF4E4");
  s.addText("Table preference", { x: M + 8.15, y: 2.24, w: 3.2, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  bullets(s, [
    "Derived, not hand-typed",
    "Flat demand → 24-hour; spiky demand → 8-hour, aligned to its own peak",
    "Location score from distance to entrance, walkways and cage",
  ], M + 8.15, 2.66, 3.2, 2.6, 11.8);

  card(s, M, 5.75, 11.6, 1.1, NAVY);
  s.addText("Preferences are soft: if overall demand requires it, the optimiser will override a table's preferred pattern rather than leave an hour uncovered.", {
    x: M + 0.38, y: 5.95, w: 10.9, h: 0.7, margin: 0,
    fontFace: BFONT, fontSize: 13, color: TEAL_LT });
}

// ══════════════════════════════════════════════════════════════════════════
// SECTION 6 — OPERATIONS
// ══════════════════════════════════════════════════════════════════════════
sectionDivider("06", "Running it", "Operations, maintenance\nand what comes next",
  "The model is designed to be run by an operator, not a data scientist.");

// ── 30 — How to run
{
  const s = lightSlide();
  title(s, "Three commands cover normal operation", "Daily use");

  const cmds = [
    { m: "Fast daily forecast", t: "~3 min", c: "uv run python forecast.py", d: "Single LightGBM model. The routine daily driver." },
    { m: "Highest accuracy", t: "~35 min", c: "uv run python forecast.py --full --blend selection", d: "Six-model ensemble with selection and holiday anchor. For committed plans." },
    { m: "Scheduling cycle", t: "varies", c: "uv run python run_schedule.py --initial 2026-06-08 2026-07-05", d: "Freezes an initial plan, then re-forecasts on set dates and tracks drift." },
  ];
  cmds.forEach((c, i) => {
    const y = 1.5 + i * 1.44;
    card(s, M, y, 11.6, 1.3);
    s.addText(c.m, { x: M + 0.3, y: y + 0.16, w: 3.4, h: 0.34, margin: 0,
      fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
    s.addText(c.t, { x: M + 3.7, y: y + 0.2, w: 1.1, h: 0.28, margin: 0, align: "center",
      fontFace: BFONT, fontSize: 10, bold: true, color: TEAL });
    s.addShape(pres.ShapeType.roundRect, {
      x: M + 0.3, y: y + 0.55, w: 8.2, h: 0.42, rectRadius: 0.04,
      fill: { color: "1B3346" }, line: { color: "1B3346" } });
    s.addText(c.c, { x: M + 0.42, y: y + 0.55, w: 8.0, h: 0.42, margin: 0, valign: "middle",
      fontFace: "Courier New", fontSize: 10.5, color: "9FE3C4" });
    s.addText(c.d, { x: M + 8.7, y: y + 0.4, w: 2.6, h: 0.7, margin: 0,
      fontFace: BFONT, fontSize: 10.8, color: GREY });
  });

  card(s, M, 5.9, 11.6, 1.05, OFFWHITE);
  s.addText("Every run is archived under forecasts/run_<date>/ with its predictions, chart and metadata — so any past forecast can be audited against what actually happened.", {
    x: M + 0.35, y: 6.12, w: 11, h: 0.6, margin: 0,
    fontFace: BFONT, fontSize: 13, color: INK });
}

// ── 31 — Maintenance
{
  const s = lightSlide();
  title(s, "What needs a human, and how often", "Maintenance");

  const tasks = [
    { w: "Daily", t: "Append yesterday's actual demand", d: "One row in rawdata.csv. The only mandatory recurring task.", c: TEAL },
    { w: "Weekly", t: "Run the high-accuracy forecast", d: "Refresh the committed plan with the full ensemble.", c: TEAL },
    { w: "Quarterly", t: "Refresh typhoon records", d: "One command; picks up new Signal 8+ events from HKO.", c: GOLD },
    { w: "Quarterly", t: "Re-validate accuracy", d: "Backtest on a recent window to confirm no drift.", c: GOLD },
    { w: "Annually", t: "Extend the holiday calendar", d: "Add next year's holiday anchor dates before forecasting into it.", c: RED },
  ];
  tasks.forEach((t, i) => {
    const y = 1.5 + i * 1.06;
    card(s, M, y, 11.6, 0.94);
    s.addShape(pres.ShapeType.roundRect, {
      x: M + 0.25, y: y + 0.24, w: 1.15, h: 0.46, rectRadius: 0.05,
      fill: { color: t.c }, line: { color: t.c } });
    s.addText(t.w, { x: M + 0.25, y: y + 0.24, w: 1.15, h: 0.46, margin: 0,
      align: "center", valign: "middle", fontFace: BFONT, fontSize: 10.5, bold: true, color: WHITE });
    s.addText(t.t, { x: M + 1.6, y: y + 0.14, w: 4.6, h: 0.36, margin: 0,
      fontFace: HFONT, fontSize: 14, bold: true, color: NAVY });
    s.addText(t.d, { x: M + 1.6, y: y + 0.5, w: 9.6, h: 0.34, margin: 0,
      fontFace: BFONT, fontSize: 11.5, color: GREY });
  });

  footer(s, "The forecast itself needs no retraining schedule — every run trains from scratch on all data available at that moment.");
}

// ── 32 — Roadmap
{
  const s = lightSlide();
  title(s, "Where the next gains come from", "Roadmap");

  const road = [
    { p: "Now", t: "Let history accumulate", d: "The largest single lever, and it is free. Measured trajectory suggests typical-window error falls toward 2.5% as the dataset passes three years.", c: GREEN },
    { p: "Next", t: "Calibrate prediction intervals", d: "Replace in-sample residuals with out-of-sample ones so the P10–P90 band delivers its stated 80% coverage.", c: TEAL },
    { p: "Next", t: "Activate hotel pacing", d: "Infrastructure is built and leakage-guarded. Needs 18–24 months of daily snapshots to train the relationship.", c: TEAL },
    { p: "Then", t: "Complete the downstream chain", d: "Table sizing, dealer headcount and shift assignment need per-table operational data — revenue per hour, utilisation, floor coordinates.", c: GOLD },
    { p: "Then", t: "Forecast dealer absence", d: "A companion model for sick leave, replacing the flat absence assumption with a predicted rate.", c: GOLD },
  ];
  road.forEach((r, i) => {
    const y = 1.5 + i * 1.08;
    card(s, M, y, 11.6, 0.96);
    s.addShape(pres.ShapeType.roundRect, {
      x: M + 0.25, y: y + 0.25, w: 0.92, h: 0.46, rectRadius: 0.05,
      fill: { color: r.c }, line: { color: r.c } });
    s.addText(r.p, { x: M + 0.25, y: y + 0.25, w: 0.92, h: 0.46, margin: 0,
      align: "center", valign: "middle", fontFace: BFONT, fontSize: 10, bold: true, color: WHITE });
    s.addText(r.t, { x: M + 1.35, y: y + 0.14, w: 4.0, h: 0.36, margin: 0,
      fontFace: HFONT, fontSize: 14, bold: true, color: NAVY });
    s.addText(r.d, { x: M + 1.35, y: y + 0.5, w: 9.9, h: 0.36, margin: 0,
      fontFace: BFONT, fontSize: 11.2, color: GREY });
  });
}

// ── 33 — Closing
{
  const s = darkSlide();
  s.addShape(pres.ShapeType.ellipse, {
    x: -1.9, y: 4.0, w: 5.4, h: 5.4, fill: { color: NAVY_MID }, line: { color: NAVY_MID } });
  s.addShape(pres.ShapeType.ellipse, {
    x: 11.2, y: -1.4, w: 3.6, h: 3.6, fill: { color: TEAL }, line: { color: TEAL } });

  s.addText("In summary", {
    x: M + 0.6, y: 1.5, w: 8, h: 0.4, margin: 0,
    fontFace: BFONT, fontSize: 12.5, bold: true, color: GOLD, charSpacing: 2.5 });
  s.addText("A measured forecast,\nhonestly validated", {
    x: M + 0.6, y: 1.95, w: 9, h: 1.5, margin: 0,
    fontFace: HFONT, fontSize: 38, bold: true, color: WHITE });

  const pts = [
    "Two-thirds less error than the planning method it replaces",
    "Every accuracy figure measured out-of-sample, with leakage guards on all inputs",
    "Running in production today; the downstream planning chain is specified and ready to build",
    "The clearest path to further gains is more history — which accrues automatically",
  ];
  s.addText(
    pts.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i !== pts.length - 1 } })),
    { x: M + 0.6, y: 3.7, w: 8.6, h: 2.2, margin: 0,
      fontFace: BFONT, fontSize: 14.5, color: TEAL_LT, lineSpacing: 22, paraSpaceAfter: 9 }
  );

  s.addText("Questions welcome — technical detail available in the appendix and project documentation.", {
    x: M + 0.6, y: 6.35, w: 9, h: 0.4, margin: 0,
    fontFace: BFONT, fontSize: 12, italic: true, color: "8FA6BA" });
}

// ── 34 — Appendix
{
  const s = lightSlide();
  title(s, "Appendix — technical reference", "For follow-up");

  card(s, M, 1.5, 5.7, 2.5, OFFWHITE);
  s.addText("Repository layout", { x: M + 0.3, y: 1.68, w: 5, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  s.addText(
    "forecast.py — production entry point\n" +
    "blend.py — selection + holiday anchor\n" +
    "v2/_shared.py — feature engineering\n" +
    "evaluate.py — accuracy analysis\n" +
    "run_schedule.py — scheduling cycle\n" +
    "hourly/ — hourly split module\n" +
    "research/ — validation experiments",
    { x: M + 0.3, y: 2.08, w: 5.1, h: 1.8, margin: 0,
      fontFace: "Courier New", fontSize: 10, color: INK, lineSpacing: 15 });

  card(s, M + 5.95, 1.5, 5.65, 2.5, OFFWHITE);
  s.addText("Key configuration", { x: M + 6.25, y: 1.68, w: 5, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  s.addText(
    "HOLDOUT_DAYS = 28 — forecast horizon\n" +
    "HOLIDAY_ANCHORS — holiday dates by year\n" +
    "HOLIDAY_WINDOWS — demand window per holiday\n" +
    "HOLIDAY_UPWEIGHT = 3.0 — holiday sample weight\n" +
    "USE_RESERVATIONS — hotel pacing toggle\n" +
    "USE_WEATHER — typhoon feature toggle",
    { x: M + 6.25, y: 2.08, w: 5.1, h: 1.8, margin: 0,
      fontFace: "Courier New", fontSize: 10, color: INK, lineSpacing: 15 });

  card(s, M, 4.12, 11.6, 2.68, OFFWHITE);
  s.addText("Accuracy reference — 1 to 28 May 2026 holdout", { x: M + 0.3, y: 4.26, w: 6, h: 0.34, margin: 0,
    fontFace: HFONT, fontSize: 15, bold: true, color: NAVY });
  const tbl = [
    ["Method", "MAPE", "WAPE", "RMSE"],
    ["Same day last week (peeks — see note)", "7.36%", "7.44%", "705"],
    ["Same week before cutoff (like-for-like)", "6.03%", "—", "—"],
    ["Flat 28-day average", "5.12%", "5.30%", "551"],
    ["Day-of-week average", "4.70%", "4.76%", "454"],
    ["LightGBM L2 (single model)", "3.47%", "3.50%", "297"],
    ["Six-model ensemble, equal weights", "3.30%", "3.36%", "311"],
    ["Selection + fixed-date holiday anchor", "2.52%", "2.55%", "217"],
    ["Current production (this deck's config)", "3.41%", "—", "—"],
  ];
  tbl.forEach((row, r) => {
    const y = 4.74 + r * 0.216;   // 9 rows must fit the card (ends 6.70)
    row.forEach((cell, c) => {
      const x = M + 0.3 + (c === 0 ? 0 : 5.0 + (c - 1) * 1.6);
      s.addText(cell, {
        x, y, w: c === 0 ? 4.9 : 1.5, h: 0.25, margin: 0,
        align: c === 0 ? "left" : "right",
        fontFace: BFONT, fontSize: 11, bold: r === 0 || r === tbl.length - 1,
        color: r === 0 ? GREY : (r === tbl.length - 1 ? GREEN : INK) });
    });
  });

  footer(s,
    "Rows 5–7 are the original benchmark and reproduce at the June 2026 code state. " +
    "'Same day last week' reads actuals from inside the test window, so the like-for-like naive row is given above it. " +
    "Full-year performance (4.26%) is in Results — this table is a single CNY-free month.");
}

pres.writeFile({ fileName: OUT }).then(() => console.log("Wrote " + OUT));
