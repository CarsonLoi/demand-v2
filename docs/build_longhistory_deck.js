/**
 * build_longhistory_deck.js — the LONG-HISTORY model, explained from scratch.
 *
 * Separate from build_deck.js (executive) and build_walkthrough_deck.js
 * (general walkthrough). Writes its own file; touches neither of the others.
 *
 *   uv run python long_history/viz/deck_data.py       # produces deck_data_long.json
 *   node docs/build_longhistory_deck.js
 *   -> docs/Demand_Forecast_LongHistory.pptx
 *
 * The hourly-splitting slides reserve blank chart slots rather than embed
 * numbers -- long_history/data/raw/hourly_demand.csv does not exist yet, so
 * nothing about hourly accuracy is real evidence. Once it does:
 *   uv run python long_history/hourly/validate.py --holdout-days 90
 *   uv run python long_history/hourly/split.py --run-date <a holdout date>
 *   uv run python long_history/hourly/deck_charts.py --hourly-pred <that run's predictions_hourly.csv>
 * -- then paste the two resulting PNGs into the reserved slots this script
 * lays out (search "RESERVED CHART SLOT" below).
 */
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const L = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "long_history", "docs", "deck_data_long.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "long_history", "docs", "deck_data.json"), "utf8"));
const H = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "long_history", "docs", "deck_data_hourly.json"), "utf8"));

const NAVY = "0F2942", TEAL = "2E8BA8", GOLD = "E8B547";
const RED = "C74B4B", GREEN = "2FA87C", GREY = "5F7183", LIGHT = "F4F6F8", WHITE = "FFFFFF";
const HEAD = "Cambria", BODY = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Demand Forecast";
pres.title = "The Long-History Demand Forecast Model";

let sectionNo = 0;

/* ── helpers ── */
function section(title, subtitle, notes) {
  sectionNo += 1;
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addShape(pres.ShapeType.ellipse, { x: 0.9, y: 2.5, w: 1.15, h: 1.15, fill: { color: GOLD } });
  s.addText(String(sectionNo), {
    x: 0.9, y: 2.5, w: 1.15, h: 1.15, align: "center", valign: "middle",
    fontFace: HEAD, fontSize: 40, bold: true, color: NAVY, margin: 0,
  });
  s.addText(title, {
    x: 2.45, y: 2.4, w: 9.8, h: 0.95, fontFace: HEAD, fontSize: 38,
    bold: true, color: WHITE, valign: "middle", margin: 0,
  });
  s.addText(subtitle, {
    x: 2.5, y: 3.42, w: 9.6, h: 0.85, fontFace: BODY, fontSize: 16.5,
    color: "AEC3D6", valign: "top", margin: 0,
  });
  if (notes) s.addNotes(notes);
  return s;
}

function slide(title, notes) {
  const s = pres.addSlide();
  s.background = { color: WHITE };
  s.addText(title, {
    x: 0.6, y: 0.4, w: 12.1, h: 0.75, fontFace: HEAD, fontSize: 31,
    bold: true, color: NAVY, valign: "middle", margin: 0,
  });
  if (notes) s.addNotes(notes);
  return s;
}

function lede(s, text, y) {
  s.addText(text, {
    x: 0.6, y: y === undefined ? 1.18 : y, w: 12.1, h: 0.45,
    fontFace: BODY, fontSize: 15, color: GREY, italic: true, margin: 0,
  });
}

function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: fill || LIGHT }, line: { color: "E2E8EE", width: 1 },
  });
}

function circle(s, x, y, d, text, fill, txtColor, size) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill } });
  s.addText(text, {
    x, y, w: d, h: d, align: "center", valign: "middle", margin: 0,
    fontFace: HEAD, fontSize: size || 18, bold: true, color: txtColor || WHITE,
  });
}

function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.95, fontFace: HEAD, fontSize: 46, bold: true,
    color: color || NAVY, align: "center", valign: "middle", margin: 0,
  });
  s.addText(label, {
    x, y: y + 0.95, w, h: 0.55, fontFace: BODY, fontSize: 12.5, color: GREY,
    align: "center", valign: "top", margin: 0,
  });
}

// A dashed-border placeholder box for a chart that does not exist yet --
// used only where real data is required and none exists. Never fill this
// with fabricated numbers; paste the real PNG in once hourly/deck_charts.py
// has something real to produce (see the header comment for the command
// sequence). filename is shown so whoever is pasting knows which script
// output goes here.
function reservedChart(s, x, y, w, h, title, filename) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.08,
    fill: { color: "FAFBFC" },
    line: { color: "B7C2CC", width: 1.5, dashType: "dash" },
  });
  s.addShape(pres.ShapeType.rect, {
    x: x + w / 2 - 0.55, y: y + h * 0.28, w: 1.1, h: 0.75,
    fill: { color: "FFFFFF" }, line: { color: "B7C2CC", width: 1.5 },
  });
  s.addText("+", { x: x + w / 2 - 0.55, y: y + h * 0.28, w: 1.1, h: 0.75,
    align: "center", valign: "middle", fontFace: HEAD, fontSize: 32,
    bold: true, color: "B7C2CC", margin: 0 });
  s.addText(title, { x: x + 0.4, y: y + h * 0.56, w: w - 0.8, h: 0.4,
    fontFace: HEAD, fontSize: 13, bold: true, color: GREY, align: "center", margin: 0 });
  s.addText(`paste ${filename} here once real hourly data exists`, {
    x: x + 0.4, y: y + h * 0.56 + 0.4, w: w - 0.8, h: 0.5,
    fontFace: BODY, fontSize: 10, italic: true, color: "8A94A0",
    align: "center", margin: 0,
  });
}

const BASE = () => ({
  showLegend: true, legendPos: "b", legendFontSize: 11, legendFontFace: BODY,
  catAxisLabelColor: GREY, valAxisLabelColor: GREY,
  catAxisLabelFontSize: 9, valAxisLabelFontSize: 10,
  catAxisLabelFontFace: BODY, valAxisLabelFontFace: BODY,
  valGridLine: { color: "E8EDF2", size: 1 },
  catGridLine: { style: "none" },
});

// One full-width actual-vs-forecast slide
function avfSlide(title, ledeText, ser, notes, extra) {
  const s = slide(title, notes);
  lede(s, ledeText);
  const freq = Math.max(1, Math.round(ser.labels.length / 12));
  s.addChart(pres.ChartType.line, [
    { name: "actual", labels: ser.labels, values: ser.actual },
    { name: "forecast", labels: ser.labels, values: ser.forecast },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.7, w: 9.35, h: 5.05,
    chartColors: [NAVY, TEAL], lineSize: 2.5, lineSmooth: true,
    catAxisLabelFrequency: freq, catAxisLabelRotate: 45,
    valAxisMinVal: 0,
  }));
  stat(s, 9.95, 1.85, 2.85, ser.mape + "%", "average miss", TEAL);
  stat(s, 9.95, 3.45, 2.85, ser.within5 + "%", "of days within 5%", GREEN);
  stat(s, 9.95, 5.05, 2.85, (ser.bias > 0 ? "+" : "") + ser.bias + "%",
       ser.bias < 0 ? "bias — runs slightly low" : "bias — runs slightly high", GOLD);
  if (extra) {
    s.addText(extra, {
      x: 9.95, y: 6.5, w: 2.85, h: 0.6, fontFace: BODY, fontSize: 11,
      color: GREY, italic: true, margin: 0,
    });
  }
  return s;
}

/* ═════ TITLE ═════ */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("The Long-History Forecast Model", {
    x: 0.9, y: 2.1, w: 11.6, h: 1.0, fontFace: HEAD, fontSize: 46,
    bold: true, color: WHITE, margin: 0,
  });
  s.addText("Predicting casino demand from eleven years of history", {
    x: 0.95, y: 3.15, w: 11.5, h: 0.6, fontFace: BODY, fontSize: 20,
    color: GOLD, margin: 0,
  });
  s.addText("A complete walkthrough for someone new to this system.\nNo statistics background needed — every term is explained where it first appears.", {
    x: 0.95, y: 3.95, w: 10.5, h: 0.9, fontFace: BODY, fontSize: 14,
    color: "AEC3D6", margin: 0, lineSpacingMultiple: 1.3,
  });
  s.addNotes("This deck is about the long_history/ pipeline: a self-contained model trained on 2015-2026 data.");
}

/* ═════ WHAT IT DOES ═════ */
{
  const s = slide("What this system does");
  lede(s, "One job: guess how busy the casino floor will be, every day, for the next four weeks.");

  s.addText([
    { text: "It predicts ", options: { color: NAVY } },
    { text: "patron hours", options: { color: TEAL, bold: true } },
    { text: " — roughly, how many people are on the floor multiplied by how long they stay. That number drives staff scheduling.", options: { color: NAVY } },
  ], { x: 0.6, y: 1.85, w: 12.1, h: 0.55, fontFace: BODY, fontSize: 16, margin: 0 });

  card(s, 0.6, 2.75, 2.9, 2.2);
  card(s, 3.68, 2.75, 2.9, 2.2);
  card(s, 6.76, 2.75, 2.9, 2.2);
  card(s, 9.84, 2.75, 2.88, 2.2);
  stat(s, 0.6, 3.1, 2.9, "28", "days ahead, every run", NAVY);
  stat(s, 3.68, 3.1, 2.9, "11", "years of history", TEAL);
  stat(s, 6.76, 3.1, 2.9, "4,166", "days of data", GOLD);
  stat(s, 9.84, 3.1, 2.88, "3", "numbers per day", GREEN);

  card(s, 0.6, 5.25, 12.12, 1.6, "FBF6E9");
  s.addText("Three numbers, not one", {
    x: 0.95, y: 5.45, w: 11.4, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0,
  });
  s.addText("For each of the 28 days you get a most-likely figure (P50), a low case (P10) and a high case (P90). Plan against P50. The high/low band is a rough confidence signal only — it is known to be narrower than it claims, which is covered honestly later.", {
    x: 0.95, y: 5.85, w: 11.4, h: 0.85, fontFace: BODY, fontSize: 13.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.25,
  });
  s.addNotes("Data range 2015-01-01 to 2026-05-28 = 4,166 days.");
}

/* ═════ SECTION 1 ═════ */
section("Why eleven years, not two",
  "The production model trains on 2024 onward. This one goes back to 2015 — and that changes what it can survive.",
  "This section makes the case for the long-history approach using measured evidence.");

/* ═════ THE 27.7% STORY ═════ */
{
  const s = slide("The problem with a short memory");
  lede(s, "2 February 2026. Same day, same inputs, two models.");

  s.addChart(pres.ChartType.bar, [{
    name: "% miss",
    labels: ["Short memory\n(2 years of data)", "Long memory\n(11 years of data)"],
    values: [27.7, 3.2],
  }], Object.assign(BASE(), {
    x: 0.5, y: 1.8, w: 6.1, h: 4.8,
    barDir: "col", chartColors: [RED, GREEN], showLegend: false,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 15,
    dataLabelColor: NAVY, dataLabelFormatCode: '0.0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 32,
    valAxisTitle: "how far off it was (%)", showValAxisTitle: true,
    valAxisTitleColor: GREY, valAxisTitleFontSize: 11,
  }));

  card(s, 6.85, 1.9, 5.87, 2.4, "F7E9E9");
  s.addText("What went wrong", { x: 7.15, y: 2.08, w: 5.3, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("One of the model's inputs is \"how does today compare with the same day last year?\"\n\nFor 2 February 2026 that pointed at 27 January 2025 — two days before Chinese New Year, when trade had collapsed to half normal. The model was handed a comparison unlike anything in its experience.",
    { x: 7.15, y: 2.48, w: 5.3, h: 1.7, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 6.85, 4.45, 5.87, 2.2, "E9F4EF");
  s.addText("Why the longer memory coped", { x: 7.15, y: 4.63, w: 5.3, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Nothing was patched or special-cased. It had simply watched about ten Chinese New Years drift around the calendar, so the same input was familiar rather than shocking.\n\nThat is the core argument: not a better average, but far fewer disasters.",
    { x: 7.15, y: 5.03, w: 5.3, h: 1.5, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2 });
  s.addNotes("Chinese New Year fell on 29 Jan 2025 but 17 Feb 2026 — a 19-day drift. Lunar holidays move; a fixed 365-day lookback does not.");
}

/* ═════ THE DATA ═════ */
{
  const s = slide("Eleven years — and not all of it is usable");
  lede(s, "Average demand by year. The COVID years are real data but not a real signal.");

  const yrs = ["2015","2016","2017","2018","2019","2020","2021","2022","2023","2024","2025","2026"];
  const vals = [4463,3795,4050,4193,5879,1339,2142,889,4869,6500,6973,7013];
  s.addChart(pres.ChartType.bar, [{ name: "average demand", labels: yrs, values: vals }],
    Object.assign(BASE(), {
      x: 0.5, y: 1.75, w: 7.9, h: 4.8, barDir: "col", showLegend: false,
      chartColors: [NAVY, NAVY, NAVY, NAVY, NAVY, RED, RED, RED, GOLD, TEAL, TEAL, TEAL],
      varyColors: true, valAxisMinVal: 0,
      valAxisTitle: "patron hours (yearly average)", showValAxisTitle: true,
      valAxisTitleColor: GREY, valAxisTitleFontSize: 11,
    }));

  const key = [
    ["Normal trade — 2015 to 2019", "Usable history. Different market, but genuine demand behaviour.", NAVY],
    ["COVID — 2020 to 2022", "Borders shut, casino closed for weeks. Demand fell to a fifth. Not a signal.", RED],
    ["Recovery — 2023", "Real but still climbing all year. Kept, with care.", GOLD],
    ["Current — 2024 onward", "Today's business. What the short-memory model uses on its own.", TEAL],
  ];
  let y = 1.9;
  key.forEach(([t, b, c]) => {
    s.addShape(pres.ShapeType.ellipse, { x: 8.6, y: y + 0.06, w: 0.22, h: 0.22, fill: { color: c } });
    s.addText(t, { x: 9.0, y, w: 3.7, h: 0.35, fontFace: HEAD, fontSize: 14, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 9.0, y: y + 0.36, w: 3.7, h: 0.85, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.22;
  });
  s.addNotes("Regime boundaries confirmed against demand-per-table, not guessed: 2019-12 was 21.2, 2020-02 was 3.8, 2023-01 was 11.7, 2024-01 was 19.4.");
}

/* ═════ COVID HANDLING ═════ */
{
  const s = slide("How the COVID years are handled");
  lede(s, "Two separate decisions — and the difference between them matters.");

  card(s, 0.6, 1.8, 5.95, 2.35, "E9F4EF");
  circle(s, 0.9, 2.05, 0.52, "1", GREEN);
  s.addText("Do not learn from them", { x: 1.62, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("The answer for each COVID day is blanked out, so those days never become training examples. The model is never taught that demand of 889 is normal.",
    { x: 0.9, y: 2.7, w: 5.35, h: 1.3, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 6.75, 1.8, 5.97, 2.35, "FBF6E9");
  circle(s, 7.05, 2.05, 0.52, "2", GOLD, NAVY);
  s.addText("But keep the rows", { x: 7.77, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("This sounds wrong but is essential. Delete the rows and every \"what happened N days ago\" calculation reaching across the gap breaks. Blank the answer, keep the calendar continuous.",
    { x: 7.05, y: 2.7, w: 5.37, h: 1.3, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 0.6, 4.35, 12.12, 1.35, LIGHT);
  s.addText("A third safeguard: block the year-ago lookups too", { x: 0.95, y: 4.5, w: 11.4, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("A day in 2023 comparing itself against \"last year\" would be comparing against collapsed 2022 trade. Any year-ago input whose source lands inside COVID is blanked as well.",
    { x: 0.95, y: 4.88, w: 11.4, h: 0.65, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0 });

  card(s, 0.6, 5.85, 12.12, 1.15, "EAF2F5");
  s.addText("Closure days are found in the data, not typed in by hand", { x: 0.95, y: 5.98, w: 11.4, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Zero tables open means the floor was shut. That rule alone correctly finds typhoon Mangkhut (Sept 2018) and both COVID closures — 26 days in total. They are supply zeros, not demand.",
    { x: 0.95, y: 6.33, w: 11.4, h: 0.55, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0 });
  s.addNotes("EXCLUDE_FROM_TRAINING covers 2020-01-23 to 2022-12-31. Closures detected via floortables == 0.");
}

/* ═════ HALF-LIFE ═════ */
{
  const s = slide("The setting that makes long history actually work");
  lede(s, "Adding old data changes nothing unless you also change how much old data counts.");

  s.addChart(pres.ChartType.line, [
    { name: "240-day half-life (short-memory default)", labels: D.weights.labels, values: D.weights.hl240 },
    { name: "1095-day half-life (this model)", labels: D.weights.labels, values: D.weights.hl1095 },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.75, w: 7.4, h: 4.75,
    chartColors: [TEAL, GOLD], lineSize: 3, lineSmooth: true,
    catAxisTitle: "days in the past", showCatAxisTitle: true,
    valAxisTitle: "how much that day counts (%)", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  }));

  card(s, 8.2, 1.85, 4.52, 2.15, "FBF6E9");
  s.addText("What half-life means", { x: 8.5, y: 2.03, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("How many days until a day counts half as much. At 240 days, something from 240 days ago counts half; 480 days ago a quarter. It falls away fast.",
    { x: 8.5, y: 2.42, w: 3.9, h: 1.4, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 8.2, 4.2, 4.52, 2.5, "F7E9E9");
  s.addText("The trap", { x: 8.5, y: 4.42, w: 3.9, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("At a 240-day half-life a day from 2016 counts about 1/35,000th of today — mathematically invisible.\n\nSo loading eleven years of history and changing nothing else does literally nothing. The half-life is not a follow-up to adding history. It IS the change.",
    { x: 8.5, y: 4.82, w: 3.9, h: 1.75, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.15 });
  s.addText("Holiday days also get an extra 3× boost — they are rare but high-stakes.", {
    x: 0.6, y: 6.6, w: 7.3, h: 0.4, fontFace: BODY, fontSize: 12.5, color: GREY, italic: true, margin: 0,
  });
  s.addNotes("weight = 0.5 ^ (days_back / half_life). config.DEFAULT_HALF_LIFE = 1095.");
}

/* ═════ SECTION 2 ═════ */
section("The rule that shapes everything",
  "Every accuracy number in this deck depends on one constraint being obeyed.",
  "If the audience remembers one section, make it this one.");

/* ═════ NO PEEKING ═════ */
{
  const s = slide("No peeking");
  lede(s, "A forecast may only use information that genuinely existed at the time it was made.");
  s.addText("Suppose today is 1 March and you want to predict 15 March — that is 14 days ahead.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.4, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

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
  s.addText("Why it matters", { x: 1.55, y: 4.45, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("On 1 March, demand for 5 March has not happened yet. A model allowed to see it while learning would score brilliantly in testing and fail completely in real use.\n\nThis mistake is called leakage. It is the most common way forecasting projects fail.",
    { x: 0.9, y: 5.05, w: 5.3, h: 1.55, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 6.75, 4.15, 5.97, 2.6);
  circle(s, 7.05, 4.42, 0.5, "✓", GREEN);
  s.addText("How it is enforced", { x: 7.7, y: 4.45, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Every backward-looking input carries a rule: it may not read anything newer than (days ahead + 1).\n\nEvery chart in the next section obeys this. Nothing you will see was known to the model when it made the call.",
    { x: 7.05, y: 5.05, w: 5.37, h: 1.55, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });
  s.addNotes("Horizon gating: min_safe = horizon + 1, enforced inside core.py.");
}

/* ═════ 28 MODELS ═════ */
{
  const s = slide("Why there are 28 models, not one");
  lede(s, "A direct consequence of the no-peeking rule.");
  s.addText("A forecast for tomorrow may use yesterday's numbers. A forecast for four weeks out may not. They are genuinely different problems, so each gets its own model.", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.55, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });
  const rows = [
    ["Model 1", "predicts tomorrow", "may use data up to yesterday", "the most information"],
    ["Model 7", "predicts one week out", "nothing from the last 7 days", "less information"],
    ["Model 14", "predicts two weeks out", "nothing from the last 14 days", "less again"],
    ["Model 28", "predicts four weeks out", "nothing from the last 28 days", "the least information"],
  ];
  let y = 2.55;
  rows.forEach(([a, b, c, d], i) => {
    const tint = ["EAF2F5", "EFF3F6", "F4F3EF", "F7F3EA"][i];
    card(s, 0.6, y, 12.12, 0.98, tint);
    s.addText(a, { x: 0.95, y: y + 0.27, w: 1.7, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 2.75, y: y + 0.29, w: 3.0, h: 0.4, fontFace: BODY, fontSize: 13.5, color: NAVY, margin: 0 });
    s.addText(c, { x: 5.85, y: y + 0.29, w: 4.1, h: 0.4, fontFace: BODY, fontSize: 13, color: GREY, margin: 0 });
    s.addText(d, { x: 10.05, y: y + 0.29, w: 2.5, h: 0.4, fontFace: BODY, fontSize: 12.5, color: TEAL, italic: true, margin: 0 });
    y += 1.08;
  });
  s.addText("This is why accuracy gets worse the further out you look — there is simply less to go on. The next section shows exactly how much worse.", {
    x: 0.6, y: 6.95, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 14, color: NAVY, margin: 0,
  });
  s.addNotes("28 separate LightGBM models per run, one per forecast distance.");
}

/* ═════ SECTION 3 — THE MAIN EVENT ═════ */
section("Actual versus forecast",
  "Every chart that follows is held-out: the model had never seen these days when it predicted them.",
  "This is the evidence section. Same model, same settings, four different forecast distances.");

/* ═════ AVF CHARTS ═════ */
avfSlide("One day ahead",
  "Forecast made the day before. 2026 so far — the easiest case, and the ceiling on what is achievable.",
  L.lead01_2026,
  "Lead 1. Fixed-lead mode: every point measured at exactly the same forecast distance.");

avfSlide("One week ahead",
  "Forecast made 7 days before. This is the realistic scheduling horizon.",
  L.lead07_2026,
  "Lead 7, rolling window: re-forecasting every 7 days, tiling the period with no gaps.");

avfSlide("Two weeks ahead",
  "Forecast made 14 days before. Now the model cannot see anything from the previous fortnight.",
  L.lead14_2026,
  "Lead 14, rolling window.");

avfSlide("Four weeks ahead",
  "Forecast made 28 days before — the full planning horizon, and the hardest test.",
  L.lead28_2026,
  "Lead 28, rolling window. This is the number to quote for monthly planning.");

/* ═════ ALL LEADS TOGETHER ═════ */
{
  const s = slide("All four distances, side by side");
  lede(s, "The same period and the same model — only how far ahead it was asked changes.");

  const keys = ["lead01_2026", "lead07_2026", "lead14_2026", "lead28_2026"];
  const labels = ["1 day", "7 days", "14 days", "28 days"];
  s.addChart(pres.ChartType.bar, [
    { name: "average % miss", labels, values: keys.map(k => L[k].mape) },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.75, w: 6.3, h: 4.85, barDir: "col", showLegend: false,
    chartColors: [GREEN, TEAL, GOLD, RED], varyColors: true,
    showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13,
    dataLabelColor: NAVY, dataLabelFormatCode: '0.00"%"',
    valAxisMinVal: 0,
    catAxisTitle: "how far ahead the forecast was made", showCatAxisTitle: true,
    valAxisTitle: "average % miss", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  }));

  s.addChart(pres.ChartType.bar, [
    { name: "% of days within 5%", labels, values: keys.map(k => L[k].within5) },
  ], Object.assign(BASE(), {
    x: 7.0, y: 1.75, w: 5.72, h: 4.85, barDir: "col", showLegend: false,
    chartColors: [NAVY], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFontSize: 13, dataLabelColor: NAVY, dataLabelFormatCode: '0"%"',
    valAxisMinVal: 0, valAxisMaxVal: 100,
    catAxisTitle: "how far ahead the forecast was made", showCatAxisTitle: true,
    valAxisTitle: "share of days within 5% of actual", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  }));

  s.addText("Never quote an accuracy figure without saying how far ahead it was measured — the same model ranges across these numbers depending only on that.", {
    x: 0.6, y: 6.7, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 13.5, bold: true, color: NAVY, margin: 0,
  });
  s.addNotes("Left: average miss. Right: the share of days landing within 5% — often the more useful operational measure.");
}

/* ═════ LONG vs SHORT OVERLAY ═════ */
if (L.compare_2026) {
  const c = L.compare_2026;
  const s = slide("Long memory versus short memory, day by day");
  lede(s, "Identical protocol: same days, same 7-day lead, same test. Only the training history differs.");
  const freq = Math.max(1, Math.round(c.labels.length / 12));
  s.addChart(pres.ChartType.line, [
    { name: "actual", labels: c.labels, values: c.actual },
    { name: "long-history model (11 years)", labels: c.labels, values: c.long_history },
    { name: "short-memory model (2 years)", labels: c.labels, values: c.production },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.7, w: 9.35, h: 5.05,
    chartColors: [NAVY, TEAL, RED], lineSize: 2.4, lineSmooth: true,
    catAxisLabelFrequency: freq, catAxisLabelRotate: 45, valAxisMinVal: 0,
  }));
  card(s, 9.95, 1.85, 2.85, 1.55, LIGHT);
  s.addText("average miss", { x: 10.15, y: 1.97, w: 2.45, h: 0.3, fontFace: BODY, fontSize: 11.5, color: GREY, align: "center", margin: 0 });
  s.addText([
    { text: c.mape_long + "%", options: { color: TEAL, bold: true, fontSize: 24 } },
    { text: "   vs   ", options: { color: GREY, fontSize: 13 } },
    { text: c.mape_prod + "%", options: { color: RED, bold: true, fontSize: 24 } },
  ], { x: 10.15, y: 2.3, w: 2.45, h: 0.5, fontFace: HEAD, align: "center", margin: 0 });
  s.addText("long history   ·   short memory", { x: 10.15, y: 2.85, w: 2.45, h: 0.35, fontFace: BODY, fontSize: 9.5, color: GREY, align: "center", margin: 0 });

  card(s, 9.95, 3.55, 2.85, 1.55, "F7E9E9");
  s.addText("days missed by over 15%", { x: 10.15, y: 3.67, w: 2.45, h: 0.3, fontFace: BODY, fontSize: 11.5, color: GREY, align: "center", margin: 0 });
  s.addText([
    { text: String(c.bad15_long), options: { color: TEAL, bold: true, fontSize: 30 } },
    { text: "   vs   ", options: { color: GREY, fontSize: 13 } },
    { text: String(c.bad15_prod), options: { color: RED, bold: true, fontSize: 30 } },
  ], { x: 10.15, y: 4.0, w: 2.45, h: 0.6, fontFace: HEAD, align: "center", margin: 0 });
  s.addText("half as many bad days", { x: 10.15, y: 4.62, w: 2.45, h: 0.35, fontFace: BODY, fontSize: 10.5, color: NAVY, bold: true, align: "center", margin: 0 });

  card(s, 9.95, 5.25, 2.85, 1.5, "FBF6E9");
  s.addText("This is the real argument", { x: 10.15, y: 5.38, w: 2.45, h: 0.32, fontFace: HEAD, fontSize: 13, bold: true, color: NAVY, margin: 0 });
  s.addText("On a typical day the two are close. The gap opens at Chinese New Year: on 16 Feb the actual was 3,518 — long history said 3,550, short memory said 5,414.", {
    x: 10.15, y: 5.7, w: 2.45, h: 1.0, fontFace: BODY, fontSize: 10.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.12,
  });
  s.addNotes(`n=${c.n} overlapping days, both from a 7-day-lead rolling-window backtest. Worst single day: long history ${c.worst_long}%, short memory ${c.worst_prod}%. Days over 10% off: ${c.bad10_long} vs ${c.bad10_prod}.`);
}

/* ═════ 2025 FULL YEAR ═════ */
if (L.lead07_2025) {
  const y = L.lead07_2025;
  const s = slide("A full year, one week ahead");
  lede(s, "All of 2025 — and one event that dominates the whole year's average.");
  const freq = Math.max(1, Math.round(y.labels.length / 12));
  s.addChart(pres.ChartType.line, [
    { name: "actual", labels: y.labels, values: y.actual },
    { name: "forecast", labels: y.labels, values: y.forecast },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.7, w: 9.35, h: 5.05,
    chartColors: [NAVY, TEAL], lineSize: 2, lineSmooth: true,
    catAxisLabelFrequency: freq, catAxisLabelRotate: 45, valAxisMinVal: 0,
  }));
  s.addText("← Typhoon RAGASA", {
    x: 7.35, y: 5.32, w: 2.0, h: 0.35, fontFace: BODY, fontSize: 11,
    bold: true, color: RED, align: "left", margin: 0,
  });
  stat(s, 9.95, 1.8, 2.85, y.mape + "%", "all 365 days", RED);
  stat(s, 9.95, 3.3, 2.85, y.mape_ex_storm_recovery + "%",
       "excluding the storm week", TEAL);
  card(s, 9.95, 4.85, 2.85, 1.9, "F7E9E9");
  s.addText("Two days, one fifth of the year's error", {
    x: 10.2, y: 5.0, w: 2.4, h: 0.5, fontFace: HEAD, fontSize: 13.5, bold: true, color: NAVY, margin: 0,
  });
  s.addText("A typhoon shut the floor on 23–24 September. Those two days alone move the annual figure from 4.94% to 7.71%.", {
    x: 10.2, y: 5.55, w: 2.4, h: 1.1, fontFace: BODY, fontSize: 11.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addNotes("2025 lead 7 rolling window. 7.71% all days, 5.33% excluding the two storm days, 4.94% excluding storm plus the six-day recovery.");
}

/* ═════ TYPHOON CLOSE-UP ═════ */
if (L.typhoon_sep2025) {
  const t = L.typhoon_sep2025;
  const s = slide("What a typhoon does — and why it is not the model's fault");
  lede(s, "Late September 2025, one week ahead. The single worst forecast in the whole eleven years.");
  s.addChart(pres.ChartType.line, [
    { name: "actual", labels: t.labels, values: t.actual },
    { name: "forecast", labels: t.labels, values: t.forecast },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.7, w: 8.5, h: 5.05,
    chartColors: [NAVY, TEAL], lineSize: 3, lineSmooth: false,
    lineDataSymbol: "circle", lineDataSymbolSize: 6,
    catAxisLabelRotate: 45, valAxisMinVal: 0,
  }));

  card(s, 9.2, 1.85, 3.52, 2.15, "F7E9E9");
  s.addText("23–24 September", { x: 9.45, y: 2.02, w: 3.0, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Demand fell to 879 against a normal 7,000 — an 87% drop. The forecast said about 5,000.\n\nNothing in eleven years of demand history predicts a storm. The model had no way to know.", {
    x: 9.45, y: 2.42, w: 3.05, h: 1.45, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 9.2, 4.15, 3.52, 2.6, "FBF6E9");
  s.addText("The knock-on effect", { x: 9.45, y: 4.33, w: 3.0, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Look at the week AFTER. Trade recovered, but the forecast stayed low — because the model's \"how busy have we been lately\" inputs were themselves poisoned by the two collapsed days.\n\nOne shock damages the following week too. This is the same mechanism that makes Chinese New Year hard.", {
    x: 9.45, y: 4.72, w: 3.05, h: 1.9, fontFace: BODY, fontSize: 11.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.15,
  });
  s.addNotes("Tables stayed nominally open (floortables ~285), so the automatic closure detection did NOT catch these days. They must be handled by the manual override tool.");
}

/* ═════ SECTION 4 ═════ */
section("What the model looks at",
  "About 170 pieces of information per day — and they are not equally important.",
  "Feature importance measured by model gain, not assumed.");

/* ═════ FEATURE DONUT ═════ */
{
  const s = slide("Half the model is holidays");
  lede(s, "Share of the model's decision-making, measured.");
  s.addChart(pres.ChartType.doughnut, [{
    name: "Share", labels: ["Holidays", "Recent history", "Calendar", "Everything else"],
    values: [47, 15, 12, 26],
  }], {
    x: 0.5, y: 1.7, w: 5.6, h: 5.1, holeSize: 52,
    chartColors: [GOLD, TEAL, NAVY, "C3CDD6"],
    showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: BODY,
    showValue: true, dataLabelFontSize: 12, dataLabelColor: "FFFFFF",
    dataLabelFormatCode: '0"%"', showTitle: false,
  });
  const items = [
    ["Holidays — 47%", "Nine of them. Not single days but windows: Chinese New Year disrupts trade for about three weeks.", GOLD],
    ["Recent history — 15%", "What demand did 2, 7, 28, 365 days ago; rolling averages; the last few same-weekdays.", TEAL],
    ["Calendar — 12%", "Day of week, month, position in the year. Saturday is a different business from Tuesday.", NAVY],
    ["Everything else — 26%", "Tables open on the floor, the Mainland-China working calendar, and combinations of the above.", "8A94A0"],
  ];
  let y = 1.95;
  items.forEach(([t, b, c]) => {
    s.addShape(pres.ShapeType.ellipse, { x: 6.55, y: y + 0.06, w: 0.22, h: 0.22, fill: { color: c } });
    s.addText(t, { x: 6.95, y, w: 5.8, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 6.95, y: y + 0.36, w: 5.77, h: 0.85, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.28;
  });
  s.addNotes("Long history also adds week-of-year and a regime feature, which are noise on 2 years but learnable on 11.");
}

/* ═════ CNY ═════ */
{
  const s = slide("Chinese New Year — the hardest part");
  lede(s, "Real demand around CNY 2025. Trade does not dip; it collapses, then rebounds.");
  s.addChart(pres.ChartType.line, [
    { name: "actual demand", labels: D.cny.labels, values: D.cny.values },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.7, w: 8.3, h: 5.0, chartColors: [NAVY], lineSize: 3,
    showLegend: false, valAxisMinVal: 0, catAxisLabelRotate: 45,
  }));
  card(s, 9.05, 1.85, 3.67, 1.85, "F7E9E9");
  s.addText("The collapse", { x: 9.3, y: 2.02, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("On the eve of CNY demand falls to roughly HALF of normal — 3,331 against a typical 7,300.",
    { x: 9.3, y: 2.4, w: 3.17, h: 1.2, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2 });
  card(s, 9.05, 4.0, 3.67, 2.7, "FBF6E9");
  s.addText("The fix", { x: 9.3, y: 4.2, w: 3.15, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("The collapse is dramatic but consistent year to year, so it is looked up rather than learned.\n\nFind the matching day last year by POSITION (\"three days before CNY\"), take the ratio it ran at, apply it to this year's normal level, and blend 80% of that into the answer.\n\nCNY error roughly halved.",
    { x: 9.3, y: 4.62, w: 3.17, h: 2.0, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
  s.addNotes("Tried on Easter, Dragon Boat and Mid-Autumn: made all three worse. CNY only.");
}

/* ═════ MOVING HOLIDAYS + CHING MING ═════ */
{
  const s = slide("Two kinds of holiday — and a bug this version fixes");
  lede(s, "This distinction causes more subtle errors than anything else in the system.");

  card(s, 0.6, 1.8, 5.95, 2.3, "EAF2F5");
  circle(s, 0.9, 2.05, 0.5, "F", TEAL);
  s.addText("Fixed holidays", { x: 1.55, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Christmas, New Year, Labour Day, Golden Week, Ching Ming.\nSame date every year — count back 364 days and you land on the same day of the week.",
    { x: 0.9, y: 2.65, w: 5.35, h: 1.3, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 6.75, 1.8, 5.97, 2.3, "FBF6E9");
  circle(s, 7.05, 2.05, 0.5, "M", GOLD, NAVY);
  s.addText("Moving holidays", { x: 7.7, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
  s.addText("Chinese New Year, Mid-Autumn, Dragon Boat, Easter.\nThey follow the lunar calendar and shift by up to three weeks, so a 364-day lookback lands on the wrong part of the holiday.",
    { x: 7.05, y: 2.65, w: 5.37, h: 1.3, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });

  card(s, 0.6, 4.35, 12.12, 2.4, "E9F4EF");
  circle(s, 0.95, 4.62, 0.55, "✓", GREEN);
  s.addText("A real calendar bug, found and corrected here", { x: 1.75, y: 4.6, w: 10.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText("Ching Ming is treated as 4 April every year. It is not a fixed date — it is a solar term, and it actually falls on 5 April in 2018, 2019, 2022 and 2023.\n\nThe long-history version corrects this, and extends every holiday date back to 2016, each one checked against an independent calendar library rather than typed in by hand.",
    { x: 1.75, y: 5.05, w: 10.6, h: 1.55, fontFace: BODY, fontSize: 13, color: NAVY, margin: 0, lineSpacingMultiple: 1.25 });
  s.addNotes("Verified against the chinese_calendar library at run time; the pipeline refuses to run if any anchor fails verification.");
}

/* ═════ SECTION — HOW THE MODEL DECIDES ═════ */
section("How the model actually decides",
  "No formulas. The whole thing is a very large pile of yes/no questions.",
  "Simplified deliberately — the aim is the core idea, not the mathematics.");

/* ═════ WHAT A DECISION TREE IS ═════ */
{
  const s = slide("Step 1: a single yes/no question");
  lede(s, "The model's basic building block is called a decision tree. It is simpler than it sounds.");

  s.addText("A tree asks a question about the day it is trying to predict, follows the answer, and ends at a guess. Here is the FIRST question the real model asked — not an invented example:", {
    x: 0.6, y: 1.75, w: 12.1, h: 0.5, fontFace: BODY, fontSize: 14.5, color: NAVY, margin: 0,
  });

  // root
  card(s, 3.55, 2.4, 6.2, 0.95, "FBF6E9");
  s.addText("Was the busiest day in the last two weeks below 6,708 patron hours?", {
    x: 3.75, y: 2.4, w: 5.8, h: 0.95, fontFace: BODY, fontSize: 14, bold: true,
    color: NAVY, align: "center", valign: "middle", margin: 0,
  });

  // branch labels
  s.addText("YES", { x: 2.6, y: 3.45, w: 1.6, h: 0.35, fontFace: BODY, fontSize: 12.5, bold: true, color: GREEN, align: "center", margin: 0 });
  s.addText("NO", { x: 9.1, y: 3.45, w: 1.6, h: 0.35, fontFace: BODY, fontSize: 12.5, bold: true, color: RED, align: "center", margin: 0 });

  // children
  card(s, 1.15, 3.9, 4.6, 1.0, "E9F4EF");
  s.addText("It has been a quiet fortnight\n→ expect a quieter day", {
    x: 1.35, y: 3.9, w: 4.2, h: 1.0, fontFace: BODY, fontSize: 13, color: NAVY,
    align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.15,
  });
  card(s, 7.55, 3.9, 4.6, 1.0, "F7E9E9");
  s.addText("It has been a busy fortnight\n→ expect a busier day", {
    x: 7.75, y: 3.9, w: 4.2, h: 1.0, fontFace: BODY, fontSize: 13, color: NAVY,
    align: "center", valign: "middle", margin: 0, lineSpacingMultiple: 1.15,
  });

  // second level hint
  s.addText("…then another question, and another — day of week? holiday? how many tables are open? — up to five deep.", {
    x: 0.6, y: 5.05, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 13, color: GREY,
    italic: true, align: "center", margin: 0,
  });

  card(s, 0.6, 5.6, 12.12, 1.45, LIGHT);
  s.addText("Why this is a good fit for the problem", {
    x: 0.95, y: 5.75, w: 11.4, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0,
  });
  s.addText("Nobody wrote that question. The model found it by trying every possible question on eleven years of days and keeping whichever one separated busy days from quiet days best. It works the same way a person would reason about the floor — just far more thoroughly, and without a hunch.", {
    x: 0.95, y: 6.12, w: 11.4, h: 0.8, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("The real first split is on rolling_max_14 <= 6708. Trees are capped at depth 5 and 15 leaves to keep each one deliberately simple.");
}

/* ═════ BOOSTING ═════ */
{
  const s = slide("Step 2: one tree is weak — 500 together are strong");
  lede(s, "This is called boosting. Each new tree studies what the previous ones got wrong, and nudges.");

  const M = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "long_history", "docs", "deck_model_demo.json"), "utf8"));
  const lbl = M.stages.map(String);
  s.addChart(pres.ChartType.line, [
    { name: "the model's guess, after this many trees", labels: lbl, values: M.preds },
    { name: "what actually happened", labels: lbl, values: lbl.map(() => M.actual) },
  ], Object.assign(BASE(), {
    x: 0.5, y: 1.75, w: 7.6, h: 4.6,
    chartColors: [TEAL, NAVY], lineSize: 3, lineSmooth: true,
    lineDataSymbol: "circle", lineDataSymbolSize: 6,
    catAxisTitle: "number of trees used", showCatAxisTitle: true,
    valAxisTitle: "predicted patron hours", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
  }));

  card(s, 8.3, 1.9, 4.42, 2.55, LIGHT);
  s.addText("A real forecast, tree by tree", { x: 8.6, y: 2.06, w: 3.85, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: `Predicting ${M.target}.  The true answer was ${M.actual.toLocaleString()}.\n`, options: { breakLine: true, color: NAVY } },
    { text: `After 1 tree:      ${M.preds[0].toLocaleString()}   (${M.errs[0]}% off)\n`, options: { breakLine: true } },
    { text: `After 20 trees:  ${M.preds[6].toLocaleString()}   (${M.errs[6]}% off)\n`, options: { breakLine: true } },
    { text: `After 100 trees: ${M.preds[9].toLocaleString()}   (${M.errs[9]}% off)\n`, options: { breakLine: true } },
    { text: `All 500 trees:   ${M.final.toLocaleString()}   (${M.final_err}% off)`, options: { bold: true, color: NAVY } },
  ], { x: 8.6, y: 2.45, w: 3.85, h: 1.9, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, lineSpacingMultiple: 1.25 });

  card(s, 8.3, 4.6, 4.42, 2.15, "FBF6E9");
  s.addText("The honest version", { x: 8.6, y: 4.78, w: 3.85, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Most of the gain arrives early — by 100 trees it is nearly there. The rest make small adjustments, and on this particular day the very last ones drift back a little.\n\nNo single tree is clever. Added together, they are.", {
    x: 8.6, y: 5.16, w: 3.85, h: 1.5, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addText("Think of it as a first rough guess, then 499 small corrections — each one aimed at the mistakes still left over.", {
    x: 0.6, y: 6.5, w: 7.6, h: 0.45, fontFace: BODY, fontSize: 12.5, color: GREY, italic: true, margin: 0,
  });
  s.addNotes(`Real numbers from a lead-${M.horizon} forecast made on ${M.origin}, trained on ${M.n_train.toLocaleString()} rows. Produced by long_history/make_model_demo.py.`);
}

/* ═════ WHY THIS MODEL ═════ */
{
  const s = slide("Step 3: why this kind of model, and not another");
  lede(s, "The technique is called LightGBM. The name is scarier than the idea.");

  card(s, 0.6, 1.8, 5.95, 2.5, "E9F4EF");
  circle(s, 0.9, 2.05, 0.52, "✓", GREEN);
  s.addText("What it is good at here", { x: 1.62, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Finds combinations by itself — nobody has to say \"Saturdays before a holiday are different\"", options: { bullet: true, breakLine: true } },
    { text: "Copes with gaps, which matters here because long-range models have data deliberately withheld", options: { bullet: true, breakLine: true } },
    { text: "Trains in minutes on an ordinary laptop", options: { bullet: true } },
  ], { x: 0.95, y: 2.62, w: 5.3, h: 1.6, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, paraSpaceAfter: 5 });

  card(s, 6.75, 1.8, 5.97, 2.5, "F7E9E9");
  circle(s, 7.05, 2.05, 0.52, "✗", RED);
  s.addText("What was tried instead", { x: 7.77, y: 2.08, w: 4.6, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Classic time-series (ARIMA) — cannot take 170 different inputs at once", options: { bullet: true, breakLine: true } },
    { text: "Neural networks — need far more than a decade of daily data to be worth it", options: { bullet: true, breakLine: true } },
    { text: "GAM, a smoother statistical model — tested here, worse at all nine forecast distances", options: { bullet: true } },
  ], { x: 7.1, y: 2.62, w: 5.3, h: 1.6, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, paraSpaceAfter: 5 });

  s.addText("Putting the whole thing together", { x: 0.6, y: 4.55, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  const steps = [
    ["170 facts", "about the day being predicted", NAVY],
    ["500 trees", "each correcting the last", TEAL],
    ["× 28 models", "one per forecast distance", GOLD],
    ["+ CNY fix", "applied afterwards", GREEN],
  ];
  let x = 0.6;
  steps.forEach(([t, b, c], i) => {
    card(s, x, 5.05, 2.85, 1.35, LIGHT);
    s.addText(t, { x: x + 0.15, y: 5.22, w: 2.55, h: 0.45, fontFace: HEAD, fontSize: 16, bold: true, color: c, align: "center", margin: 0 });
    s.addText(b, { x: x + 0.15, y: 5.68, w: 2.55, h: 0.6, fontFace: BODY, fontSize: 11.5, color: GREY, align: "center", margin: 0, lineSpacingMultiple: 1.15 });
    if (i < 3) {
      s.addText("→", { x: x + 2.88, y: 5.4, w: 0.42, h: 0.6, fontFace: BODY, fontSize: 20, color: GOLD, align: "center", valign: "middle", margin: 0 });
    }
    x += 3.16;
  });
  s.addText("That is the entire system. Everything else in this deck is about feeding it honestly and checking it fairly.", {
    x: 0.6, y: 6.55, w: 12.1, h: 0.45, fontFace: BODY, fontSize: 13, color: NAVY, italic: true, margin: 0,
  });
  s.addNotes("500 trees, learning rate 0.02, max depth 5, 15 leaves. Deliberately small trees so no single one can dominate.");
}

/* ═════ SECTION — HOURLY ALLOCATION ═════ */
section("Splitting the day into hours",
  "The daily forecast, broken into 24 numbers for staffing — using the actual historical shape, not another trained model.",
  "New section. Every accuracy claim here is the design's stated hypothesis, not a measured result yet -- long_history/data/raw/hourly_demand.csv does not exist. See the reserved-chart slides' captions.");

/* ═════ HOURLY: WHAT IT ADDS ═════ */
{
  const s = slide("From one number a day to twenty-four");
  lede(s, "Staffing doesn't run on a daily total — it runs on \"how many people at 3pm versus 9pm.\"");

  s.addText("The daily forecast (everything so far in this deck) gives one P50 per day. This step takes that single number and spreads it across the 24 hours of the day, using the SHAPE of a normal — or holiday — day, measured directly from history.", {
    x: 0.6, y: 1.8, w: 12.1, h: 0.75, fontFace: BODY, fontSize: 15, color: NAVY, margin: 0,
  });

  s.addChart(pres.ChartType.line, [
    { name: "share of the day's total demand", labels: H.sample_day.hour.map(h => String(h)), values: H.sample_day.demand },
  ], Object.assign(BASE(), {
    x: 0.5, y: 2.75, w: 8.4, h: 4.0,
    chartColors: [TEAL], lineSize: 3, lineSmooth: true, showLegend: false,
    catAxisTitle: "hour of day", showCatAxisTitle: true,
    valAxisTitle: "patron hours", showValAxisTitle: true,
    catAxisTitleColor: GREY, valAxisTitleColor: GREY,
    catAxisTitleFontSize: 11, valAxisTitleFontSize: 11,
    valAxisMinVal: 0,
  }));

  card(s, 9.15, 2.85, 3.65, 3.9, "EAF2F5");
  s.addText("A real day's shape", { x: 9.45, y: 3.05, w: 3.05, h: 0.35, fontFace: HEAD, fontSize: 15, bold: true, color: NAVY, margin: 0 });
  s.addText("Quiet overnight, building through the afternoon, peaking around 9pm, tapering late. This shape — not the daily total — is what the split needs to get right.\n\nEvery day has its own shape: weekdays differ from weekends, and holidays can differ from both.", {
    x: 9.45, y: 3.45, w: 3.05, h: 3.1, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.25,
  });
  s.addNotes("Illustrative shape from hourly/sample_hourly_data.csv (a 10-day format reference, not real production history) — used here only to show what an hourly demand curve looks like, not as an accuracy claim.");
}

/* ═════ HOURLY: HOW IT DECIDES ═════ */
{
  const s = slide("How the split decides which shape to use");
  lede(s, "The same measured-not-assumed discipline as the rest of this model — just applied to shape instead of level.");

  const steps = [
    ["1", "Look at the last 90 days", "Average the hourly SHARE of the day for the same weekday type (weekday / Friday / Saturday / Sunday), excluding any holiday windows.", TEAL],
    ["2", "Check if a holiday looks different", "Compare that holiday-day's historical shape against its own normal-day baseline, using Total Variation Distance — the same kind of measured comparison used everywhere else in this model.", GOLD],
    ["3", "Pick the closer match", "Close enough → inherit the weekday/weekend shape. Genuinely different → use a recency-weighted average of that exact holiday-day's own past occurrences.", GREEN],
  ];
  let x = 0.6;
  steps.forEach(([n, t, b, c]) => {
    card(s, x, 1.9, 3.93, 3.2, LIGHT);
    circle(s, x + 0.3, 2.15, 0.55, n, c, WHITE, 18);
    s.addText(t, { x: x + 0.3, y: 2.85, w: 3.35, h: 0.65, fontFace: HEAD, fontSize: 15.5, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 3.55, w: 3.35, h: 1.4, fontFace: BODY, fontSize: 11.5, color: GREY, margin: 0, lineSpacingMultiple: 1.2 });
    x += 4.09;
  });

  card(s, 0.6, 5.3, 12.12, 1.4, "FBF6E9");
  s.addText("Nothing here is a trained model", { x: 0.95, y: 5.46, w: 11.4, h: 0.35, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Every number the split uses is a directly-measured average from real history, chosen by a distance threshold — not a prediction from a fitted model. The next slide explains why that was the deliberate choice.", {
    x: 0.95, y: 5.82, w: 11.4, h: 0.75, fontFace: BODY, fontSize: 12.5, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("Mirrors hourly/analyze.py's TVD decision cascade: TVD<0.05 (+consistent match) -> inherit DOW; TVD<0.10 -> inherit DOW, flagged for review; TVD>=0.10 or inconsistent -> holiday-specific profile.");
}

/* ═════ HOURLY: WHY NOT A MODEL ═════ */
{
  const s = slide("Why not just train a model for this too?");
  lede(s, "The working hypothesis this design is built on — and how it will be checked.");

  card(s, 0.6, 1.85, 5.95, 2.5, "E9F4EF");
  circle(s, 0.9, 2.1, 0.52, "✓", GREEN);
  s.addText("The case for the historical shape", { x: 1.62, y: 2.13, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("An hourly shape is a slow-moving, highly repeatable pattern — Tuesday afternoon looks like Tuesday afternoon. That is exactly the situation where a direct historical average is hard to beat: there is little unexplained structure left for a model to learn that isn't already captured by \"what did this kind of day look like recently.\"", {
    x: 0.9, y: 2.55, w: 5.35, h: 1.75, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 6.75, 1.85, 5.97, 2.5, "EAF2F5");
  circle(s, 7.05, 2.1, 0.52, "?", TEAL);
  s.addText("Built to test that claim, not just assert it", { x: 7.77, y: 2.13, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("A lightweight comparison model was built specifically to challenge this — same calendar inputs, honest lag features, no peeking. It exists only to be measured against, never to produce an actual forecast. The expectation is that it will not beat the historical shape by enough to justify the extra complexity.", {
    x: 7.05, y: 2.55, w: 5.35, h: 1.75, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });

  card(s, 0.6, 4.55, 12.12, 2.15, "F7E9E9");
  s.addText("⚠ This is a hypothesis, not yet a result", { x: 0.95, y: 4.73, w: 11.4, h: 0.4, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
  s.addText("Every tool needed to check this honestly already exists and runs end-to-end — walk-forward, no leakage, applied to real daily totals, broken down by day type. What is missing is real hourly history: today the pipeline only has a 10-day format-reference sample, far too little to prove anything. The next two slides show exactly where the real result goes once it exists — and this project's own record (see the earlier slide on rejected ideas) is that assumptions like this one get overturned as often as they get confirmed. Treat this slide as the plan, not the conclusion.", {
    x: 0.95, y: 5.12, w: 11.4, h: 1.5, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2,
  });
  s.addNotes("Per explicit instruction, this slide states the expected/positive outcome as the design's working hypothesis. It is NOT a measured result -- no real hourly data exists yet. Do not let this slide stand alone without the following reserved-chart slides.");
}

/* ═════ HOURLY: RESERVED CHART 1 ═════ */
{
  const s = slide("The result, once it exists: accuracy by day type");
  lede(s, "RESERVED CHART SLOT — this is where the real comparison goes.");

  reservedChart(s, 0.6, 1.85, 12.1, 4.35,
    "Historical shape vs. comparison model — MAPE by day type",
    "deck_mape_by_daytype.png");

  card(s, 0.6, 6.35, 12.12, 0.85, LIGHT);
  s.addText("Generate with: uv run python long_history/hourly/validate.py --holdout-days 90  →  uv run python long_history/hourly/deck_charts.py", {
    x: 0.9, y: 6.35, w: 11.5, h: 0.85, fontFace: "Courier New", fontSize: 11, color: NAVY, valign: "middle", margin: 0,
  });
  s.addNotes("deck_charts.py refuses to produce a trustworthy version of this chart under 30 holdout days, and visibly watermarks any chart built from fewer -- so a mechanism-check chart can never be mistaken for evidence if someone forgets and pastes it here anyway.");
}

/* ═════ HOURLY: RESERVED CHART 2 ═════ */
{
  const s = slide("The result, once it exists: a real day, hour by hour");
  lede(s, "RESERVED CHART SLOT — actual hourly demand vs. the historical-shape split, for one illustrative day.");

  reservedChart(s, 0.6, 1.85, 12.1, 4.35,
    "Actual vs. split forecast — one day, 24 hours",
    "deck_actual_vs_split.png");

  card(s, 0.6, 6.35, 12.12, 0.85, LIGHT);
  s.addText("Generate with: uv run python long_history/hourly/split.py --run-date <date>  →  uv run python long_history/hourly/deck_charts.py --hourly-pred <predictions_hourly.csv>", {
    x: 0.9, y: 6.35, w: 11.5, h: 0.85, fontFace: "Courier New", fontSize: 10, color: NAVY, valign: "middle", margin: 0,
  });
  s.addNotes("Pick a holdout day for --run-date so the chart shows an honest, never-seen prediction. Overlay the real actual hourly values by hand until deck_charts.py is extended to read them from hourly_demand.csv directly.");
}

/* ═════ SECTION 5 ═════ */
section("Running it, and what it cannot do",
  "How to use the pipeline — and the limits you should know before relying on it.",
  "Close on operations and honesty.");

/* ═════ THE PIPELINE ═════ */
{
  const s = slide("The pipeline: one command, three steps");
  lede(s, "Each step must pass before the next runs, so you cannot forecast from bad data.");

  const steps = [
    ["1", "Validate", "Checks the data file for gaps, duplicates and impossible values.", "A single missing day silently corrupts every backward-looking input. This is a hard gate.", TEAL],
    ["2", "Evaluate", "Measures accuracy over a period and lead time you choose.", "Produces the charts in this deck, plus a per-holiday breakdown.", GOLD],
    ["3", "Forecast", "Produces the real 28-day forecast from the latest data.", "Writes predictions, a chart, and a record of every setting used.", GREEN],
  ];
  let x = 0.6;
  steps.forEach(([n, t, b, why, c]) => {
    card(s, x, 1.85, 3.93, 3.1, LIGHT);
    circle(s, x + 0.3, 2.1, 0.55, n, c, WHITE, 18);
    s.addText(t, { x: x + 0.3, y: 2.8, w: 3.35, h: 0.4, fontFace: HEAD, fontSize: 18, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 3.25, w: 3.35, h: 0.85, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, lineSpacingMultiple: 1.2 });
    s.addText(why, { x: x + 0.3, y: 4.1, w: 3.35, h: 0.75, fontFace: BODY, fontSize: 11, color: GREY, italic: true, margin: 0, lineSpacingMultiple: 1.15 });
    x += 4.09;
  });

  s.addShape(pres.ShapeType.roundRect, { x: 0.6, y: 5.2, w: 12.12, h: 0.72, rectRadius: 0.06, fill: { color: NAVY } });
  s.addText("uv run python long_history/run_pipeline.py", {
    x: 0.9, y: 5.2, w: 11.5, h: 0.72, fontFace: "Courier New", fontSize: 14, color: "9FE0C8", valign: "middle", margin: 0,
  });
  card(s, 0.6, 6.05, 12.12, 1.0, "EAF2F5");
  s.addText("The feature table takes about four minutes to build on eleven years, so it is cached. Later runs load it in a tenth of a second — and the cache invalidates itself automatically if the feature code changes.", {
    x: 0.95, y: 6.05, w: 11.4, h: 1.0, fontFace: BODY, fontSize: 12.5, color: NAVY, valign: "middle", margin: 0,
  });
  s.addNotes("Individual steps: --steps validate | evaluate | forecast. Also experiment, for comparing settings.");
}

/* ═════ LIMITATIONS ═════ */
{
  const s = slide("What it cannot do — read this before relying on it");
  lede(s, "Stated plainly, because knowing the limits is part of using it safely.");
  const lims = [
    ["The high/low range is too narrow", "The P10–P90 band should contain the real answer 80% of the time. Measured, it manages 46–54%. Treat it as a rough signal, not a guarantee.", RED],
    ["It cannot see one-off events", "A concert, a competitor closing, a policy change. If it is not in the calendar and not in the past, the model does not know.", GOLD],
    ["Typhoons are not predicted", "Deliberately. RAGASA shut the floor in September 2025 and the forecast missed by 468%; TORAJI did not and demand was normal. Similar storms, opposite outcomes — the deciding factor is a human decision, so a person applies it manually.", GOLD],
    ["The holiday calendar ends in 2026", "The underlying calendar library has no 2027 dates. Forecasts made late in 2026 will reach past it. A known, dated deadline.", RED],
  ];
  let y = 1.8;
  lims.forEach(([t, b, c]) => {
    card(s, 0.6, y, 12.12, 1.25, c === RED ? "F7E9E9" : "FBF6E9");
    circle(s, 0.9, y + 0.35, 0.52, "!", c, WHITE, 17);
    s.addText(t, { x: 1.62, y: y + 0.18, w: 10.85, h: 0.38, fontFace: HEAD, fontSize: 16, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: 1.62, y: y + 0.56, w: 10.85, h: 0.62, fontFace: BODY, fontSize: 12.5, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.35;
  });
  s.addNotes("The interval-coverage problem is a correctness issue, not an accuracy one, but people may be sizing risk off that band.");
}

/* ═════ HONEST STATUS ═════ */
{
  const s = slide("Is it better than the short-memory model? Honestly: partly");
  lede(s, "Reported as measured, including the parts that did not work.");

  card(s, 0.6, 1.8, 5.95, 2.35, "E9F4EF");
  circle(s, 0.9, 2.02, 0.5, "✓", GREEN);
  s.addText("What it clearly wins on", { x: 1.55, y: 2.05, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Removes catastrophic single-day errors (27.7% → 3.2%)", options: { bullet: true, breakLine: true } },
    { text: "Has seen ~10 Chinese New Years, not 2", options: { bullet: true, breakLine: true } },
    { text: "Fixes the Ching Ming calendar bug", options: { bullet: true, breakLine: true } },
    { text: "Cannot affect the live model — fully self-contained", options: { bullet: true } },
  ], { x: 0.95, y: 2.55, w: 5.3, h: 1.5, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, paraSpaceAfter: 4 });

  card(s, 6.75, 1.8, 5.97, 2.35, "FBF6E9");
  circle(s, 7.05, 2.02, 0.5, "~", GOLD, NAVY);
  s.addText("What it does not win on", { x: 7.7, y: 2.05, w: 4.7, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  s.addText([
    { text: "Only about 0.3 percentage points better on average", options: { bullet: true, breakLine: true } },
    { text: "Improves most test windows but makes two worse", options: { bullet: true, breakLine: true } },
    { text: "Does not pass the strict \"nothing may get worse\" bar", options: { bullet: true, breakLine: true } },
    { text: "Slower: eleven years takes minutes to prepare", options: { bullet: true } },
  ], { x: 7.1, y: 2.55, w: 5.3, h: 1.5, fontFace: BODY, fontSize: 12, color: NAVY, margin: 0, paraSpaceAfter: 4 });

  s.addText("Ideas tested on this pipeline and rejected", { x: 0.6, y: 4.35, w: 12.1, h: 0.4, fontFace: HEAD, fontSize: 17, bold: true, color: NAVY, margin: 0 });
  const rej = [
    ["Blocking the contaminated year-ago comparison", "Looked convincing on one Chinese New Year. Tested against six events it failed — and the long-history model had already solved the problem by itself, without the patch."],
    ["Swapping in a GAM (a smoother model)", "Worse at all nine forecast distances tested. Also no better on holidays, so a holiday-only hybrid was ruled out too."],
  ];
  let x = 0.6;
  rej.forEach(([t, b]) => {
    card(s, x, 4.85, 5.95, 1.95, "F7E9E9");
    s.addText(t, { x: x + 0.3, y: 5.05, w: 5.35, h: 0.4, fontFace: BODY, fontSize: 13, bold: true, color: NAVY, margin: 0 });
    s.addText(b, { x: x + 0.3, y: 5.45, w: 5.35, h: 1.25, fontFace: BODY, fontSize: 12, color: GREY, margin: 0, lineSpacingMultiple: 1.15 });
    x += 6.17;
  });
  s.addNotes("The single-event-versus-six-events lesson is the most transferable thing in this deck.");
}

/* ═════ CLOSING ═════ */
{
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Five things to remember", {
    x: 0.9, y: 0.65, w: 11.5, h: 0.8, fontFace: HEAD, fontSize: 36, bold: true, color: WHITE, margin: 0,
  });
  const pts = [
    ["Eleven years only helps if old days actually count", "The half-life setting is the change. Without it, a day from 2016 counts 1/35,000th of today and the extra history does nothing."],
    ["No peeking", "There are 28 separate models because each forecast distance may legally see a different amount of history. Every chart here obeys it."],
    ["The win is fewer disasters, not a better average", "About 0.3 points better typically — but it turned a 27.7% miss into 3.2% on the worst measured day."],
    ["Half the model is holidays", "And Chinese New Year gets its own correction, because it cannot be learned from one or two examples."],
    ["Measure before believing", "Most sensible-sounding improvements tested here made things worse. A result from a single event is not a result."],
  ];
  let y = 1.7;
  pts.forEach(([t, b], i) => {
    circle(s, 0.9, y + 0.05, 0.55, String(i + 1), GOLD, NAVY, 18);
    s.addText(t, { x: 1.75, y, w: 10.6, h: 0.38, fontFace: HEAD, fontSize: 17.5, bold: true, color: WHITE, margin: 0 });
    s.addText(b, { x: 1.75, y: y + 0.38, w: 10.6, h: 0.6, fontFace: BODY, fontSize: 12.5, color: "AEC3D6", margin: 0, lineSpacingMultiple: 1.15 });
    y += 1.06;
  });
  s.addNotes("Close on the measurement discipline point.");
}

pres.writeFile({ fileName: path.join(__dirname, "Demand_Forecast_LongHistory.pptx") })
  .then(f => console.log("wrote", f));
