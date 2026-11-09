// The Phase 7 rule, the same as retake/decide.py. The numbers come from decision.json,
// which scripts/63_serve_phone.py writes from configs/decision.yaml.

const QUALITY_NAMES = ["good", "usable", "reject"];

function softmaxOf(values) {
  const top = Math.max(...values);
  const powers = values.map((value) => Math.exp(value - top));
  const total = powers.reduce((a, b) => a + b, 0);
  return powers.map((value) => value / total);
}

function decide(qualityLogits, referableLogit, config) {
  const scaled = Array.from(qualityLogits).map((value) => value / config.quality_temperature);
  const quality = softmaxOf(scaled);
  const qualityClass = quality.indexOf(Math.max(...quality));
  const platt = config.referable_platt;
  const referable = 1 / (1 + Math.exp(-(platt.scale * referableLogit + platt.shift)));
  const limits = config.thresholds;

  // a bad photo can refer, but it can never clear
  let decision = "retake";
  let why = "";
  if (qualityClass === 0) {
    decision = referable >= limits.good_refer ? "refer" : "clear";
  } else if (qualityClass === 1) {
    if (referable >= limits.usable_refer) decision = "refer";
    else if (referable < limits.usable_clear) decision = "clear";
    else why = "borderline disease score on a usable photo";
  } else if (referable >= limits.reject_refer) {
    decision = "refer";
  } else {
    why = "poor photo";
  }

  let sure = quality[2];
  if (decision === "refer") sure = referable;
  if (decision === "clear") sure = 1 - referable;
  return { decision, why, confidence: sure, referable, quality, qualityName: QUALITY_NAMES[qualityClass] };
}

if (typeof module !== "undefined") {
  module.exports = { decide };
}
