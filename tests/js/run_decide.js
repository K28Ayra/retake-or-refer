// node tests/js/run_decide.js cases.json decision.json out.json
const fs = require("fs");
const path = require("path");
const { decide } = require(path.join(__dirname, "..", "..", "phone", "decide.js"));

const [casesPath, configPath, outPath] = process.argv.slice(2);
const cases = JSON.parse(fs.readFileSync(casesPath));
const config = JSON.parse(fs.readFileSync(configPath));
const answers = cases.map((one) => decide(one.quality, one.referable, config));
fs.writeFileSync(outPath, JSON.stringify(answers));
