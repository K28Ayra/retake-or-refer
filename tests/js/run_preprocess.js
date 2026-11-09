// node tests/js/run_preprocess.js in.rgb width height out.rgb
const fs = require("fs");
const path = require("path");
const { modelInput } = require(path.join(__dirname, "..", "..", "phone", "preprocess.js"));

const [inPath, width, height, outPath] = process.argv.slice(2);
const pixels = new Uint8Array(fs.readFileSync(inPath));
const result = modelInput({ pixels, width: Number(width), height: Number(height) });
fs.writeFileSync(outPath, Buffer.from(result.pixels));
