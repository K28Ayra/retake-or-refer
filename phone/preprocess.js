// The same steps as retake/compress.py model_input, written out by hand so the phone
// gives the network exactly what training gave it. Images are flat RGB byte arrays.

const CACHE_SIDE = 512;
const MODEL_SIDE = 384;
const DARK = 10;

// Python rounds halves to the even number, and the sizes have to match it
function pythonRound(value) {
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

// OpenCV rounds halves up when it shrinks by a whole number, and to even otherwise
function toByte(value, halvesUp) {
  const rounded = halvesUp ? Math.floor(value + 0.5) : pythonRound(value);
  return Math.min(255, Math.max(0, rounded));
}

// OpenCV's grey uses these fixed-point weights, not plain 0.299 / 0.587 / 0.114
function grey(red, green, blue) {
  return (red * 4899 + green * 9617 + blue * 1868 + 8192) >> 14;
}

function rgbaToRgb(rgba, width, height) {
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < width * height; i++) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return rgb;
}

function cropToRetina(image) {
  const { pixels, width, height } = image;
  let top = height;
  let bottom = -1;
  let left = width;
  let right = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const at = (y * width + x) * 3;
      if (grey(pixels[at], pixels[at + 1], pixels[at + 2]) > DARK) {
        if (y < top) top = y;
        if (y > bottom) bottom = y;
        if (x < left) left = x;
        if (x > right) right = x;
      }
    }
  }
  if (bottom < 0) return image;

  const newWidth = right - left + 1;
  const newHeight = bottom - top + 1;
  const cropped = new Uint8Array(newWidth * newHeight * 3);
  for (let y = 0; y < newHeight; y++) {
    const from = ((y + top) * width + left) * 3;
    cropped.set(pixels.subarray(from, from + newWidth * 3), y * newWidth * 3);
  }
  return { pixels: cropped, width: newWidth, height: newHeight };
}

// For each output position, which input positions it covers and by how much.
// Shrinking only: this is OpenCV's INTER_AREA when the image gets smaller.
// Kept in flat typed arrays, because this loop runs millions of times on a phone photo.
function areaWeights(inSize, outSize) {
  const scale = inSize / outSize;
  const first = new Int32Array(outSize);
  const count = new Int32Array(outSize);
  const positions = [];
  const amounts = [];
  for (let out = 0; out < outSize; out++) {
    const start = out * scale;
    const end = start + scale;
    first[out] = positions.length;
    for (let at = Math.floor(start); at < Math.ceil(end) && at < inSize; at++) {
      const overlap = Math.min(end, at + 1) - Math.max(start, at);
      if (overlap > 1e-9) {
        positions.push(at);
        amounts.push(overlap / scale);
      }
    }
    count[out] = positions.length - first[out];
  }
  return { first, count, at: Int32Array.from(positions), weight: Float32Array.from(amounts) };
}

function areaResize(image, outWidth, outHeight) {
  const { pixels, width, height } = image;
  const wholeNumber = width % outWidth === 0 && height % outHeight === 0;
  const across = areaWeights(width, outWidth);
  const down = areaWeights(height, outHeight);
  const rowLength = outWidth * 3;

  // OpenCV adds up in 32-bit floats, and matching it matters for values that land near a half
  const rows = new Float32Array(height * rowLength);
  for (let y = 0; y < height; y++) {
    const inRow = y * width * 3;
    const outRow = y * rowLength;
    for (let x = 0; x < outWidth; x++) {
      let red = 0;
      let green = 0;
      let blue = 0;
      const last = across.first[x] + across.count[x];
      for (let k = across.first[x]; k < last; k++) {
        const from = inRow + across.at[k] * 3;
        const weight = across.weight[k];
        red = Math.fround(red + Math.fround(pixels[from] * weight));
        green = Math.fround(green + Math.fround(pixels[from + 1] * weight));
        blue = Math.fround(blue + Math.fround(pixels[from + 2] * weight));
      }
      rows[outRow + x * 3] = red;
      rows[outRow + x * 3 + 1] = green;
      rows[outRow + x * 3 + 2] = blue;
    }
  }

  // then down each column, a whole output row at a time
  const output = new Uint8Array(outHeight * rowLength);
  const sums = new Float32Array(rowLength);
  for (let y = 0; y < outHeight; y++) {
    sums.fill(0);
    const last = down.first[y] + down.count[y];
    for (let k = down.first[y]; k < last; k++) {
      const from = down.at[k] * rowLength;
      const weight = down.weight[k];
      for (let i = 0; i < rowLength; i++) {
        sums[i] = sums[i] + Math.fround(rows[from + i] * weight);
      }
    }
    const to = y * rowLength;
    for (let i = 0; i < rowLength; i++) {
      output[to + i] = toByte(sums[i], wholeNumber);
    }
  }
  return { pixels: output, width: outWidth, height: outHeight };
}

function shrinkLongSide(image, longSide) {
  const { width, height } = image;
  if (Math.max(width, height) <= longSide) return image;
  const scale = longSide / Math.max(width, height);
  const newWidth = Math.max(1, pythonRound(width * scale));
  const newHeight = Math.max(1, pythonRound(height * scale));
  return areaResize(image, newWidth, newHeight);
}

function padToSquare(image) {
  const { pixels, width, height } = image;
  const size = Math.max(width, height);
  const square = new Uint8Array(size * size * 3);
  const top = Math.floor((size - height) / 2);
  const left = Math.floor((size - width) / 2);
  for (let y = 0; y < height; y++) {
    const from = y * width * 3;
    square.set(pixels.subarray(from, from + width * 3), ((y + top) * size + left) * 3);
  }
  return { pixels: square, width: size, height: size };
}

// Photo in, the 384 x 384 RGB bytes the ONNX file takes out
function modelInput(image) {
  const cropped = cropToRetina(image);
  const smaller = shrinkLongSide(cropped, CACHE_SIDE);
  const square = padToSquare(smaller);
  if (square.width === MODEL_SIDE) return square;
  return areaResize(square, MODEL_SIDE, MODEL_SIDE);
}

if (typeof module !== "undefined") {
  module.exports = { modelInput, cropToRetina, areaResize, padToSquare, rgbaToRgb, MODEL_SIDE };
}
