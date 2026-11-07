"""Make one web page for hand-labelling why real Reject photos are bad. It works offline.

The page holds EyePACS photos, so it stays in outputs/ and must not be shared or committed.
"""
import base64
import json
from pathlib import Path

import cv2
import pandas as pd

N_PHOTOS = 100
SIDE = 480
SEED = 0

CHOICES = [
    ["blur", "Blurred: vessels look soft or smeared"],
    ["uneven_light", "Uneven light: one side much darker or brighter"],
    ["low_contrast", "Low contrast: washed out, flat, hazy"],
    ["too_dark", "Too dark: dark all over"],
    ["cut_off", "Cut off: retina badly framed or partly missing"],
    ["artefact", "Artefact: lashes, reflections, flash spots, marks"],
    ["other", "Something else"],
    ["looks_fine", "Looks fine to me"],
]

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Label photos</title>
<style>
body { font-family: sans-serif; margin: 0 auto; max-width: 980px; padding: 16px; background: #1c1c1c; color: #eee; }
.row { display: flex; gap: 20px; flex-wrap: wrap; }
img { width: 480px; max-width: 100%; background: #000; }
button { display: block; width: 100%; margin: 0 0 6px; padding: 9px; text-align: left; font-size: 15px;
  background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; cursor: pointer; }
button.on { background: #2d6a4f; border-color: #74c69d; }
.nav button { display: inline-block; width: auto; margin-right: 8px; }
.box { min-width: 300px; flex: 1; }
</style></head><body>
<h3 id="where"></h3>
<div class="row"><img id="photo" alt="retinal photo"><div class="box"><div id="choices"></div>
<p>Pick every fault you can see. Keys 1 to 8 pick, Enter goes to the next photo.</p>
<div class="nav"><button id="back">Back</button><button id="next">Next</button><button id="save">Download labels</button></div>
<p id="done"></p></div></div>
<script>
const PHOTOS = __PHOTOS__;
const CHOICES = __CHOICES__;
const KEY = "reject_labels";
let answers = {};
try { answers = JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) {}
let at = 0;
while (at < PHOTOS.length - 1 && answers[PHOTOS[at].name]) { at++; }

function remember() {
  try { localStorage.setItem(KEY, JSON.stringify(answers)); } catch (e) {}
}
function show() {
  const photo = PHOTOS[at];
  document.getElementById("where").textContent = "Photo " + (at + 1) + " of " + PHOTOS.length;
  document.getElementById("photo").src = "data:image/jpeg;base64," + photo.data;
  const picked = answers[photo.name] || [];
  const box = document.getElementById("choices");
  box.innerHTML = "";
  CHOICES.forEach((choice, index) => {
    const button = document.createElement("button");
    button.textContent = (index + 1) + ". " + choice[1];
    if (picked.includes(choice[0])) { button.className = "on"; }
    button.onclick = () => toggle(index);
    box.appendChild(button);
  });
  const count = Object.values(answers).filter(list => list.length > 0).length;
  document.getElementById("done").textContent = count + " labelled so far";
}
function toggle(index) {
  const name = PHOTOS[at].name;
  const picked = answers[name] || [];
  const choice = CHOICES[index][0];
  answers[name] = picked.includes(choice) ? picked.filter(item => item !== choice) : picked.concat([choice]);
  remember();
  show();
}
function go(step) {
  at = Math.min(PHOTOS.length - 1, Math.max(0, at + step));
  show();
}
function download() {
  let text = "image,labels\\n";
  PHOTOS.forEach(photo => { text += photo.name + "," + (answers[photo.name] || []).join("+") + "\\n"; });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([text], { type: "text/csv" }));
  link.download = "reject_labels.csv";
  link.click();
}
document.getElementById("back").onclick = () => go(-1);
document.getElementById("next").onclick = () => go(1);
document.getElementById("save").onclick = download;
document.addEventListener("keydown", event => {
  const number = parseInt(event.key);
  if (number >= 1 && number <= CHOICES.length) { toggle(number - 1); }
  if (event.key === "Enter") { event.preventDefault(); go(1); }
});
show();
</script></body></html>
"""


def main():
    eyeq = pd.read_csv("data/splits/eyeq.csv", dtype={"patient": str})
    val = eyeq[(eyeq["split"] == "val") & (eyeq["quality"] == 2)]
    chosen = val.sample(N_PHOTOS, random_state=SEED)

    photos = []
    for name in chosen["image"]:
        image = cv2.imread(f"data/eyepacs/cache512/{name}.png")
        small = cv2.resize(image, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        _, encoded = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 85])
        text = base64.b64encode(encoded.tobytes()).decode("ascii")
        photos.append({"name": name, "data": text})

    page = PAGE.replace("__PHOTOS__", json.dumps(photos)).replace("__CHOICES__", json.dumps(CHOICES))
    out = Path("outputs/phase4/label_rejects.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out, round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
