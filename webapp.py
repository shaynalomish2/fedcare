"""
FedCare Web App: enter a patient's measurements in your browser and get a prediction
====================================================================================
Uses the SAME model code as fedcare.py (it imports it), so you need no new libraries.

Run:   python webapp.py
Then your browser opens at http://127.0.0.1:8000  (press Ctrl+C in the terminal to stop)

IMPORTANT: this is a student demo of federated learning. It is NOT a medical device
and must never be used to make decisions about a real person's health.
"""

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import fedcare as fc  # <- your existing code: model, hospital split, federated training

FEATURE_NAMES = [str(n) for n in load_breast_cancer().feature_names]  # 30 names, fixed order

MODEL_LABELS = {
    "priority": "Federated - Priority-aware (your novelty)",
    "fedavg": "Federated - FedAvg",
    "central": "Centralized (no privacy, for comparison)",
}


# ----------------------------------------------------------------------------
# TRAIN THE MODELS ONCE, WHEN THE APP STARTS (same steps as fedcare.py)
# ----------------------------------------------------------------------------
def build_models():
    X, y = fc.load_builtin()  # 569 patients, label 1 = malignant
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=fc.SEED
    )
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)

    rng = np.random.default_rng(fc.SEED)
    hospitals = fc.split_into_hospitals(Xtr, y_train, fc.NUM_HOSPITALS, fc.NON_IID, fc.DIRICHLET_ALPHA, rng)
    d = Xtr.shape[1]

    central = fc.local_train(np.zeros(d + 1), Xtr, y_train, fc.ROUNDS * fc.LOCAL_EPOCHS, fc.LEARNING_RATE, rng)
    fedavg, _ = fc.run_federated(hospitals, Xte, y_test, fc.ROUNDS, False, rng)
    prio, _ = fc.run_federated(hospitals, Xte, y_test, fc.ROUNDS, True, rng)

    params = {"central": central, "fedavg": fedavg, "priority": prio}
    metrics = {k: fc.evaluate(p, Xte, y_test) for k, p in params.items()}

    # Example patients taken from the held-out TEST set (the models never trained on them)
    examples = {
        "Average patient (training average)": {"values": X_train.mean(axis=0).tolist(), "truth": None},
        "Example patient A (from test set)": {"values": X_test[np.where(y_test == 1)[0][0]].tolist(), "truth": "malignant"},
        "Example patient B (from test set)": {"values": X_test[np.where(y_test == 0)[0][0]].tolist(), "truth": "benign"},
    }
    return {
        "scaler": scaler,
        "params": params,
        "metrics": metrics,
        "examples": examples,
        "train_min": X_train.min(axis=0),
        "train_max": X_train.max(axis=0),
    }


def predict(state, values, model_key):
    x = np.array(values, dtype=float).reshape(1, -1)
    p = float(fc.predict_proba(state["params"][model_key], state["scaler"].transform(x))[0])
    warnings = []
    lo, hi = state["train_min"], state["train_max"]
    span = np.maximum(hi - lo, 1e-9)
    far = [FEATURE_NAMES[i] for i in range(30) if x[0, i] < lo[i] - 0.5 * span[i] or x[0, i] > hi[i] + 0.5 * span[i]]
    if far:
        warnings.append(
            "These values are far outside anything the model saw in training, so the result "
            "is unreliable: " + ", ".join(far[:5]) + ("..." if len(far) > 5 else "")
        )
    return {"probability": p, "label": "malignant" if p >= 0.5 else "benign", "warnings": warnings}


# ----------------------------------------------------------------------------
# THE WEB PAGE
# ----------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FedCare - Disease Prediction Demo</title>
<style>
  body{font-family:system-ui,-apple-system,Segoe UI,Arial,sans-serif;margin:0;background:#f3f8f7;color:#16302f}
  header{background:#16302f;color:#fff;padding:22px 32px}
  header h1{margin:0 0 4px;font-size:24px} header p{margin:0;color:#a9d6d0;font-size:14px}
  main{max-width:1100px;margin:24px auto;padding:0 20px}
  .warn{background:#fff4e0;border:1px solid #f0c36d;border-radius:8px;padding:12px 16px;margin-bottom:18px;font-size:14px}
  .card{background:#fff;border-radius:10px;padding:20px;margin-bottom:18px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
  .card h2{margin:0 0 12px;font-size:17px}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-end}
  label{font-size:13px;color:#37514d;display:block;margin-bottom:3px}
  select,input,textarea,button{font:inherit}
  select,input[type=number],textarea{border:1px solid #b9cfcb;border-radius:6px;padding:7px 9px;background:#fff}
  textarea{width:100%;box-sizing:border-box;height:56px}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px 14px}
  .grid input{width:100%;box-sizing:border-box}
  button{background:#028090;color:#fff;border:0;border-radius:6px;padding:10px 20px;cursor:pointer;font-weight:600}
  button.sec{background:#e3f2f0;color:#16302f}
  button:hover{filter:brightness(1.08)}
  #result{display:none}
  .verdict{font-size:22px;font-weight:700;margin:4px 0 10px}
  .bar{height:22px;background:#e3f2f0;border-radius:11px;overflow:hidden}
  .bar>div{height:100%;background:#c0392b;width:0}
  .small{font-size:13px;color:#5c7a76}
  table{border-collapse:collapse;font-size:14px} td,th{padding:5px 14px;text-align:left;border-bottom:1px solid #e3eeec}
</style></head><body>
<header><h1>FedCare: Disease Prediction Demo</h1>
<p>Breast cancer prediction from 30 cell measurements, using a model trained with federated learning</p></header>
<main>
  <div class="warn"><b>Student demo only.</b> This is a class project showing federated learning. It is <b>not</b> a medical
  tool. Never use it to decide anything about a real person's health. If you are worried about your health, see a doctor.</div>

  <div class="card"><h2>1. Choose a model and fill in the patient</h2>
    <div class="row">
      <div><label>Model</label><select id="model"></select></div>
      <div><label>Start from an example</label><select id="example"></select></div>
      <button class="sec" id="loadEx">Load example</button>
    </div>
    <p class="small" style="margin:14px 0 6px">Or paste one row of 30 numbers (from Excel or a CSV, same column order as the dataset) and click Fill form:</p>
    <textarea id="paste" placeholder="17.99, 10.38, 122.8, 1001, 0.1184, ..."></textarea>
    <div style="margin-top:8px"><button class="sec" id="fill">Fill form</button></div>
  </div>

  <div class="card"><h2>2. The 30 measurements</h2>
    <div id="groups"></div>
    <div style="margin-top:16px"><button id="go">Predict</button></div>
  </div>

  <div class="card" id="result"><h2>3. Result</h2>
    <div class="verdict" id="verdict"></div>
    <div class="bar"><div id="barfill"></div></div>
    <p class="small" id="detail"></p>
    <div id="warnbox" class="warn" style="display:none"></div>
  </div>

  <div class="card"><h2>How good is each model? (on 114 held-out test patients)</h2>
    <table id="metrics"></table>
    <p class="small">"Recall" = of the truly sick patients, how many the model caught.</p>
  </div>
</main>
<script>
const DATA = __DATA__;
const $ = id => document.getElementById(id);

// model + example drop-downs
for (const [k, v] of Object.entries(DATA.models)) $("model").add(new Option(v, k));
for (const name of Object.keys(DATA.examples)) $("example").add(new Option(name, name));

// 30 input boxes in three groups
const groupTitles = ["Mean values (average over all cell nuclei)", "Error values (how much they vary)", "Worst values (the most extreme nuclei)"];
groupTitles.forEach((title, g) => {
  const h = document.createElement("div");
  h.innerHTML = "<h3 style='margin:14px 0 8px;font-size:15px'>" + title + "</h3><div class='grid' id='g" + g + "'></div>";
  $("groups").appendChild(h);
  for (let i = g * 10; i < g * 10 + 10; i++) {
    const w = document.createElement("div");
    w.innerHTML = "<label>" + DATA.features[i] + "</label><input type='number' step='any' id='f" + i + "'>";
    $("g" + g).appendChild(w);
  }
});

function setValues(vals) { vals.forEach((v, i) => $("f" + i).value = String(Number(Number(v).toPrecision(6)))); }
function getValues() { return DATA.features.map((_, i) => $("f" + i).value.trim() === "" ? NaN : Number($("f" + i).value)); }

$("loadEx").onclick = () => setValues(DATA.examples[$("example").value].values);
$("fill").onclick = () => {
  const nums = $("paste").value.split(/[\\s,;]+/).filter(s => s !== "").map(Number);
  if (nums.length !== 30 || nums.some(isNaN)) { alert("Please paste exactly 30 numbers (you pasted " + nums.length + ")."); return; }
  setValues(nums);
};
$("go").onclick = async () => {
  const vals = getValues();
  if (vals.some(isNaN)) { alert("Please fill in all 30 boxes first (or click 'Load example')."); return; }
  const r = await fetch("/predict", {method: "POST", headers: {"Content-Type": "application/json"},
                                     body: JSON.stringify({values: vals, model: $("model").value})});
  const out = await r.json();
  if (out.error) { alert(out.error); return; }
  const pct = (out.probability * 100).toFixed(1);
  const mal = out.label === "malignant";
  $("result").style.display = "block";
  $("verdict").textContent = mal ? "Model says: likely MALIGNANT" : "Model says: likely BENIGN";
  $("verdict").style.color = mal ? "#c0392b" : "#1e8449";
  $("barfill").style.width = pct + "%";
  $("detail").textContent = "Estimated probability of malignancy: " + pct + "% (model: " + DATA.models[$("model").value] + "). A probability this close to 50% means the model is unsure.";
  const ex = DATA.examples[$("example").value];
  const wb = $("warnbox");
  if (out.warnings.length) { wb.style.display = "block"; wb.textContent = out.warnings.join(" "); } else wb.style.display = "none";
  $("result").scrollIntoView({behavior: "smooth"});
};

// metrics table
let t = "<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th></tr>";
for (const [k, label] of Object.entries(DATA.models)) {
  const m = DATA.metrics[k];
  t += "<tr><td>" + label + "</td><td>" + m.accuracy.toFixed(3) + "</td><td>" + m.precision.toFixed(3) + "</td><td>" + m.recall.toFixed(3) + "</td><td>" + m.f1.toFixed(3) + "</td></tr>";
}
$("metrics").innerHTML = t;
$("loadEx").click();
</script></body></html>
"""


def make_handler(state):
    page_data = {
        "features": FEATURE_NAMES,
        "models": MODEL_LABELS,
        "examples": state["examples"],
        "metrics": state["metrics"],
    }
    html = PAGE.replace("__DATA__", json.dumps(page_data).replace("</", "<\\/")).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, html, "text/html; charset=utf-8")
            else:
                self._send(404, b"Not found", "text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b"Not found", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", 0))
                if n <= 0 or n > 20000:
                    raise ValueError("bad request size")
                body = json.loads(self.rfile.read(n))
                values, model = body["values"], body["model"]
                if model not in MODEL_LABELS:
                    raise ValueError("unknown model")
                if not isinstance(values, list) or len(values) != 30:
                    raise ValueError("Need exactly 30 numbers.")
                values = [float(v) for v in values]
                if not all(np.isfinite(values)):
                    raise ValueError("All 30 values must be real numbers.")
                out = predict(state, values, model)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self._send(400, json.dumps({"error": str(e)}).encode(), "application/json")
            self._send(200, json.dumps(out).encode(), "application/json")

        def log_message(self, *args):  # keep the terminal quiet
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    print("Training the models (a few seconds)...")
    state = build_models()
    print("Model accuracy on test patients:", {k: round(v["accuracy"], 3) for k, v in state["metrics"].items()})

    server = None
    for port in range(args.port, args.port + 10):  # try the next ports if 8000 is busy
        try:
            server = HTTPServer(("127.0.0.1", port), make_handler(state))
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit("Could not find a free port. Close other programs and try again.")

    url = f"http://127.0.0.1:{server.server_address[1]}"
    print(f"\nFedCare web app is running at {url}")
    print("Press Ctrl+C in this terminal to stop it.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
