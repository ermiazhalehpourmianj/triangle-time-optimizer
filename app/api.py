"""
FastAPI app for the triangle-time system.

Endpoints:
- GET  /          → Triangle Console (HTML UI)
- GET  /health    → Health check
- GET  /self-test → End-to-end smoke test
- POST /predict_time
- POST /log_task

This is what you run with gunicorn or uvicorn and (later) deploy.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Make src/ importable ----------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from triangle_time.schema import Task, ModelParams  # noqa: E402
from triangle_time.triangle_model import (  # noqa: E402
    predict_time_for_task,
    update_task_proportions,
)
from triangle_time.data_io import (  # noqa: E402
    load_tasks_from_csv,
    save_tasks_to_csv,
)

# --- Config-ish constants ----------------------------------------------------

DEFAULT_PARAMS_PATH = REPO_ROOT / "model_params.json"
DEFAULT_TASK_LOG_CSV = REPO_ROOT / "data" / "tasks_logged.csv"

MODEL_PARAMS_PATH = Path(
    os.getenv("TT_MODEL_PARAMS_PATH", str(DEFAULT_PARAMS_PATH))
)

TASK_LOG_CSV_PATH = Path(
    os.getenv("TT_TASK_LOG_CSV_PATH", str(DEFAULT_TASK_LOG_CSV))
)

# FastAPI app: docs at /docs, our console UI at /
app = FastAPI(
    title="Triangle Time API",
    docs_url="/docs",
    redoc_url=None,
)


# --- Azure-style UI at root --------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def triangle_console() -> str:
    """
    Triangle Time Console – HTML UI sitting on top of the API.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Triangle Time Console</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --azure-blue: #0078d4;
      --azure-blue-dark: #005a9e;
      --bg-dark: #0b1620;
      --bg-panel: #111827;
      --bg-card: #1f2937;
      --text-main: #f9fafb;
      --text-muted: #9ca3af;
      --accent: #22c55e;
      --danger: #f97373;
      --border-soft: #374151;
      --shadow-soft: 0 14px 30px rgba(0,0,0,0.45);
      --radius-lg: 14px;
    }

    * {
      box-sizing: border-box;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body {
      margin: 0;
      padding: 0;
      background: radial-gradient(circle at top, #111827 0, #020617 55%);
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      align-items: stretch;
      justify-content: center;
    }

    .shell {
      width: 100%;
      max-width: 1200px;
      margin: 32px auto;
      padding: 24px;
      background: linear-gradient(135deg, #020617 0, #020617 40%, #020617 100%);
      border-radius: 24px;
      border: 1px solid #1f2937;
      box-shadow: var(--shadow-soft);
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
      gap: 18px;
    }

    @media (max-width: 900px) {
      .shell {
        grid-template-columns: 1fr;
        margin: 12px;
        padding: 16px;
      }
    }

    .header {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
    }

    .title-block {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .title {
      font-size: 1.4rem;
      font-weight: 650;
      letter-spacing: 0.03em;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .pill {
      padding: 4px 10px;
      border-radius: 999px;
      background: rgba(0, 120, 212, 0.15);
      border: 1px solid rgba(56, 189, 248, 0.5);
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #7dd3fc;
    }

    .subtitle {
      font-size: 0.9rem;
      color: var(--text-muted);
    }

    .badge-triangle {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 40px;
      height: 40px;
      border-radius: 12px;
      background: radial-gradient(circle at 20% 0, #38bdf8, #0f172a 55%, #22c55e);
      border: 1px solid rgba(148, 163, 184, 0.3);
      font-weight: 700;
      font-size: 1rem;
    }

    .left-panel, .right-panel {
      background: var(--bg-panel);
      border-radius: var(--radius-lg);
      border: 1px solid var(--border-soft);
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }

    .section-title {
      font-size: 0.9rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: #9ca3af;
    }

    .card {
      background: var(--bg-card);
      border-radius: 12px;
      border: 1px solid #111827;
      padding: 12px 14px;
    }

    .card + .card {
      margin-top: 8px;
    }

    .chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 6px;
    }

    .chip {
      font-size: 0.75rem;
      padding: 4px 8px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: rgba(15, 23, 42, 0.9);
      color: var(--text-muted);
    }

    .chip strong {
      color: #e5e7eb;
    }

    .equation {
      font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.78rem;
      background: #020617;
      border-radius: 8px;
      padding: 6px 8px;
      margin: 4px 0;
      border: 1px solid #1f2937;
      overflow-x: auto;
    }

    .equation span.keyword {
      color: #7dd3fc;
    }

    .equation span.sym {
      color: #f97316;
    }

    .equation span.num {
      color: #a5b4fc;
    }

    .equation span.fn {
      color: #4ade80;
    }

    .mini-heading {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #9ca3af;
      margin-bottom: 2px;
      margin-top: 4px;
    }

    .body-text {
      font-size: 0.82rem;
      color: #e5e7eb;
      line-height: 1.4;
    }

    /* Right panel */

    .form-row {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 4px;
    }

    .field label {
      font-size: 0.78rem;
      color: var(--text-muted);
    }

    .field input {
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1f2937;
      padding: 6px 8px;
      color: var(--text-main);
      font-size: 0.85rem;
      outline: none;
    }

    .field input:focus {
      border-color: var(--azure-blue);
      box-shadow: 0 0 0 1px rgba(56,189,248,0.5);
    }

    .toggle-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-top: 6px;
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .toggle-row span {
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    .toggle-row code {
      font-size: 0.72rem;
      color: #e5e7eb;
      background: #020617;
      padding: 1px 4px;
      border-radius: 4px;
    }

    .button-row {
      display: flex;
      gap: 10px;
      margin-top: 10px;
    }

    button {
      border-radius: 999px;
      border: none;
      padding: 7px 14px;
      font-size: 0.82rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: background 0.12s ease, transform 0.08s ease, box-shadow 0.12s ease;
    }

    .btn-primary {
      background: linear-gradient(135deg, var(--azure-blue) 0, #38bdf8 100%);
      color: white;
      box-shadow: 0 8px 18px rgba(56, 189, 248, 0.35);
    }

    .btn-primary:hover {
      transform: translateY(-1px);
      box-shadow: 0 12px 24px rgba(56, 189, 248, 0.45);
      background: linear-gradient(135deg, var(--azure-blue-dark) 0, #0ea5e9 100%);
    }

    .btn-ghost {
      background: transparent;
      color: var(--text-muted);
      border: 1px solid #1f2937;
    }

    .btn-ghost:hover {
      background: rgba(15,23,42,0.8);
      color: #e5e7eb;
    }

    .status-bar {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 6px;
      min-height: 18px;
    }

    .status-bar span.ok {
      color: var(--accent);
    }

    .status-bar span.err {
      color: var(--danger);
    }

    .results-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 10px;
    }

    .metric {
      background: #020617;
      border-radius: 10px;
      padding: 8px 10px;
      border: 1px solid #1f2937;
    }

    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    .metric-value {
      margin-top: 2px;
      font-size: 0.98rem;
      font-weight: 600;
    }

    .metric-sub {
      font-size: 0.72rem;
      color: var(--text-muted);
      margin-top: 2px;
    }

    .footer-note {
      font-size: 0.78rem;
      color: var(--text-muted);
      margin-top: 8px;
    }

    .footer-note code {
      background: #020617;
      padding: 2px 6px;
      border-radius: 6px;
      border: 1px solid #1f2937;
      font-size: 0.72rem;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="header">
      <div class="title-block">
        <div class="title">
          <div class="badge-triangle">Δt</div>
          <span>Triangle Time Console</span>
          <span class="pill">G / A / D · rules not vibes</span>
        </div>
        <div class="subtitle">
          Triangle in, time out. No circle. No vibes. Just quant.
        </div>
      </div>
    </div>

    <!-- LEFT: Theory / explanation -->
    <div class="left-panel">
      <div class="section-title">Triangle model · mental model</div>

      <div class="card">
        <div class="mini-heading">Vertices</div>
        <div class="body-text">
          Every task sits inside a triangle with three vertices:
        </div>
        <div class="chip-row">
          <div class="chip"><strong>G – Gov</strong> · approvals, policy, bureaucracy</div>
          <div class="chip"><strong>A – Azure</strong> · infra, identity, integration plumbing</div>
          <div class="chip"><strong>D – Data</strong> · ETL, ML, analytics, dashboards</div>
        </div>
        <div class="body-text" style="margin-top:6px;">
          For each task we track time in hours:
        </div>
        <div class="equation">
          <span class="keyword">T</span><span class="sym"> = </span>
          <span class="keyword">T_G</span> +
          <span class="keyword">T_A</span> +
          <span class="keyword">T_D</span>
        </div>
        <div class="body-text">
          And we convert them into triangle coordinates:
        </div>
        <div class="equation">
          <span class="keyword">p_G</span> = T_G / T,
          <span class="keyword">p_A</span> = T_A / T,
          <span class="keyword">p_D</span> = T_D / T,
          <span class="sym">p_G + p_A + p_D = 1</span>
        </div>
      </div>

      <div class="card">
        <div class="mini-heading">Predictive formula</div>
        <div class="body-text">
          From historical tasks we learn base times for pure vertices:
        </div>
        <div class="equation">
          <span class="keyword">T_G*</span>, <span class="keyword">T_A*</span>, <span class="keyword">T_D*</span>
          <span class="sym">→</span>
          typical time for pure-G / pure-A / pure-D work.
        </div>
        <div class="body-text">
          For a task at <code>(p_G, p_A, p_D)</code>, predicted time:
        </div>
        <div class="equation">
          <span class="keyword">T_pred</span>
          <span class="sym"> = </span>
          p_G · T_G* + p_A · T_A* + p_D · T_D*
        </div>
        <div class="body-text">
          Optional: add multi-owner drag using Shannon entropy:
        </div>
        <div class="equation">
          <span class="keyword">H(p)</span>
          <span class="sym"> = </span>
          - Σ p_i log p_i
        </div>
        <div class="equation">
          <span class="keyword">T_pred</span>
          <span class="sym"> = </span>
          p_G · T_G* + p_A · T_A* + p_D · T_D* + η · H(p)
        </div>
      </div>

      <div class="card">
        <div class="mini-heading">Flow · end-to-end</div>
        <div class="body-text">
          1. Track real tasks → log <code>T_G, T_A, T_D</code><br/>
          2. Fit model params <code>(T_G*, T_A*, T_D*, η)</code><br/>
          3. For new tasks: estimate <code>(p_G, p_A, p_D)</code><br/>
          4. Use this console to get <strong>T_pred</strong> and log the task.
        </div>
      </div>
    </div>

    <!-- RIGHT: Interactive console -->
    <div class="right-panel">
      <div class="section-title">Triangle input · calculator</div>

      <div class="card">
        <div class="body-text" style="margin-bottom:6px;">
          Enter either <strong>raw hours</strong> or <strong>triangle proportions</strong> and let the model predict total time.
        </div>
        <div class="form-row">
          <div class="field">
            <label for="task_id">Task ID (optional)</label>
            <input id="task_id" type="text" placeholder="e.g. TASK-123" />
          </div>
          <div class="field">
            <label for="t_gov">T_G (hours – gov)</label>
            <input id="t_gov" type="number" step="0.01" placeholder="e.g. 2" />
          </div>
          <div class="field">
            <label for="t_azure">T_A (hours – Azure)</label>
            <input id="t_azure" type="number" step="0.01" placeholder="e.g. 3" />
          </div>
        </div>

        <div class="form-row" style="margin-top:8px;">
          <div class="field">
            <label for="t_ds">T_D (hours – data)</label>
            <input id="t_ds" type="number" step="0.01" placeholder="e.g. 1" />
          </div>
          <div class="field">
            <label for="p_gov">p_G (0–1)</label>
            <input id="p_gov" type="number" step="0.01" placeholder="optional" />
          </div>
          <div class="field">
            <label for="p_azure">p_A (0–1)</label>
            <input id="p_azure" type="number" step="0.01" placeholder="optional" />
          </div>
        </div>

        <div class="form-row" style="margin-top:8px;">
          <div class="field">
            <label for="p_ds">p_D (0–1)</label>
            <input id="p_ds" type="number" step="0.01" placeholder="optional" />
          </div>
          <div class="field">
            <label for="t_total">T_total (optional)</label>
            <input id="t_total" type="number" step="0.01" placeholder="auto if empty" />
          </div>
          <div></div>
        </div>

        <div class="button-row">
          <button class="btn-primary" onclick="handlePredict()">
            ▶ Predict time
          </button>
          <button class="btn-ghost" onclick="handleLogTask()">
            ⬩ Log task only
          </button>
        </div>

        <div class="status-bar" id="status_bar"></div>
      </div>

      <div class="card">
        <div class="mini-heading">Result snapshot</div>
        <div class="results-grid">
          <div class="metric">
            <div class="metric-label">Predicted total time</div>
            <div class="metric-value" id="metric_tpred">–</div>
            <div class="metric-sub">hours</div>
          </div>
          <div class="metric">
            <div class="metric-label">Triangle mix (p_G / p_A / p_D)</div>
            <div class="metric-value" id="metric_mix">–</div>
            <div class="metric-sub">normalized from inputs</div>
          </div>
        </div>

        <div class="mini-heading" style="margin-top:8px;">Model parameters</div>
        <div class="results-grid">
          <div class="metric">
            <div class="metric-label">Base times (G / A / D)</div>
            <div class="metric-value" id="metric_bases">–</div>
            <div class="metric-sub">T_G*, T_A*, T_D*</div>
          </div>
          <div class="metric">
            <div class="metric-label">Entropy weight η</div>
            <div class="metric-value" id="metric_eta">–</div>
            <div class="metric-sub">if using multi-owner drag</div>
          </div>
        </div>

        <div class="footer-note">
          Backed by your <code>model_params.json</code> and task log in <code>data/tasks_logged.csv</code>.
        </div>
      </div>
    </div>
  </div>

  <script>
    function readNumber(id) {
      const el = document.getElementById(id);
      if (!el) return null;
      const v = el.value.trim();
      if (v === "") return null;
      const num = Number(v);
      return isNaN(num) ? null : num;
    }

    function setStatus(msg, type = "info") {
      const bar = document.getElementById("status_bar");
      if (!bar) return;
      if (!msg) {
        bar.textContent = "";
        return;
      }
      const span = document.createElement("span");
      span.textContent = msg;
      span.className = type === "err" ? "err" : "ok";
      bar.innerHTML = "";
      bar.appendChild(span);
    }

    function updateMetricsFromPredict(resp, payload) {
      const tpredEl = document.getElementById("metric_tpred");
      const mixEl = document.getElementById("metric_mix");
      const basesEl = document.getElementById("metric_bases");
      const etaEl = document.getElementById("metric_eta");

      if (tpredEl) tpredEl.textContent = resp.T_pred?.toFixed(2) ?? "–";

      const p_g = payload.p_gov ?? null;
      const p_a = payload.p_azure ?? null;
      const p_d = payload.p_ds ?? null;

      let mixText = "–";
      if (p_g != null || p_a != null || p_d != null) {
        const pg = p_g != null ? p_g : 0;
        const pa = p_a != null ? p_a : 0;
        const pd = p_d != null ? p_d : 0;
        mixText = `${pg.toFixed(2)} / ${pa.toFixed(2)} / ${pd.toFixed(2)}`;
      }
      if (mixEl) mixEl.textContent = mixText;

      const params = resp.model_params || {};
      const tg = params.T_gov_star ?? params.T_G_star ?? null;
      const ta = params.T_azure_star ?? params.T_A_star ?? null;
      const td = params.T_ds_star ?? params.T_D_star ?? null;
      if (basesEl) {
        if (tg != null && ta != null && td != null) {
          basesEl.textContent = `${tg.toFixed(2)} / ${ta.toFixed(2)} / ${td.toFixed(2)}`;
        } else {
          basesEl.textContent = "–";
        }
      }
      if (etaEl) {
        const eta = params.eta ?? params.eta_entropy ?? null;
        etaEl.textContent = eta != null ? eta.toFixed(4) : "–";
      }
    }

    async function handlePredict() {
      setStatus("Calling /predict_time ...", "info");
      const taskId = document.getElementById("task_id").value.trim() || null;

      const payload = {
        task_id: taskId,
        T_gov: readNumber("t_gov") ?? 0,
        T_azure: readNumber("t_azure") ?? 0,
        T_ds: readNumber("t_ds") ?? 0,
        T_total: readNumber("t_total"),
        p_gov: readNumber("p_gov"),
        p_azure: readNumber("p_azure"),
        p_ds: readNumber("p_ds")
      };

      try {
        const res = await fetch("/predict_time", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const txt = await res.text();
          setStatus("Error from /predict_time: " + txt, "err");
          return;
        }
        const data = await res.json();
        updateMetricsFromPredict(data, payload);
        setStatus("Prediction OK for " + (data.task_id || "task") + ".", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Failed to call /predict_time – check server logs.", "err");
      }
    }

    async function handleLogTask() {
      setStatus("Logging task via /log_task ...", "info");
      const taskId = document.getElementById("task_id").value.trim() || null;

      const payload = {
        task_id: taskId,
        T_gov: readNumber("t_gov") ?? 0,
        T_azure: readNumber("t_azure") ?? 0,
        T_ds: readNumber("t_ds") ?? 0,
        T_total: readNumber("t_total"),
        p_gov: readNumber("p_gov"),
        p_azure: readNumber("p_azure"),
        p_ds: readNumber("p_ds")
      };

      try {
        const res = await fetch("/log_task", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        if (!res.ok) {
          const txt = await res.text();
          setStatus("Error from /log_task: " + txt, "err");
          return;
        }
        await res.json();
        setStatus("Task logged to backend (CSV / future DB).", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Failed to call /log_task – check server logs.", "err");
      }
    }
  </script>
</body>
</html>
    """


# --- Helpers -----------------------------------------------------------------


def load_model_params(path: Path = MODEL_PARAMS_PATH) -> ModelParams:
    """
    Load ModelParams from a JSON file.

    Expected keys: T_gov_star, T_azure_star, T_ds_star, eta, use_entropy.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model params file not found at {path}. "
            "Run `python -m app.cli fit data/samples/example_tasks.csv` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return ModelParams(**data)


def append_task_to_csv(task: Task, path: Path = TASK_LOG_CSV_PATH) -> None:
    """
    Append a single task to the task log CSV.

    This is intentionally simple: load existing, append, overwrite.
    Good enough for low-volume / demo. Replace with DB/Azure in prod.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    existing_tasks = []
    if path.exists():
        existing_tasks = load_tasks_from_csv(str(path))

    existing_tasks.append(update_task_proportions(task))
    save_tasks_to_csv(existing_tasks, str(path))


# --- Request / Response schemas ----------------------------------------------


class TaskPayload(BaseModel):
    """
    Input payload for both /predict_time and /log_task.

    You can supply:
    - Raw times: T_gov, T_azure, T_ds, (optional) T_total
    - Optionally p_gov, p_azure, p_ds if you already have proportions.

    If proportions are missing, they will be computed from times.
    """

    task_id: Optional[str] = None

    T_gov: float = 0.0
    T_azure: float = 0.0
    T_ds: float = 0.0
    T_total: Optional[float] = None

    p_gov: Optional[float] = None
    p_azure: Optional[float] = None
    p_ds: Optional[float] = None


class PredictResponse(BaseModel):
    task_id: Optional[str]
    T_pred: float
    model_params: dict


class LogTaskResponse(BaseModel):
    status: str
    task: dict


# --- Health + self-test ------------------------------------------------------


@app.get("/health")
def health() -> dict:
    """Basic health check."""
    return {"status": "ok"}


@app.get("/self-test")
def self_test() -> dict:
    """
    End-to-end smoke test:

    1. Load example tasks from data/samples/example_tasks.csv
    2. Load model params from model_params.json
    3. Predict time for the first task
    4. Append that task into the task log CSV
    """
    sample_csv = REPO_ROOT / "data" / "samples" / "example_tasks.csv"
    if not sample_csv.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Sample CSV not found at {sample_csv}",
        )

    tasks = load_tasks_from_csv(str(sample_csv))
    if not tasks:
        raise HTTPException(
            status_code=500,
            detail="No tasks found in example_tasks.csv",
        )

    task = tasks[0]

    try:
        params = load_model_params()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    T_pred = predict_time_for_task(task, params)
    append_task_to_csv(task)

    return {
        "ok": True,
        "sample_task": asdict(task),
        "T_pred": T_pred,
        "model_params": asdict(params),
        "task_log_csv": str(TASK_LOG_CSV_PATH),
    }


# --- Main endpoints ----------------------------------------------------------


@app.post("/predict_time", response_model=PredictResponse)
def predict_time(payload: TaskPayload) -> PredictResponse:
    """
    Predict the total time for a task.

    You can send either raw times T_G/T_A/T_D (and optional T_total),
    or pre-computed proportions p_G/p_A/p_D.
    """
    try:
        params = load_model_params()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    task = Task(
        task_id=payload.task_id,
        T_gov=payload.T_gov,
        T_azure=payload.T_azure,
        T_ds=payload.T_ds,
        T_total=payload.T_total,
        p_gov=payload.p_gov,
        p_azure=payload.p_azure,
        p_ds=payload.p_ds,
    )

    T_pred = predict_time_for_task(task, params)

    return PredictResponse(
        task_id=task.task_id,
        T_pred=T_pred,
        model_params=asdict(params),
    )


@app.post("/log_task", response_model=LogTaskResponse)
def log_task(payload: TaskPayload) -> LogTaskResponse:
    """
    Log a completed task with actual time into the CSV log.
    """
    task = Task(
        task_id=payload.task_id,
        T_gov=payload.T_gov,
        T_azure=payload.T_azure,
        T_ds=payload.T_ds,
        T_total=payload.T_total,
        p_gov=payload.p_gov,
        p_azure=payload.p_azure,
        p_ds=payload.p_ds,
    )

    task = update_task_proportions(task)
    append_task_to_csv(task)

    return LogTaskResponse(
        status="ok",
        task=asdict(task),
    )


# Convenience for local dev:
# uvicorn app.api:app --reload
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
