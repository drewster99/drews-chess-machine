#!/usr/bin/env python3
"""
regen_dashboard.py — rebuild experiment_results.js from experiments/.

Walks every test folder under $REPO_ROOT/experiments/, reads
proposal.json / analysis.json / parameters.json / previous_parameters.json,
emits a single `window.EXPERIMENTS = [ ... ];` file at the repo root that the
static HTML dashboard polls via a cache-busted <script> tag.

Also writes experiment_results.html on first run (left alone thereafter so
user edits survive).

Invoked from the autotrain skill at:
  - end of seed (step 3)
  - end of propose (step 5, to show an in-progress row)
  - end of run-failure stub (step 6)
  - end of accept AND end of reject (step 8)
"""

import json
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
OUT_JS = REPO_ROOT / "experiment_results.js"
OUT_HTML = REPO_ROOT / "experiment_results.html"


def load_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def parse_timestamp(folder_name: str):
    # Expected "YYYYMMDD-HHMMSS" with optional "-seed" suffix.
    base = folder_name.split("-seed", 1)[0]
    try:
        return datetime.strptime(base, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def diff_params(old, new):
    if not isinstance(old, dict) or not isinstance(new, dict):
        return []
    diffs = []
    for key in sorted(set(old) | set(new)):
        if old.get(key) != new.get(key):
            diffs.append({"key": key, "old": old.get(key), "new": new.get(key)})
    return diffs


def classify(folder: Path, proposal, analysis):
    if folder.name.endswith("-seed"):
        return "SEED"
    if analysis is None:
        # Folder exists but analysis hasn't been written yet — run is live.
        return "IN_PROGRESS"
    if not isinstance(analysis, dict):
        return "FAILED"
    commentary = str(analysis.get("analysis_commentary", ""))
    if commentary.startswith("training run failed"):
        return "FAILED"
    return "ACCEPTED" if analysis.get("is_result_improved") else "REJECTED"


def build_row(folder: Path):
    ts = parse_timestamp(folder.name)
    if ts is None:
        return None
    proposal = load_json(folder / "proposal.json") or {}
    analysis = load_json(folder / "analysis.json")
    prev_params = load_json(folder / "previous_parameters.json")
    new_params = load_json(folder / "parameters.json")
    status = classify(folder, proposal, analysis)
    if status == "SEED":
        change_details = "(seed baseline)"
    else:
        change_details = proposal.get("change_details", "") or ""
    commentary = ""
    if isinstance(analysis, dict):
        commentary = analysis.get("analysis_commentary", "") or ""
    return {
        "timestamp": folder.name,
        "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
        "change_details": change_details,
        "changed_params": diff_params(prev_params, new_params),
        "analysis_commentary": commentary,
        "folder": f"experiments/{folder.name}",
    }


def collect():
    if not EXPERIMENTS_DIR.is_dir():
        return []
    rows = []
    for folder in sorted(EXPERIMENTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        row = build_row(folder)
        if row is not None:
            rows.append(row)
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Autotrain Experiments</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
         margin: 16px; color: #222; }
  header { display: flex; align-items: baseline; justify-content: space-between;
           border-bottom: 1px solid #ccc; margin-bottom: 12px; padding-bottom: 6px; }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  #status { font-size: 12px; color: #666; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #e3e3e3; vertical-align: top;
           text-align: left; }
  th { background: #f4f4f4; position: sticky; top: 0; z-index: 1; }
  tr.ACCEPTED    { background: #eaf7ea; }
  tr.REJECTED    { background: #fff5e6; }
  tr.FAILED      { background: #fde8e8; }
  tr.SEED        { background: #e6f0fb; }
  tr.IN_PROGRESS { background: #fff9c4; }
  code, .mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  .diff { white-space: nowrap; line-height: 1.5; }
  .old { color: #888; text-decoration: line-through; }
  .new { color: #111; font-weight: 600; }
  .commentary { max-width: 520px; white-space: pre-wrap; }
  .details { max-width: 360px; white-space: pre-wrap; }
  tr.flash { animation: flash 1.6s ease-out; }
  @keyframes flash {
    from { background-color: #fff59d; }
    to { }
  }
</style>
</head>
<body>
<header>
  <h1>Autotrain Experiments</h1>
  <span id="status">loading&hellip;</span>
</header>
<table id="experiments">
  <thead>
    <tr>
      <th>Start (UTC)</th>
      <th>Status</th>
      <th>Change</th>
      <th>Param deltas</th>
      <th>Analysis</th>
      <th>Folder</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
<script>
const POLL_INTERVAL_MS = 15000;
let currentScriptEl = null;

function escHTML(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmtVal(v) {
  if (v === undefined) return '<em>absent</em>';
  if (v === null) return 'null';
  if (typeof v === 'number') return escHTML(v.toString());
  return escHTML(JSON.stringify(v));
}

function rowKey(exp) {
  return exp.timestamp;
}

function renderRow(exp) {
  const tr = document.createElement('tr');
  tr.dataset.key = rowKey(exp);
  tr.className = exp.status;
  const deltas = (exp.changed_params && exp.changed_params.length)
    ? exp.changed_params.map(d =>
        `<div class="diff"><code>${escHTML(d.key)}</code>: ` +
        `<span class="old">${fmtVal(d.old)}</span> &rarr; ` +
        `<span class="new">${fmtVal(d.new)}</span></div>`
      ).join('')
    : '<em>(none)</em>';
  tr.innerHTML = `
    <td class="mono">${escHTML(exp.start_time_iso || '')}</td>
    <td>${escHTML(exp.status || '')}</td>
    <td class="details">${escHTML(exp.change_details || '')}</td>
    <td>${deltas}</td>
    <td class="commentary">${escHTML(exp.analysis_commentary || '')}</td>
    <td><a href="${escHTML(exp.folder || '')}/"><code>${escHTML(exp.folder || '')}</code></a></td>
  `;
  return tr;
}

function isNearBottom() {
  const threshold = 80; // px slack so we still count as "at bottom" after small drift
  return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - threshold);
}

function reconcile(experiments) {
  // Snapshot scroll stickiness BEFORE mutating the DOM — if the user is
  // already watching the newest row at the bottom, we'll re-stick after
  // rendering. If they scrolled up to read an older row, we leave them alone.
  const wasAtBottom = isNearBottom();

  const tbody = document.querySelector('#experiments tbody');
  const byKey = new Map();
  Array.from(tbody.children).forEach(tr => byKey.set(tr.dataset.key, tr));

  let changed = false;
  experiments.forEach(exp => {
    const key = rowKey(exp);
    // Canonical payload for equality checks. Comparing innerHTML is unreliable
    // (browsers normalize attributes and entities inconsistently), which
    // caused every row to spuriously re-render and flash on every poll.
    const payload = JSON.stringify(exp);
    const existing = byKey.get(key);
    if (!existing) {
      const fresh = renderRow(exp);
      fresh.dataset.payload = payload;
      fresh.classList.add('flash');
      tbody.appendChild(fresh);
      changed = true;
    } else if (existing.dataset.payload !== payload) {
      const fresh = renderRow(exp);
      fresh.dataset.payload = payload;
      fresh.classList.add('flash');
      tbody.replaceChild(fresh, existing);
      changed = true;
    }
    // Unchanged rows are left in place — no flash, no DOM churn.
  });

  // Rows that disappeared from the backing data (e.g. folder deleted manually)
  // are left alone — we don't destructively remove anything the user might be
  // looking at.

  if (changed && wasAtBottom) {
    window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
  }
}

function loadData() {
  if (currentScriptEl) currentScriptEl.remove();
  const s = document.createElement('script');
  s.src = `experiment_results.js?t=${Date.now()}`;
  s.onload = () => {
    reconcile(window.EXPERIMENTS || []);
    const n = (window.EXPERIMENTS || []).length;
    document.getElementById('status').textContent =
      `${n} run${n === 1 ? '' : 's'} — updated ${new Date().toLocaleTimeString()}`;
  };
  s.onerror = () => {
    document.getElementById('status').textContent =
      `load error at ${new Date().toLocaleTimeString()}`;
  };
  document.head.appendChild(s);
  currentScriptEl = s;
}

loadData();
setInterval(loadData, POLL_INTERVAL_MS);
</script>
</body>
</html>
"""


def main():
    rows = collect()
    payload = "window.EXPERIMENTS = " + json.dumps(rows, indent=2) + ";\n"
    OUT_JS.write_text(payload)
    # Only write the HTML shell if it doesn't exist yet — overwriting on
    # every regen would reset the file's mtime even when the template hasn't
    # changed, and (historically) felt like it was causing spurious visual
    # changes on the open page. To pick up template tweaks in this script,
    # delete the file manually and run this generator again.
    if not OUT_HTML.is_file():
        OUT_HTML.write_text(HTML_TEMPLATE)
    print(f"regen_dashboard: {len(rows)} runs -> {OUT_JS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
