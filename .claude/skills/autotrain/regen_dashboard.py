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
    # Expected "YYYYMMDD-HHMMSS" with optional "-seed" / "-replicate" /
    # "-codechange" suffix.
    base = folder_name
    for suffix in ("-seed", "-replicate", "-codechange"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    try:
        return datetime.strptime(base, "%Y%m%d-%H%M%S")
    except ValueError:
        return None


def folder_mode(folder_name: str):
    if folder_name.endswith("-seed"):
        return "seed"
    if folder_name.endswith("-replicate"):
        return "replicate"
    if folder_name.endswith("-codechange"):
        return "codechange"
    return "normal"


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
    # Stub failure commentaries written by the skill when training crashed,
    # timed out, or the proposer emitted an out-of-bounds proposal.
    failure_prefixes = (
        "training run failed",
        "proposal failed bounds check",
        "proposer returned invalid JSON",
    )
    if any(commentary.startswith(p) for p in failure_prefixes):
        return "FAILED"
    # New trichotomy field takes precedence.
    c = analysis.get("classification")
    if c == "improved":
        return "ACCEPTED"
    if c == "neutral":
        return "NEUTRAL"
    if c == "regressed":
        return "REJECTED"
    # Back-compat: legacy is_result_improved boolean from pre-trichotomy runs.
    if "is_result_improved" in analysis:
        return "ACCEPTED" if analysis.get("is_result_improved") else "REJECTED"
    return "FAILED"


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
    # Training-time resolution. Authoritative source going forward
    # is `result.json["training_elapsed_seconds"]` — actual wall
    # clock the training loop ran for, written by the app. For runs
    # that completed before that field existed, fall back to
    # `training_time.txt` (a pre-run budget file the skill used to
    # write) and then to `proposal.json["training_time_seconds"]`
    # (the same budget, restated). No fallback is ever applied to a
    # NEW run — if a run fires after this change and its result
    # lacks the field, something is wrong and the duration column
    # should stay empty so the gap is visible.
    training_time = None
    result = load_json(folder / "result.json")
    if isinstance(result, dict):
        raw = result.get("training_elapsed_seconds")
        if isinstance(raw, (int, float)):
            training_time = int(raw)
    # Fallbacks ONLY when result.json doesn't carry the new field
    # (i.e. the run pre-dates the field's introduction).
    if training_time is None:
        tt_file = folder / "training_time.txt"
        if tt_file.is_file():
            try:
                training_time = int(tt_file.read_text().strip())
            except (ValueError, OSError):
                training_time = None
    if training_time is None and isinstance(proposal, dict):
        raw = proposal.get("training_time_seconds")
        if isinstance(raw, (int, float)):
            training_time = int(raw)
    # Arenas + promotions for this iteration. Pulled from the same
    # arena_results array that feeds AGGREGATES, but exposed
    # per-row so the table can show "5/0" style counts (5 arenas
    # ran, 0 promoted) instead of only a global rolling total.
    arena_count = None
    arena_promotions = None
    if isinstance(result, dict):
        ar = result.get("arena_results")
        if isinstance(ar, list):
            arena_count = len(ar)
            arena_promotions = sum(1 for a in ar if isinstance(a, dict) and a.get("promoted"))
    return {
        "timestamp": folder.name,
        "start_time_iso": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": status,
        "mode": folder_mode(folder.name),
        "change_details": change_details,
        "changed_params": diff_params(prev_params, new_params),
        "parameters": new_params if isinstance(new_params, dict) else None,
        "analysis_commentary": commentary,
        "training_time_seconds": training_time,
        "arena_count": arena_count,
        "arena_promotions": arena_promotions,
        "folder": f"experiments/{folder.name}",
    }


def extract_arenas(folder: Path):
    """Pull the (usually tiny) arena_results list from a folder's result.json."""
    result = load_json(folder / "result.json")
    if not isinstance(result, dict):
        return []
    arenas = result.get("arena_results")
    return arenas if isinstance(arenas, list) else []


def collect():
    if not EXPERIMENTS_DIR.is_dir():
        return [], None
    rows = []
    folder_arenas = []  # parallel to rows, for aggregate computation
    for folder in sorted(EXPERIMENTS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        row = build_row(folder)
        if row is not None:
            rows.append(row)
            folder_arenas.append(extract_arenas(folder))
    aggregates = compute_aggregates(rows, folder_arenas)
    return rows, aggregates


def compute_aggregates(rows, folder_arenas):
    if not rows:
        return {
            "total_iterations": 0,
            "counts": {},
            "accept_rate": None,
            "failure_streak": 0,
            "trailing_replicates": 0,
            "iterations_since_codechange": 0,
            "code_iteration_due": False,
            "code_iteration_interval": 40,
            "arena_count": 0,
            "promotions": 0,
            "best_arena_score": None,
            "best_arena_folder": None,
        }
    counts = {"SEED": 0, "ACCEPTED": 0, "NEUTRAL": 0, "REJECTED": 0, "FAILED": 0, "IN_PROGRESS": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    # Consecutive-failure streak — walk from newest backward.
    #   REJECTED or FAILED → add 1 (it's a regression or crash).
    #   NEUTRAL             → transparent: doesn't add, doesn't reset. The
    #                         baseline held but didn't advance; streak pauses.
    #   IN_PROGRESS         → transparent (not yet a decision).
    #   ACCEPTED or SEED    → stop walking, streak = 0.
    streak = 0
    for r in reversed(rows):
        s = r["status"]
        if s in ("REJECTED", "FAILED"):
            streak += 1
        elif s in ("NEUTRAL", "IN_PROGRESS"):
            continue
        else:
            break

    # Trailing replicates — newest-first walk, stop at first non-replicate.
    # IN_PROGRESS still counts if it's a replicate folder (mode is set from
    # folder name, which doesn't depend on analysis).
    trailing_replicates = 0
    for r in reversed(rows):
        if r.get("mode") == "replicate":
            trailing_replicates += 1
        else:
            break

    # Code-iteration cadence: count "normal" iterations since the last
    # codechange folder (or since the SEED if there's never been one).
    # When this counter hits CODE_ITERATION_INTERVAL, the autotrain skill's
    # step 0.6 routes the next iteration into CODE-CHANGE mode instead of
    # normal/replicate.
    #
    # Why count only "normal" — replicates and codechange folders are not
    # parameter-tuning iterations, so they shouldn't push the cadence
    # forward. SEEDs do reset the count (a fresh start gets a fresh window
    # before we attempt code surgery).
    CODE_ITERATION_INTERVAL = 40
    iterations_since_codechange = 0
    for r in reversed(rows):
        mode = r.get("mode")
        if mode == "codechange" or r.get("status") == "SEED":
            break
        if mode == "normal":
            iterations_since_codechange += 1
        # replicate/seed/IN_PROGRESS don't add to the count, but also don't
        # break — they're transparent.
    code_iteration_due = (
        iterations_since_codechange >= CODE_ITERATION_INTERVAL
    )

    # Accept rate denominator includes NEUTRAL: neutrals are decisions, just
    # not promotions. A long streak of neutrals correctly shows 0% accept.
    decided = counts["ACCEPTED"] + counts["NEUTRAL"] + counts["REJECTED"] + counts["FAILED"]
    accept_rate = (counts["ACCEPTED"] / decided) if decided > 0 else None

    arena_count = 0
    promotions = 0
    best_arena_score = None
    best_arena_folder = None
    for r, arenas in zip(rows, folder_arenas):
        for a in arenas:
            arena_count += 1
            if a.get("promoted"):
                promotions += 1
            score = a.get("score")
            if isinstance(score, (int, float)):
                if best_arena_score is None or score > best_arena_score:
                    best_arena_score = score
                    best_arena_folder = r["folder"]

    total_iters = sum(v for k, v in counts.items() if k != "SEED")

    return {
        "total_iterations": total_iters,
        "counts": counts,
        "accept_rate": accept_rate,
        "failure_streak": streak,
        "trailing_replicates": trailing_replicates,
        "iterations_since_codechange": iterations_since_codechange,
        "code_iteration_due": code_iteration_due,
        "code_iteration_interval": CODE_ITERATION_INTERVAL,
        "arena_count": arena_count,
        "promotions": promotions,
        "best_arena_score": best_arena_score,
        "best_arena_folder": best_arena_folder,
    }


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
  tr.NEUTRAL     { background: #f4f1e4; }
  tr.REJECTED    { background: #fff5e6; }
  tr.FAILED      { background: #fde8e8; }
  tr.SEED        { background: #e6f0fb; }
  tr.IN_PROGRESS { background: #fff9c4; }
  tr.mode-replicate td:first-child { border-left: 4px solid #7c4dff; }
  .badge { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px;
           margin-left: 4px; font-weight: 600; letter-spacing: 0.02em; vertical-align: middle; }
  .badge-replicate { background: #7c4dff; color: #fff; }
  code, .mono { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  .diff { white-space: nowrap; line-height: 1.5; }
  .old { color: #888; text-decoration: line-through; }
  .new { color: #111; font-weight: 600; }
  .commentary { max-width: 520px; white-space: pre-wrap; }
  .details { max-width: 360px; white-space: pre-wrap; }
  /* Column sizing — start column wider so the local timestamp stays on
     a single line; param-deltas narrower since the diff is short. */
  th.col-start, td.col-start { width: 180px; min-width: 180px; white-space: nowrap; }
  th.col-deltas, td.col-deltas { width: 240px; max-width: 240px; }
  td.col-deltas .diff { white-space: normal; word-break: break-word; }
  tr.flash { animation: flash 1.6s ease-out; }
  @keyframes flash {
    from { background-color: #fff59d; }
    to { }
  }
  /* Aggregate banner */
  #aggregates {
    display: flex; flex-wrap: wrap; gap: 0; margin: 10px 0 12px 0;
    border: 1px solid #ccc; border-radius: 4px; overflow: hidden;
    background: #fafafa;
  }
  .agg-cell {
    flex: 1 1 140px; padding: 8px 14px; border-right: 1px solid #e3e3e3;
    display: flex; flex-direction: column; gap: 2px; min-width: 120px;
  }
  .agg-cell:last-child { border-right: none; }
  .agg-label { font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 0.04em; }
  .agg-value { font-size: 18px; font-weight: 600; font-family: "SF Mono", Menlo, Consolas, monospace; }
  .agg-cell.streak-hint .agg-value { color: #b27500; }
  .agg-cell.streak-warn { background: #ffebee; }
  .agg-cell.streak-warn .agg-value { color: #c62828; }
  .agg-sub { font-size: 11px; color: #888; }
  /* params-modal */
  .params-link { cursor: pointer; color: #1a5fb4; text-decoration: underline;
                 font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 12px; }
  .params-link:hover { color: #0d3d7a; }
  #params-modal-backdrop {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.45);
    z-index: 10; align-items: center; justify-content: center;
  }
  #params-modal-backdrop.open { display: flex; }
  #params-modal {
    background: #fff; border-radius: 6px; max-width: 720px; width: 90%;
    max-height: 85vh; display: flex; flex-direction: column;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  }
  #params-modal header {
    border-bottom: 1px solid #ddd; padding: 10px 14px; margin: 0;
    display: flex; align-items: center; justify-content: space-between;
  }
  #params-modal header h2 { margin: 0; font-size: 14px; font-weight: 600; }
  #params-modal header .close { cursor: pointer; font-size: 18px; color: #666;
                                 background: none; border: none; padding: 0 4px; }
  #params-modal header .close:hover { color: #000; }
  #params-modal pre {
    margin: 0; padding: 14px; overflow: auto; font-size: 12px;
    font-family: "SF Mono", Menlo, Consolas, monospace; white-space: pre;
    background: #fafafa; flex: 1 1 auto; border-radius: 0 0 6px 6px;
  }
  #params-modal footer {
    padding: 8px 14px; border-top: 1px solid #eee; font-size: 11px; color: #666;
    display: flex; justify-content: space-between; align-items: center;
  }
  #params-modal footer a { color: #1a5fb4; }
</style>
</head>
<body>
<header>
  <h1>Autotrain Experiments</h1>
  <span id="status">loading&hellip;</span>
</header>
<div id="aggregates"></div>
<div id="filters" style="margin: 8px 0 4px; font-size: 13px;">
  <label style="cursor: pointer; user-select: none;">
    <input type="checkbox" id="filter-promotions" />
    Show only iterations with promotions
  </label>
  <span id="filter-count" style="margin-left: 12px; color: #666;"></span>
</div>
<table id="experiments">
  <thead>
    <tr>
      <th class="col-start">Start (local)</th>
      <th>Status</th>
      <th>Dur.</th>
      <th title="Arenas run / promotions during this iteration">Arenas</th>
      <th>Change</th>
      <th class="col-deltas">Param deltas</th>
      <th>Params</th>
      <th>Analysis</th>
      <th>Folder</th>
    </tr>
  </thead>
  <tbody></tbody>
</table>
<div id="params-modal-backdrop" role="dialog" aria-modal="true">
  <div id="params-modal">
    <header>
      <h2 id="params-modal-title">parameters.json</h2>
      <button class="close" type="button" aria-label="Close">&times;</button>
    </header>
    <pre id="params-modal-body"></pre>
    <footer>
      <span id="params-modal-sub"></span>
      <a id="params-modal-file" href="#" target="_blank">open file</a>
    </footer>
  </div>
</div>
<script>
const POLL_INTERVAL_MS = 15000;
let currentScriptEl = null;

function escHTML(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

// Cache the local TZ abbreviation (e.g. "CDT") once per page load.
const LOCAL_TZ_ABBR = (() => {
  try {
    const parts = new Intl.DateTimeFormat(undefined, { timeZoneName: 'short' })
      .formatToParts(new Date());
    const tz = parts.find(p => p.type === 'timeZoneName');
    return tz ? tz.value : '';
  } catch (e) { return ''; }
})();

function fmtLocal(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const ss = String(d.getSeconds()).padStart(2, '0');
  const tz = LOCAL_TZ_ABBR ? ` ${LOCAL_TZ_ABBR}` : '';
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}${tz}`;
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
  tr.className = exp.status + (exp.mode === 'replicate' ? ' mode-replicate' : '');
  const statusCell = escHTML(exp.status || '') +
    (exp.mode === 'replicate' ? '<span class="badge badge-replicate">REPLICATE</span>' : '');
  const deltas = (exp.changed_params && exp.changed_params.length)
    ? exp.changed_params.map(d =>
        `<div class="diff"><code>${escHTML(d.key)}</code>: ` +
        `<span class="old">${fmtVal(d.old)}</span> &rarr; ` +
        `<span class="new">${fmtVal(d.new)}</span></div>`
      ).join('')
    : '<em>(none)</em>';
  const durCell = (typeof exp.training_time_seconds === 'number')
    ? `${Math.round(exp.training_time_seconds / 60)}m`
    : '—';
  // Arenas cell: "<count>/<promoted>" — e.g. "5/0" means 5 arenas
  // ran and 0 promoted. Falls back to em-dash for runs that
  // pre-date the field or didn't produce a result.json.
  let arenasCell = '—';
  if (typeof exp.arena_count === 'number') {
    const promo = (typeof exp.arena_promotions === 'number') ? exp.arena_promotions : 0;
    arenasCell = `<span class="mono" title="${exp.arena_count} arenas, ${promo} promoted">${exp.arena_count}/${promo}</span>`;
  }
  const paramsCell = exp.parameters
    ? `<span class="params-link" data-key="${escHTML(rowKey(exp))}">view</span>`
    : '<em>—</em>';
  tr.innerHTML = `
    <td class="mono col-start">${escHTML(fmtLocal(exp.start_time_iso))}</td>
    <td>${statusCell}</td>
    <td class="mono">${durCell}</td>
    <td>${arenasCell}</td>
    <td class="details">${escHTML(exp.change_details || '')}</td>
    <td class="col-deltas">${deltas}</td>
    <td>${paramsCell}</td>
    <td class="commentary">${escHTML(exp.analysis_commentary || '')}</td>
    <td><a href="${escHTML(exp.folder || '')}/"><code>${escHTML(exp.folder || '')}</code></a></td>
  `;
  return tr;
}

function paramsByKey(key) {
  const list = window.EXPERIMENTS || [];
  for (const exp of list) {
    if (rowKey(exp) === key) return exp;
  }
  return null;
}

function openParamsModal(key) {
  const exp = paramsByKey(key);
  if (!exp || !exp.parameters) return;
  document.getElementById('params-modal-title').textContent =
    `parameters.json — ${exp.timestamp}`;
  document.getElementById('params-modal-body').textContent =
    JSON.stringify(exp.parameters, null, 2);
  document.getElementById('params-modal-sub').textContent =
    `${Object.keys(exp.parameters).length} keys`;
  const fileLink = document.getElementById('params-modal-file');
  fileLink.href = `${exp.folder || ''}/parameters.json`;
  document.getElementById('params-modal-backdrop').classList.add('open');
}

function closeParamsModal() {
  document.getElementById('params-modal-backdrop').classList.remove('open');
}

document.addEventListener('click', (ev) => {
  const link = ev.target.closest('.params-link');
  if (link) {
    openParamsModal(link.dataset.key);
    return;
  }
  if (ev.target.closest('#params-modal .close')) {
    closeParamsModal();
    return;
  }
  // Click on backdrop (outside modal content) closes the modal.
  if (ev.target.id === 'params-modal-backdrop') closeParamsModal();
});

document.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape') closeParamsModal();
});

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

  // Re-apply the active row-level filter so newly added rows respect
  // the current "promotions only" toggle without the user having to
  // toggle it off-and-on after each poll.
  applyRowFilter();
}

// Toggle row visibility based on the "Show only iterations with
// promotions" checkbox. Rows where arena_promotions > 0 stay visible;
// the rest get display:none. Hidden rows are not removed from the
// DOM — the toggle is reversible and re-applied after every reconcile.
function applyRowFilter() {
  const checkbox = document.getElementById('filter-promotions');
  const countEl = document.getElementById('filter-count');
  if (!checkbox) return;
  const onlyPromotions = checkbox.checked;
  const tbody = document.querySelector('#experiments tbody');
  if (!tbody) return;
  let visible = 0;
  let total = 0;
  for (const tr of tbody.children) {
    total += 1;
    if (!onlyPromotions) {
      tr.style.display = '';
      visible += 1;
      continue;
    }
    // Pull arena_promotions out of the row's stashed payload — that
    // way we don't have to add yet another data-attribute and the
    // filter stays in sync with whatever shape the row was rendered
    // with.
    let promotions = 0;
    try {
      const payload = JSON.parse(tr.dataset.payload || '{}');
      promotions = (typeof payload.arena_promotions === 'number') ? payload.arena_promotions : 0;
    } catch (e) { promotions = 0; }
    if (promotions > 0) {
      tr.style.display = '';
      visible += 1;
    } else {
      tr.style.display = 'none';
    }
  }
  if (countEl) {
    countEl.textContent = onlyPromotions
      ? `${visible} of ${total} rows shown (promotions only)`
      : `${total} rows`;
  }
}

document.addEventListener('change', (ev) => {
  if (ev.target && ev.target.id === 'filter-promotions') applyRowFilter();
});

function renderAggregates(agg) {
  const el = document.getElementById('aggregates');
  if (!agg || !agg.total_iterations) {
    el.innerHTML = '';
    return;
  }
  const streakClass = agg.failure_streak >= 8 ? 'streak-warn'
                     : (agg.failure_streak >= 5 ? 'streak-hint' : '');
  const rate = (agg.accept_rate === null || agg.accept_rate === undefined)
               ? '—' : `${Math.round(agg.accept_rate * 100)}%`;
  const best = (agg.best_arena_score === null || agg.best_arena_score === undefined)
               ? '—' : agg.best_arena_score.toFixed(3);
  const counts = agg.counts || {};
  const c = (k) => counts[k] || 0;
  el.innerHTML = `
    <div class="agg-cell">
      <span class="agg-label">Iterations</span>
      <span class="agg-value">${agg.total_iterations}</span>
      <span class="agg-sub">A:${c('ACCEPTED')} N:${c('NEUTRAL')} R:${c('REJECTED')} F:${c('FAILED')}</span>
    </div>
    <div class="agg-cell">
      <span class="agg-label">Accept rate</span>
      <span class="agg-value">${rate}</span>
    </div>
    <div class="agg-cell ${streakClass}">
      <span class="agg-label">Fail streak</span>
      <span class="agg-value">${agg.failure_streak}</span>
      <span class="agg-sub">${agg.failure_streak >= 15 ? 'replicate mode' : (agg.failure_streak >= 10 ? 'replicate at 15' : (agg.failure_streak >= 5 ? 'watch' : ''))}</span>
    </div>
    ${(agg.trailing_replicates && agg.trailing_replicates > 0) ? `
    <div class="agg-cell ${agg.trailing_replicates >= 3 ? 'streak-warn' : (agg.trailing_replicates >= 2 ? 'streak-hint' : '')}">
      <span class="agg-label">Replicates</span>
      <span class="agg-value">${agg.trailing_replicates}/3</span>
      <span class="agg-sub">${agg.trailing_replicates >= 3 ? 'halted' : 'probing baseline'}</span>
    </div>` : ''}
    <div class="agg-cell ${agg.code_iteration_due ? 'streak-hint' : ''}">
      <span class="agg-label">Code iter</span>
      <span class="agg-value">${agg.iterations_since_codechange || 0}/${agg.code_iteration_interval || 40}</span>
      <span class="agg-sub">${agg.code_iteration_due ? 'due next' : ''}</span>
    </div>
    <div class="agg-cell">
      <span class="agg-label">Arenas</span>
      <span class="agg-value">${agg.arena_count}</span>
    </div>
    <div class="agg-cell">
      <span class="agg-label">Promotions</span>
      <span class="agg-value">${agg.promotions}</span>
    </div>
    <div class="agg-cell">
      <span class="agg-label">Best arena</span>
      <span class="agg-value">${best}</span>
      <span class="agg-sub">${agg.best_arena_folder ? `<code>${escHTML(agg.best_arena_folder)}</code>` : ''}</span>
    </div>
  `;
}

function loadData() {
  if (currentScriptEl) currentScriptEl.remove();
  const s = document.createElement('script');
  s.src = `experiment_results.js?t=${Date.now()}`;
  s.onload = () => {
    reconcile(window.EXPERIMENTS || []);
    renderAggregates(window.AGGREGATES);
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
    rows, aggregates = collect()
    payload = (
        "window.EXPERIMENTS = " + json.dumps(rows, indent=2) + ";\n"
        "window.AGGREGATES = " + json.dumps(aggregates, indent=2) + ";\n"
    )
    OUT_JS.write_text(payload)
    # Write the HTML shell on first run. On subsequent runs, only overwrite if
    # the file is missing a marker string that identifies the current template
    # revision — that way existing dashboards auto-upgrade to new features
    # (like the params modal) without the user having to delete the file, but
    # we don't churn mtime every regen when the template hasn't changed.
    # Marker bumps when the template gains a new feature so existing
    # dashboards auto-upgrade. Each marker should be a token that
    # appears only in the new template, never the prior one.
    # History: "col-start" (initial), "Arenas</th>" (per-row arena
    # count + promotions column).
    template_marker = "filter-promotions"
    if not OUT_HTML.is_file() or template_marker not in OUT_HTML.read_text():
        OUT_HTML.write_text(HTML_TEMPLATE)
    print(f"regen_dashboard: {len(rows)} runs -> {OUT_JS.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
