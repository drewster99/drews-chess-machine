#!/usr/bin/env python3
"""Regenerate v5-strength.html from the live probe data.

Run 1 (original v5 training, cumulative steps 1..268k) is preserved verbatim
from the prior HTML so we never re-transcribe it; run 2 (corpus-replay warm
start) is rebuilt from new_ckpts_run2.jsonl each time. Both plot on one
cumulative step axis (cum = 268506 + run2_step) with the reboot seam at 268.5k.
"""
import json, os, re

MON = os.path.dirname(os.path.abspath(__file__))
OFF = 268506
SRC = os.path.join(MON, "v5-strength.html")

html = open(SRC).read()

# --- run 1: keep every cumulative checkpoint <= 268k from the existing array ---
old = re.search(r"const D=\[(.*?)\];", html, re.S).group(1)
run1 = []
for e in re.findall(r"\[([^\]]*)\]", old):
    p = [float(x) for x in e.split(",")]
    if p[0] <= 268:
        run1.append(p)

# --- run 2: rebuild from the jsonl ---
run2 = []
for line in open(os.path.join(MON, "new_ckpts_run2.jsonl")):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    m = re.search(r"step(\d+)", d["model"])
    if not m:
        continue
    cumK = round((int(m.group(1)) + OFF) / 1000, 1)
    top1 = 100 * d["argmaxCorrect"] / d["n"]
    run2.append((cumK, round(d["pElo"], 1), round(top1, 1), round(d["nll"], 3)))
run2 = sorted(set(run2))

rows = [(r[0], r[1], r[2], r[3]) for r in run1] + run2
frag = ", ".join("[%g, %.1f, %.1f, %.3f]" % r for r in rows)

best = max(rows, key=lambda r: r[1])
lown = min(rows, key=lambda r: r[3])
last = rows[-1]
n = len(rows)

# Exact record values (unrounded) come from the records file, not the 3dp chart data.
rec = json.load(open(os.path.join(MON, "records.json")))
best = (rec["best_pelo_step"], rec["best_pelo"], best[2], best[3])
lown = (rec["low_nll_step"], lown[1], lown[2], rec["low_nll"])

# --- targeted swaps onto the working page ---
html = re.sub(r"const D=\[.*?\];", "const D=[%s];" % frag, html, flags=re.S)
html = html.replace("xmax:273", "xmax:568")
html = html.replace("min:900,max:1720", "min:850,max:1800")
html = html.replace("min:2.0,max:3.1", "min:1.8,max:3.5")
html = html.replace("min:18,max:54", "min:18,max:57")
# right nll ticks label already 2dp; left ticks 8 fine.

# subline
html = re.sub(
    r'(<p class="sub" id="runline">).*?(</p>)',
    r"\g<1>model 20260629-1-Uf4p · Lichess-2026-05 corpus · wide probe (n=4435) · %d checkpoints\g<2>" % n,
    html, flags=re.S)

# records banner into KPI area title? keep KPIs auto; just update notes + foot.
notes = (
 '<p><b>Reading it.</b> Points are single checkpoints on the fixed 4435-puzzle battery — '
 'noisy (±80–100 pElo swing). The <b>10-point EMA</b> (α=2/11) cuts through that. '
 '<b>nll</b> (right axis, lower is better) is the policy neg-log-likelihood of the correct '
 'move; it runs opposite to pElo.</p>'
 '<p><b>Run 1 (to 268.5k).</b> The original v5 training: volatile early, a plateau 41k–67k, '
 'a rough trough 76k–104k (deep single-checkpoint craters, incl. <code>step93k=1230</code> and '
 'a <code>gNorm</code> gradient scar at 69k=943), recovery to a run-1 high at <b>129k (1704)</b>, '
 'then a logit-inflation slump 140k–181k, and a choppy epoch-2 band with self-healing craters. '
 'Stopped cleanly at <b>step 268,506</b> for a macOS update.</p>'
 f'<p><b>Run 2 — corpus-replay warm start (right of the seam).</b> Resumed from that exact corpus '
 'position; the trainer step counter restarts at 1, so checkpoints plot at their <b>cumulative</b> '
 'step. Run 2 opened ~1650 and has ground steadily upward, its band ratcheting from the ~1660s '
 '(280k) into the <b>1740–1770</b> zone by the mid-530k–550k cluster. It set fresh all-time highs '
 f'well above the run-1 peak: <b>pElo {best[1]:.0f} at {best[0]:g}k</b> and the lowest policy nll of '
 f'the whole run, <b>{lown[3]:.4f} at {lown[0]:g}k</b> — records only ~2k steps apart, the ceiling '
 'genuinely rising rather than just oscillating. The tail (~555k–567k) is a wider-amplitude trough '
 '(down to ~1642) sitting below that peak cluster but not deteriorating — nll stays in the healthy '
 '1.86–2.02 band and top-1 ≥50%. The one monotone driver is the policy logit-abs-max, which has '
 'climbed past 175 with no strength cost so far — the metric being watched.</p>'
)
html = re.sub(r'(<div class="notes">).*?(</div>\s*<p class="foot")',
              r"\g<1>\n    " + notes + r"\n  \g<2>", html, flags=re.S)

# foot template
html = re.sub(
    r'D\.length\+"[^;]*?α=0\.182"',
    'D.length+" checkpoints · steps 1k–"+last[0]+"k (cumulative) · wide battery n=4435 '
    '· reboot seam 268.5k · records pElo %.0f@%gk, nll %.4f@%gk · 10-pt EMA α=0.182"'
    % (best[1], best[0], lown[3], lown[0]),
    html)

open(SRC, "w").write(html)
print("wrote %s · %d checkpoints · best %.1f@%gk · low-nll %.4f@%gk · last %.1f@%gk"
      % (SRC, n, best[1], best[0], lown[3], lown[0], last[1], last[0]))
