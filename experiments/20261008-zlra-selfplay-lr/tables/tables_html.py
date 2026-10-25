import subprocess, sys, html, datetime, os
S = os.path.dirname(os.path.abspath(__file__))
runs = [
    ("replay", "R-replay", "Corpus replay · model 20261008-5-mIMw",
     "8.27M parameters, 3 blocks 9×9, SiLU. LR cycle 0.001–1.0, period 10k steps. Segment 0 ran 0–60k on build 2440; segment 1 (exact resume on build 2491, 12:20 on 10-10, row 60,001) was stopped at ~67,237 by the 21:15 GPU reset; segment 2 (exact resume from 67,000 at 21:17) was stopped by SIGINT at 86,398 on 10-11 13:05 to free its slot for the R-elite branch. The Lichess bot keeps its last R-replay model. Files: Models/20261008-zlra-replay-step*.safetensors (last 20261008-zlra-replay-step86398.safetensors, model ID 20261011-2-abM2); lineage run 0EFE2DA3-070A-4EDF-954A-DB8EB92309B9; logs dcm_log_20261008-094622, 20261010-122003, 20261010-211714."),
    ("silu", "B-siluall", "Corpus replay · SiLU everywhere",
     "5.13M parameters, 3 blocks 7×7. Same LR cycle. Segment 0 ran 0–40k; segment 1 (exact resume, row 40,001) stopped at ~45,975 on a NaN from a GPU reset; segment 1b (exact resume from 45,000, row 45,001) ran to 60k; segment 2 (exact resume, 08:21 on 10-10) stopped by SIGINT at 64,071 to change builds; segment 3 (exact resume on build 2491, 11:13, row 64,072) was stopped at ~74,377 by the 21:15 GPU reset; segment 4 (exact from 74,000 at 21:17) was stopped by the owner at 76,174 and resumes once R-replay stops. Files: Models/20261008-lrBsiluall-relcap-replay-step*.safetensors (last …-step76174, model ID 20261011-1-dVlQ); lineage run 66006003-5799-4FA0-B7AE-F3C19E6FD3A5."),
    ("elite", "R-elite", "Corpus replay · branch of R-replay on the elite corpus",
     "R-replay's model from its trainer step 86,398, trained on the Lichess Elite corpus (op24Gp: 14.6M games, mean rating 2568) at a fixed LR of 0.002, momentum 0.90, no cycle; one pass. Steps count from the branch (0). Started 13:05 on 10-11; tests whether R-replay's 1640-mean May corpus caps it. Stopped by SIGINT at 21,006 (21:34) for the R-elitecyc branch. Files: Models/20261011-zlra-elite-latest-step*.safetensors (last …-step21006, model ID 20261011-4-btGW); lineage run ACC058E9-15A5-49EA-8F60-9D1F254EAA4C; log dcm_log_20261011-130546."),
    ("elitecyc", "R-elitecyc", "Corpus replay · branch of R-elite with a gentle LR cycle",
     "R-elite's model from its trainer step 21,006, on the same elite corpus from where R-elite stopped (game 2,162,743). LR cycles 0.002 → 0.02 → 0.002 (geometric cosine, period 10k, no decay) after a 1,000-step warmup; momentum fixed 0.90; zero starting velocity. LR peaks at steps 6k, 16k, 26k, …; troughs at 1k, 11k, 21k, …. Steps count from the branch (0). Build 2503, started 21:39 on 10-11. Files: Models/20261011-zlra-elitecyc-latest-step*.safetensors and the rolling …-latest.safetensors (model ID 20261012-2-dMNh); lineage run E709F4EB-1D95-4166-90FA-F0D31002BB89; logs dcm_log_20261011-213940 (segment 0, SIGINT at 45,467, 15:25 on 10-12) and dcm_log_20261012-153015 (segment 1, exact resume on build 2523). Later segments: exact resumes on build 2523 through segment 12. Stopped by SIGINT at 384,964 (14:23 on 10-25, owner), plateaued at a best of 1779–1786 since ~260k in its third pass (epoch 2); R-elitecyc-anneal branches from that save."),
    ("anneal", "R-elitecyc-anneal", "Corpus replay · branch of R-elitecyc with the LR annealed",
     "R-elitecyc's abort save at trainer step 384,964 (model 20261025-1-qyUf), on the elite corpus from where R-elitecyc stopped (epoch 2, game 10,696,971). LR falls geometrically 0.002 → 0.0002 over 20,000 steps with no cycle (lr_cycle_keyframes: step 0 peak = trough = 0.002, step 20,000 peak = trough = 0.0002), then holds 0.0002; momentum fixed 0.90; 1,000-step warmup from zero velocity. Steps count from the branch (0); stops at 30,000. Tests why R-elitecyc plateaued: well above its best (1786) means LR noise was the limit, about the same means model capacity. Build 2620, started 14:24 on 10-25, running alone. Parameters parameters-zlra-elitecyc-anneal.json; launcher launch/elitecyc_anneal_branch_launch.sh. Files: Models/20261025-zlra-elitecyc-anneal-latest-step*.safetensors and the rolling …-latest.safetensors; lineage run 23DF16CB-51D2-4EB6-9FEE-3F22D1BA87D1; log dcm_log_20261025-142424."),
    ("elitefresh", "R-elitefresh", "Corpus replay · from scratch on the elite corpus with R-replay's parameters",
     "R-replay's own untrained start file (20261008-zlra-ab-fresh), parameters (parameters-zlra.json: inverted cycle 0.001–1, period 10k, decay over 1M steps, momentum following) and seed (20261008), fed the elite corpus (op24Gp) instead of w3aA5b; one epoch. Owner request 10-12. Build 2523, started 15:31 on 10-12, beside R-elitecyc. Files: Models/20261012-zlra-elitefresh-latest-step*.safetensors and the rolling …-latest.safetensors; lineage run F79C7568-4DDB-443E-BFB4-8131DAB27F4A; log dcm_log_20261012-153151. Later segments: exact resumes on build 2523 through segment 10. Stopped by SIGINT at 245,158 (14:23 on 10-25, owner)."),
    ("fixed", "R-fixedlr", "GUI self-play · model 20261008-4-Ew5t",
     "Fixed LR 0.01, momentum 0.90. Since 08:10 on 10-09 (step ~30.1k): draw keep 0.70, max plies 400, max draws per batch 75%. Steps are trainer steps; across 30 promotions the step count went back only three times, all early and by at most 24 steps. Since 11:34 on 10-10 (step ~61.9k): build 2491, resumed in the GUI from a SIGUSR2 save — not exact only in the replay buffer, which refilled from new games in about 13 minutes. Training halted on the 21:15 GPU reset (its own arena hang) at 70,283; since 22:44, build 2501 resumed from its 16:03 save (trainer step 66,602), so steps after that are redone. Files: GUI session folders Sessions/*-20261008-7-peGM-*.dcmsession (latest 20261012-144454-20261008-7-peGM-manual); lineage run A2599216-F200-4555-84AD-0CC1BF546011."),
]
def md_to_rows(md):
    rows = [l for l in md.splitlines() if l.startswith("|")]
    head = [c.strip() for c in rows[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows[2:]]
    return head, body
def num(x):
    try: return float(x.replace(",", ""))
    except ValueError: return None
sections, cards = [], []
for key, name, kind, desc in runs:
    md = subprocess.run([sys.executable, os.path.join(S, "step_table.py"), key], capture_output=True, text=True, check=True).stdout
    head, body = md_to_rows(md)
    pi = 2  # wide pElo column
    rated = [(num(r[pi]), r) for r in body if num(r[pi]) is not None]
    if not body:
        continue
    best = max(rated, key=lambda t: t[0]) if rated else (None, [""] * len(head))
    last = body[-1]; lastr = rated[-1] if rated else (None, ["—"] * len(head))
    th = "".join(f"<th>{html.escape(h)}</th>" for h in head)
    trs = []
    for r in body:
        step = num(r[0])
        cls = ' class="resume"' if step and step > 1 and step % 1000 else ""
        tds = []
        for i, c in enumerate(r):
            k = "t" if i == 1 else "n"
            if i == pi and rated and r is best[1]: k += " best"
            tds.append(f'<td class="{k}">{html.escape(c)}</td>')
        trs.append(f"<tr{cls}>{''.join(tds)}</tr>")
    cards.append(f'''<div class="card"><div class="cname">{name}</div><div class="ckind">{html.escape(kind)}</div>
<dl><dt>Step</dt><dd>{last[0]}</dd><dt>Latest pElo</dt><dd>{lastr[1][pi]} <span class="at">@ {lastr[1][0]}</span></dd>
<dt>Best pElo</dt><dd>{best[1][pi]} <span class="at">@ {best[1][0]}</span></dd><dt>vLoss</dt><dd>{last[5]}</dd></dl></div>''')
    sections.append(f'<section id="{key}"><h2>{name} <span class="kind">{html.escape(kind)}</span></h2><p class="desc">{html.escape(desc)}</p><div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(trs)}</tbody></table></div></section>')
# LR-cycle peaks: every 10k peak for the three live runs, ±1,000 steps.
pmd = subprocess.run([sys.executable, os.path.join(S, "peak_table.py")], capture_output=True, text=True, check=True).stdout
phead, pbody = md_to_rows(pmd)
pth = "".join(f"<th>{html.escape(h)}</th>" for h in phead)
ptrs = []
for r in pbody:
    hot = r[5] not in ("", "0")
    tds = "".join(f'<td class="{"t" if i == 1 else "n"}{" best" if hot and i in (3, 5, 6) else ""}">{html.escape(c)}</td>' for i, c in enumerate(r))
    ptrs.append(f"<tr>{tds}</tr>")
peaks = (f'<section id="peaks"><h2>LR-cycle peaks <span class="kind">all three runs, ±1,000 steps around each 10k peak</span></h2>'
         f'<p class="desc">Peaks fall at 1k, 11k, 21k, … (10,000-step cycle, starting at its peak after a 1,000-step warmup). B-siluall decays its peak over 10,000,000 steps, R-replay over 1,000,000; R-fixedlr has no cycle (LR 0.01). Median gNorm is the step lines\' rolling gradient norm in the window; clipped steps are <code>[GRAD-CLIP] applied=true</code>; alarms count raise / escalate / worsen of the two BN-variance rules. Highlighted: windows with clipped steps. An empty row: not reached yet.</p>'
         f'<div class="scroll"><table><thead><tr>{pth}</tr></thead><tbody>{"".join(ptrs)}</tbody></table></div></section>')
# LR schedules: one row per run, from each run's parameters file (hand-kept; verified 2026-10-12).
sched_head = ["run", "LR shape", "LR low", "LR peak", "period", "first peak / low (run steps)", "peak decay", "momentum", "√batch scaling", "warmup"]
sched_rows = [
    ["R-replay", "cosine cycle, geometric, inverted (starts at peak)", "0.001", "1.0", "10,000", "peaks 1k, 11k, 21k …; lows 6k, 16k, 26k …", "peak → 0.001, low → 1e-6 over 1,000,000 steps (logged peak 0.58 at 81k)", "follows LR: 0.85 at peak, 0.95 at low", "on (×1 at batch 4096)", "1,000"],
    ["B-siluall", "same as R-replay", "0.001", "1.0", "10,000", "peaks 1k, 11k, 21k …; lows 6k, 16k, 26k …", "peak → 0.0001, low → 1e-5 over 10,000,000 steps", "follows LR: 0.85 at peak, 0.95 at low", "on (×1 at batch 4096)", "1,000"],
    ["R-elite", "fixed", "0.002", "0.002", "—", "—", "none", "fixed 0.90", "off", "1,000"],
    ["R-elitecyc", "cosine cycle, geometric, not inverted (starts at low)", "0.002", "0.02", "10,000", "lows 1k, 11k, 21k, 31k …; peaks 6k, 16k, 26k, 36k …", "none", "fixed 0.90", "off", "1,000 (0 → 0.002)"],
    ["R-elitecyc-anneal", "keyframes, no cycle (peak = trough at each keyframe): geometric fall, then held", "0.0002", "0.002", "—", "0.002 at 0 (after warmup at 1k) → 0.0002 at 20k, held to 30k; never cycles back", "built into the fall", "fixed 0.90", "off", "1,000 (0 → 0.002)"],
    ["R-elitefresh", "same as R-replay", "0.001", "1.0", "10,000", "peaks 1k, 11k, 21k …; lows 6k, 16k, 26k …", "peak → 0.001, low → 1e-6 over 1,000,000 steps", "follows LR: 0.85 at peak, 0.95 at low", "on (×1 at batch 4096)", "1,000"],
    ["R-fixedlr", "fixed", "0.01", "0.01", "—", "—", "none", "fixed 0.90", "off", "1,000"],
]
sth = "".join(f"<th>{html.escape(h)}</th>" for h in sched_head)
strs = "".join("<tr>" + "".join(f'<td class="{"t" if i in (0, 1, 5, 6, 7, 8) else "n"}">{html.escape(c)}</td>' for i, c in enumerate(r)) + "</tr>" for r in sched_rows)
schedules = (f'<section id="schedules"><h2>LR schedules <span class="kind">each run\'s learning-rate and momentum settings</span></h2>'
             f'<p class="desc">Values are the LR applied at batch 4096, as the step lines log it. A cycle\'s position depends only on the trainer step: (step − warmup) mod period. Geometric interpolation spends equal time per factor of the LR, so R-elitecyc is at 0.0063 a quarter of the way up, not 0.011.</p>'
             f'<div class="scroll"><table><thead><tr>{sth}</tr></thead><tbody>{strs}</tbody></table></div></section>')
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
page = f"""<title>DCM Training Step Tables</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;600&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
/* Layout: summary strip of three run cards, then one scrolling table per run. */
:root {{ --bg:#f7f8f6; --fg:#1b1f1d; --muted:#626a66; --line:#dde2df; --head:#eaefec; --stripe:#f0f3f1; --accent:#2f6f5e; --mark:#e3f0ea; --resume:#fbf3e2;
  --serif:"Apple Garamond","EB Garamond",Garamond,Georgia,serif; --mono:"IBM Plex Mono",ui-monospace,"SF Mono",Menlo,monospace; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#131715; --fg:#e6ebe8; --muted:#97a19c; --line:#2a312e; --head:#1d2321; --stripe:#181d1b; --accent:#7cc4ad; --mark:#1f3a31; --resume:#33291a; color-scheme:dark; }} }}
:root[data-theme="dark"] {{ --bg:#131715; --fg:#e6ebe8; --muted:#97a19c; --line:#2a312e; --head:#1d2321; --stripe:#181d1b; --accent:#7cc4ad; --mark:#1f3a31; --resume:#33291a; color-scheme:dark; }}
body {{ background:var(--bg); color:var(--fg); font-family:var(--serif); font-size:16px; padding-inline:16px; padding-block:24px 40px; }}
main {{ max-width:1500px; margin:0 auto; display:grid; gap:28px; }}
h1 {{ font-weight:600; font-size:30px; margin:0; text-wrap:balance; }}
h2 {{ font-weight:600; font-size:22px; margin:0 0 4px; }}
.kind {{ font-size:15px; font-weight:400; color:var(--muted); margin-left:8px; }}
.desc, .stamp {{ color:var(--muted); margin:0 0 10px; max-width:75ch; line-height:1.4; }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:12px; }}
.card {{ border:1px solid var(--line); border-radius:6px; padding:12px 14px; background:var(--head); min-width:0; }}
.cname {{ font-weight:600; font-size:19px; }} .ckind {{ color:var(--muted); font-size:14px; margin-bottom:8px; }}
dl {{ display:grid; grid-template-columns:auto 1fr; gap:3px 14px; margin:0; }}
dt {{ color:var(--muted); font-size:14px; }} dd {{ margin:0; font-family:var(--mono); font-size:13px; font-variant-numeric:tabular-nums; text-align:right; }}
.at {{ color:var(--muted); }}
.scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:6px; }}
table {{ border-collapse:collapse; font-size:12.5px; font-family:var(--mono); font-variant-numeric:tabular-nums; width:max-content; }}
th {{ background:var(--head); font-family:var(--serif); font-size:14px; font-weight:600; padding:6px 10px; text-align:right; border-bottom:1px solid var(--line); white-space:nowrap; }}
td {{ padding:3px 10px; border-bottom:1px solid var(--line); white-space:nowrap; }}
td.n {{ text-align:right; }} td.t {{ text-align:left; color:var(--muted); }}
tbody tr:nth-child(even) {{ background:var(--stripe); }}
tr.resume {{ background:var(--resume) !important; }}
td.best {{ background:var(--mark); color:var(--accent); font-weight:600; }}
.notes {{ color:var(--muted); font-size:14px; line-height:1.45; max-width:90ch; margin:0; padding-left:18px; }}
</style>
<main>
<header><h1>DCM training step tables</h1><p class="stamp">The three runs active on 2026-10-09 and 10-10, one row per 1,000 trainer steps. Generated {now} from the session logs and the checkpoint files' test-set results.</p></header>
<div class="cards">{''.join(cards)}</div>
<ul class="notes">
<li><b>wide pElo / NLL</b>: the 4,435-puzzle Lichess set. Corpus replay rows read the step's checkpoint file; R-fixedlr averages the live GUI ticks within ±500 steps. The highlighted cell is each run's best.</li>
<li><b>rv max/med</b>: the largest batch-norm channel running variance ÷ its layer's median, worst layer, from the nearest live layer-health read.</li>
<li><b>buffer avg plies/game</b>: plies ÷ games fed since the previous row (corpus replay); the new-game average from the composition line (self-play).</li>
<li>Shaded rows mark the first step of a resumed segment; its feed counters restart, so that row has no plies/game.</li>
<li>GPU resets on 2026-10-09 at 19:36:49, 21:37:09 and 22:59:04 (each early in an R-fixedlr arena) corrupted one step in each running process: B-siluall's NaN at ~45,975 (resumed), clipped 10M-norm gradients elsewhere.</li>
<li>All three runs moved from build 2440 to 2491 on 10-10 (B-siluall 11:13, R-fixedlr 11:34, R-replay 12:20): every GPU submission now names its network, stage and batch size, and a flight recorder logs what was on the GPU around any fault (<code>[GPU-INFLIGHT]</code>). A fourth GPU hang at 21:15:26 on 10-10 came again from R-fixedlr's process 68 s into an arena; every run's safeguards stopped training with no save after it.</li>
<li>Corpus-replay pElo swings with the LR cycle: LR peaks (≈ 1.0) at steps 1k, 11k, 21k, 31k, 41k, … and bottoms out (≈ 0.001) at 6k, 16k, 26k, 36k, 46k, …; ratings peak about 1–2k steps after the bottom.</li>
</ul>
{schedules}
{peaks}
{''.join(sections)}
</main>"""
out = os.path.join(S, "training_tables.html")
open(out, "w").write(page)
print(out, len(page))
