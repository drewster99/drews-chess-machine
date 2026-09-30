"""Build every data file for the final SE-style report.

Inputs (read-only):
  - documentation/dashboards/data/se_{sb,att,none,sb2,att2,none2,zb1,zb2}.csv  (probe series)
  - experiments/20260929-se-style-ab/logs/dcm_log_*.txt.gz                       (LR / momentum per step)
  - experiments/20260929-se-style-ab/models/*-fresh.safetensors                  (step-0 weights)
  - ~/Library/Application Support/DrewsChessMachine/Models/20260929-test_SE_*-replay-step*.safetensors
    (enumerated checkpoints; identified by __metadata__ model_id + training_step, never by filename)

Outputs: experiments/20260929-se-style-ab/data/*.csv (described in data/README.md).
Run from anywhere: python3 experiments/20260929-se-style-ab/final_data.py
"""
import csv, gzip, json, math, os, re, statistics as st, struct, glob

EXP = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(EXP, '..', '..'))
DASH = os.path.join(ROOT, 'documentation', 'dashboards', 'data')
OUT = os.path.join(EXP, 'data')
MODELS_DIR = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Models')

# run key, arm label, arm code, seed, beta init, fresh ModelID, log stem
RUNS = [
    ('se_sb',    'scale+bias',     'sb',   1, 'glorot', '20260929-12-JZOe', 'dcm_log_20260929-150727'),
    ('se_att',   'attenuate-only', 'att',  1, '',       '20260929-13-06yp', 'dcm_log_20260929-150735'),
    ('se_none',  'none',           'none', 1, '',       '20260929-18-D9is', 'dcm_log_20260929-150743'),
    ('se_sb2',   'scale+bias',     'sb',   2, 'glorot', '20260930-1-H1Oq',  'dcm_log_20260930-104101'),
    ('se_att2',  'attenuate-only', 'att',  2, '',       '20260930-2-Gf9P',  'dcm_log_20260930-104109'),
    ('se_none2', 'none',           'none', 2, '',       '20260930-3-V9zk',  'dcm_log_20260930-104117'),
    ('se_zb1',   'zero-β scale+bias', 'zb', 1, 'zero',  '20260930-7-crxN',  'dcm_log_20260930-150544'),
    ('se_zb2',   'zero-β scale+bias', 'zb', 2, 'zero',  '20260930-8-8qyR',  'dcm_log_20260930-150552'),
]
RUN = {r[0]: r for r in RUNS}
KEY = {(r[2], r[3]): r[0] for r in RUNS}          # (arm code, seed) -> run key
PROBE_COLS = ['pElo', 'nll', 'loss', 'pLoss', 'vLoss', 'legalMass', 'pIllM', 'gNorm',
              'games_fed', 'elapsed_train_sec']


def load_probes():
    data = {}
    for key, *_ in RUNS:
        rows = {}
        for r in csv.DictReader(open(os.path.join(DASH, key + '.csv'))):
            if r.get('pElo'):
                rows[int(float(r['cum_step']))] = r
        data[key] = rows
    return data


def f(data, key, step, col):
    r = data[key].get(step)
    if r is None or r.get(col) in (None, ''):
        return None
    return float(r[col])


def marks(data, key):
    return sorted(s for s in data[key] if s % 1000 == 0)


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return dict(n=len(xs), mean=st.mean(xs), sd=st.stdev(xs) if len(xs) > 1 else float('nan'),
                min=min(xs), max=max(xs))


def read_safetensors(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        header = json.loads(fh.read(n))
        body = fh.read()
    return header, body


def tensor(header, body, name):
    info = header[name]
    raw = body[info['data_offsets'][0]:info['data_offsets'][1]]
    if info['dtype'] == 'F32':
        vals = list(struct.unpack('<%df' % (len(raw) // 4), raw))
    elif info['dtype'] == 'BF16':
        vals = [struct.unpack('<f', b'\0\0' + raw[i:i + 2])[0] for i in range(0, len(raw), 2)]
    else:
        raise ValueError(f'unsupported dtype {info["dtype"]} for {name}')
    return info['shape'], vals


def fc2_halves(path):
    """Per block: (gamma half, beta half) of the flattened fc2 weight."""
    header, body = read_safetensors(path)
    out = []
    for b in range(3):
        (rows, cols), w = tensor(header, body, f'blocks.{b}.se_scalebias.fc2.weight')
        half = rows // 2
        out.append((w[:half * cols], w[half * cols:]))
    return out


def relation_to_init(x, x0):
    """(cosine to the init vector, Frobenius norm of the part orthogonal to it).

    Pure weight decay only rescales a vector, so it leaves cosine = 1 and the orthogonal
    part = 0; anything else is gradient-driven change of direction."""
    n0 = math.sqrt(sum(a * a for a in x0))
    nx = math.sqrt(sum(a * a for a in x))
    if n0 == 0.0:
        return None, nx
    proj = sum(a * b for a, b in zip(x, x0)) / n0
    cos = proj / nx if nx > 0 else None
    return cos, math.sqrt(max(nx * nx - proj * proj, 0.0))


def fc2_norms(path, init_halves):
    header, body = read_safetensors(path)
    md = header['__metadata__']
    out = []
    for b in range(3):
        wshape, w = tensor(header, body, f'blocks.{b}.se_scalebias.fc2.weight')
        bshape, bias = tensor(header, body, f'blocks.{b}.se_scalebias.fc2.bias')
        rows, cols = wshape
        assert rows == 256 and cols == 32 and bshape == [256], (path, wshape, bshape)
        half = rows // 2
        g = w[:half * cols]; be = w[half * cols:]
        row_norms = [math.sqrt(sum(x * x for x in be[i * cols:(i + 1) * cols])) for i in range(half)]
        g0, b0 = init_halves[b]
        g_cos, g_orth = relation_to_init(g, g0)
        b_cos, b_orth = relation_to_init(be, b0)
        out.append(dict(
            block=b,
            gamma_cos_to_init=g_cos, gamma_orth_to_init_fro=g_orth,
            beta_cos_to_init=b_cos, beta_orth_to_init_fro=b_orth,
            gamma_W_fro=math.sqrt(sum(x * x for x in g)),
            beta_W_fro=math.sqrt(sum(x * x for x in be)),
            beta_row_norm_mean=st.mean(row_norms),
            gamma_bias_mean=st.mean(bias[:half]),
            beta_bias_mean=st.mean(bias[half:]),
            beta_bias_absmean=st.mean(abs(x) for x in bias[half:]),
        ))
    return md, out


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_probes()

    # 1. long-format probe series ------------------------------------------------------------
    with open(os.path.join(OUT, 'probes_all_runs.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['run', 'arm', 'seed', 'beta_init', 'fresh_model_id', 'step', 'is_1k_mark']
                   + PROBE_COLS + ['checkpoint_file'])
        for key, arm, code, seed, binit, mid, _ in RUNS:
            for s in sorted(data[key]):
                r = data[key][s]
                w.writerow([key, arm, seed, binit, mid, s, int(s % 1000 == 0)]
                           + [r.get(c, '') for c in PROBE_COLS] + [r.get('frozen_file', '')])

    # 2. paired gaps within a seed -------------------------------------------------------------
    comps = [('none', 'sb'), ('none', 'att'), ('att', 'sb'), ('zb', 'sb'), ('zb', 'att'), ('zb', 'none')]
    with open(os.path.join(OUT, 'paired_gaps.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['seed', 'comparison', 'step', 'a_pElo', 'b_pElo', 'd_pElo', 'a_nll', 'b_nll', 'd_nll'])
        for seed in (1, 2):
            for a, b in comps:
                ka, kb = KEY[(a, seed)], KEY[(b, seed)]
                for s in sorted(set(marks(data, ka)) & set(marks(data, kb))):
                    pa, pb = f(data, ka, s, 'pElo'), f(data, kb, s, 'pElo')
                    na, nb = f(data, ka, s, 'nll'), f(data, kb, s, 'nll')
                    w.writerow([seed, f'{a}-{b}', s, pa, pb, round(pa - pb, 6), na, nb, round(na - nb, 6)])

    # 3. seed-to-seed gaps ---------------------------------------------------------------------
    with open(os.path.join(OUT, 'seed_gaps.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['arm', 'step', 'seed1_pElo', 'seed2_pElo', 'd_pElo', 'seed1_nll', 'seed2_nll', 'd_nll'])
        for code in ('sb', 'att', 'none', 'zb'):
            k1, k2 = KEY[(code, 1)], KEY[(code, 2)]
            for s in sorted(set(marks(data, k1)) & set(marks(data, k2))):
                p1, p2 = f(data, k1, s, 'pElo'), f(data, k2, s, 'pElo')
                n1, n2 = f(data, k1, s, 'nll'), f(data, k2, s, 'nll')
                w.writerow([code, s, p1, p2, round(p2 - p1, 6), n1, n2, round(n2 - n1, 6)])

    # 4. SE fc2 gamma/beta norms from every checkpoint of the four scale+bias-style runs --------
    sources = [
        ('se_sb',  os.path.join(EXP, 'models', '20260929-test_SE_scale+bias-fresh.safetensors'),
         os.path.join(MODELS_DIR, '20260929-test_SE_scale+bias-replay-step*.safetensors')),
        ('se_sb2', os.path.join(EXP, 'models', '20260929-test_SE_scale+bias-seed2-fresh.safetensors'),
         os.path.join(MODELS_DIR, '20260929-test_SE_scale+bias-seed2-replay-step*.safetensors')),
        ('se_zb1', os.path.join(EXP, 'models', '20260929-test_SE_zerobeta-seed1-fresh.safetensors'),
         os.path.join(MODELS_DIR, '20260929-test_SE_zerobeta-seed1-replay-step*.safetensors')),
        ('se_zb2', os.path.join(EXP, 'models', '20260929-test_SE_zerobeta-seed2-fresh.safetensors'),
         os.path.join(MODELS_DIR, '20260929-test_SE_zerobeta-seed2-replay-step*.safetensors')),
    ]
    norm_rows = []
    for key, fresh, pattern in sources:
        files = [fresh] + glob.glob(pattern)
        init_halves = fc2_halves(fresh)
        for path in files:
            md, blocks = fc2_norms(path, init_halves)
            step = int(md.get('training_step', '0') or 0)
            mid = md['model_id']
            parent = md.get('parent_model_id', '')
            # identify by metadata: the fresh net, or a checkpoint whose parent is the fresh net
            if not (mid == RUN[key][5] or parent == RUN[key][5]):
                raise SystemExit(f'{path}: model_id {mid} parent {parent} is not in lineage of {key}')
            for bl in blocks:
                norm_rows.append(dict(run=key, seed=RUN[key][3], beta_init=RUN[key][4], step=step,
                                      model_id=mid, file=os.path.basename(path), **bl))
    norm_rows.sort(key=lambda r: (r['run'], r['step'], r['block']))
    cols = ['run', 'seed', 'beta_init', 'step', 'model_id', 'file', 'block', 'gamma_W_fro', 'beta_W_fro',
            'beta_row_norm_mean', 'gamma_bias_mean', 'beta_bias_mean', 'beta_bias_absmean',
            'gamma_cos_to_init', 'gamma_orth_to_init_fro', 'beta_cos_to_init', 'beta_orth_to_init_fro']
    with open(os.path.join(OUT, 'se_fc2_norms.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in norm_rows:
            w.writerow({c: ('' if r[c] is None else round(r[c], 6) if isinstance(r[c], float) else r[c])
                        for c in cols})

    # 5. LR / momentum schedule from the logs (every logged [REPLAY] step line) -----------------
    pat = re.compile(r'\[REPLAY\] step=(\d+) .* lr=([0-9.e-]+) .*mom=([0-9.]+)')
    sched = {}
    for key, *_rest in RUNS:
        log = os.path.join(EXP, 'logs', RUN[key][6] + '.txt.gz')
        d = {}
        with gzip.open(log, 'rt', errors='replace') as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    d[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
        sched[key] = d
    ref = sched['se_sb']
    mismatch = [(k, s) for k in sched for s in sched[k] if s in ref and sched[k][s] != ref[s]]
    with open(os.path.join(OUT, 'lr_schedule.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['step', 'lr', 'momentum'])
        for s in sorted(ref):
            w.writerow([s, ref[s][0], ref[s][1]])
    if mismatch:
        raise SystemExit(f'LR schedule differs between runs at {len(mismatch)} steps, e.g. {mismatch[:5]}')
    print('lr schedule: steps', len(ref), 'max', max(ref), 'identical across all runs')
    for k in sched:
        print('  ', k, 'steps logged', len(sched[k]), 'max', max(sched[k]))
    print('norm rows', len(norm_rows))
    write_readme()


def count(name):
    with open(os.path.join(OUT, name)) as fh:
        return sum(1 for _ in fh) - 1


DATA_README = '''# Data files — SE style experiment

Every file here is generated by [`../final_data.py`](../final_data.py) (run it directly, or through
[`../make_final_report.py`](../make_final_report.py), which also rebuilds the charts and reports).
Do not edit them by hand. Row counts exclude the header.

Common terms:
- **run** — dashboard run key: `se_sb` / `se_att` / `se_none` (seed 1), `se_sb2` / `se_att2` / `se_none2`
  (seed 2), `se_zb1` / `se_zb2` (zero-β scale+bias, derived from the seed-1 / seed-2 scale+bias fresh nets).
- **arm codes** — `sb` scale+bias, `att` attenuate-only, `none` no SE, `zb` zero-β scale+bias.
- **step** — training step (`training_step` in the checkpoint metadata). 1k marks are the enumerated
  checkpoints; other steps are SIGINT stop saves.
- **pElo** — puzzle-rating estimate from `--probe-model <ckpt> --probe-set wide` (4,435 puzzles); higher is better.
- **nll** — mean negative log-likelihood of the puzzle solution moves on the same probe; lower is better.

## probes_all_runs.csv ({probes} rows)

Every probed checkpoint of all eight runs, long format. Source: `documentation/dashboards/data/<run>.csv`
(rows with a pElo), written by `documentation/dashboards/replay.py` (`track` / `probe_backfill`).

| column | meaning |
|---|---|
| run, arm, seed | run key, arm label, seed (1 or 2) |
| beta_init | `glorot` / `zero` for the scale+bias-style runs, blank otherwise |
| fresh_model_id | ModelID of the run's untrained starting net |
| step, is_1k_mark | training step; 1 if it is a 1000-step mark, 0 for a stop save |
| pElo, nll | probe results (see above) |
| loss, pLoss, vLoss | training-batch total / policy / value loss from the run log's `[REPLAY] step=` line at or just before this step (not a probe measurement) |
| pIllM | training batch: mean softmax mass on illegal moves, from the same log line |
| legalMass | 1 − pIllM, computed by the tracker |
| gNorm | training batch: pre-clip global gradient L2 norm, from the same log line |
| games_fed | cumulative corpus games fed by the runner at that step (blank where the tracker did not record it) |
| elapsed_train_sec | tracker's sleep-clamped training time at that step (s); not comparable across phases because GPU sharing differed |
| checkpoint_file | the checkpoint file that was probed |

## paired_gaps.csv ({gaps} rows)

Within-seed differences between two arms at the same 1k mark: `d = a − b`.
`comparison` is `a-b` with arm codes: `none-sb`, `none-att`, `att-sb`, `zb-sb` (zero-β − its own parent),
`zb-att`, `zb-none`. Only marks both runs reached are included.

| column | meaning |
|---|---|
| seed | 1 or 2 |
| comparison | `a-b` as above |
| step | 1k mark |
| a_pElo, b_pElo, d_pElo | pElo of a, of b, and a − b (positive = a ahead) |
| a_nll, b_nll, d_nll | nll of a, of b, and a − b (negative = a better) |

## seed_gaps.csv ({seedgaps} rows)

Seed 2 − seed 1 for the same arm at the same 1k mark (marks both seeds reached: 1k–7k for sb/att/none,
1k–5k for zb). Measures run-to-run noise between independent inits (for zb, between the two derived nets).

| column | meaning |
|---|---|
| arm | arm code |
| step | 1k mark |
| seed1_pElo, seed2_pElo, d_pElo | pElo per seed and seed 2 − seed 1 |
| seed1_nll, seed2_nll, d_nll | nll per seed and seed 2 − seed 1 |

## se_fc2_norms.csv ({norms} rows)

SE fc2 statistics for the four scale+bias-style runs (`se_sb`, `se_sb2`, `se_zb1`, `se_zb2`), one row per
checkpoint × block (3 blocks). Checkpoints: the run's fresh net (step 0, from `../models/`) plus every enumerated
checkpoint and stop save found in `~/Library/Application Support/DrewsChessMachine/Models/`. Each file is
accepted only if its `__metadata__` `model_id` is the fresh net's or its `parent_model_id` is; step comes from
`training_step`, never the filename. fc2 weight is [256, 32]: rows 0–127 give γ, rows 128–255 give β.

| column | meaning |
|---|---|
| run, seed, beta_init | run key, seed, `glorot` or `zero` |
| step, model_id, file | checkpoint metadata step and ModelID, and the file read |
| block | 0, 1, 2 |
| gamma_W_fro, beta_W_fro | Frobenius norm of the γ half / β half of the fc2 weight |
| beta_row_norm_mean | mean L2 norm of the 128 β rows (each row = one channel's 32 weights) |
| gamma_bias_mean | mean of the γ half of the fc2 bias |
| beta_bias_mean, beta_bias_absmean | mean and mean absolute value of the β half of the fc2 bias |
| gamma_cos_to_init, beta_cos_to_init | cosine between the flattened half and its step-0 value (blank when the init is all zero) |
| gamma_orth_to_init_fro, beta_orth_to_init_fro | norm of the component orthogonal to the step-0 value: 0 under pure weight decay, so it measures gradient-driven change of direction (equals the full norm when the init is zero) |

## lr_schedule.csv ({lr} rows)

Learning rate and momentum at every `[REPLAY] step=` line of the seed-1 scale+bias log
(`../logs/dcm_log_20260929-150727.txt.gz`, logged every 50 steps plus step 1). `final_data.py` checks that all
eight runs' logs agree with it at every step they logged; the report is generated only if they do.

| column | meaning |
|---|---|
| step | training step |
| lr | learning rate in effect |
| momentum | SGD momentum in effect |
'''


def write_readme():
    text = DATA_README.format(probes=count('probes_all_runs.csv'), gaps=count('paired_gaps.csv'),
                              seedgaps=count('seed_gaps.csv'), norms=count('se_fc2_norms.csv'),
                              lr=count('lr_schedule.csv'))
    with open(os.path.join(OUT, 'README.md'), 'w') as fh:
        fh.write(text)


if __name__ == '__main__':
    main()
