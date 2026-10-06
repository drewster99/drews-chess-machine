"""Value-head CE by game stage on the corpus sample, against Stockfish's WDL on the same positions.
Same positions as relu_inputs.py's corpus mode (same shards, counts, seed and draw order)."""
import os, sys, json, time, numpy as np, chess, chess.engine
from concurrent.futures import ThreadPoolExecutor
sys.dont_write_bytecode = True  # importing the sibling folder's module must not leave __pycache__ there
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "20261004-head-logits-relu-inputs"))
import relu_inputs as ri

SF = os.path.expanduser("~/bin/stockfish")
NODES = int(os.environ.get("SF_NODES", "50000")); WORKERS = int(os.environ.get("SF_WORKERS", "8"))
LIMIT = int(os.environ.get("LIMIT", "0"))
VAL = {"P": 1, "N": 3, "B": 3, "R": 5, "Q": 9}

def sample(shards, games_per_shard, plies_per_game, seed):
    rng = np.random.default_rng(seed); out = []
    for sh in shards:
        games = ri.read_shard_games(os.path.join(ri.CORPUS, f"shard-{sh:05d}.dcmgames"))
        for gi in rng.choice(len(games), size=games_per_shard, replace=False):
            outcome, fen, moves = games[gi]
            if len(moves) < 2: continue
            b = chess.Board(fen) if fen else chess.Board(); tr = ri.Tracker(b)
            want = set(rng.choice(len(moves), size=min(plies_per_game, len(moves)), replace=False).tolist())
            for ply, pm in enumerate(moves):
                if ply in want:
                    stm_white = b.turn == chess.WHITE
                    target = 1 if outcome == 1 else (0 if (outcome == 0) == stm_white else 2)
                    mat = sum(VAL.get(p.symbol().upper(), 0) * (1 if p.color == b.turn else -1) for p in b.piece_map().values())
                    out.append(dict(x=ri.encode(b, tr), target=target, fen=b.fen(), ply=ply, left=len(moves) - ply, mat=mat))
                if ply >= max(want): break
                m = chess.Move(ri.dcm_sq_to_chess((pm >> 6) & 63), ri.dcm_sq_to_chess(pm & 63), ri.PROMO[(pm >> 12) & 7])
                if m not in b.legal_moves: raise ValueError(f"shard {sh} game {gi} ply {ply}: {m.uci()} illegal")
                tr.push(b, m)
    return out

def net_probs(path, X):
    md, arch, t = ri.load(path); sites = ri.dcm_arch.site_activations_md(md); V = []
    for i in range(0, len(X), 64):
        V.append(ri.forward(X[i:i + 64], arch, t, lambda *a: None, sites=sites)[1])
    V = np.concatenate(V); e = np.exp(V - V.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

def sf_eval(fens):
    res = [None] * len(fens)
    def work(chunk):
        eng = chess.engine.SimpleEngine.popen_uci(SF)
        eng.configure({"Threads": 1, "Hash": 64, "UCI_ShowWDL": True})
        try:
            for i in chunk:
                b = chess.Board(fens[i]); info = eng.analyse(b, chess.engine.Limit(nodes=NODES))
                w = info["wdl"].pov(b.turn); sc = info["score"].pov(b.turn)
                res[i] = ((w.wins / 1000, w.draws / 1000, w.losses / 1000), sc.score(mate_score=10000))
        finally:
            eng.quit()
    chunks = [list(range(k, len(fens), WORKERS)) for k in range(WORKERS)]
    with ThreadPoolExecutor(WORKERS) as ex: list(ex.map(work, chunks))
    return res

def ce(P, y): return float(np.mean(-np.log(np.clip(P[np.arange(len(y)), y], 1e-6, 1))))

def fit_calibration(x, y, iters=3000, lr=0.5):
    """3-class softmax(a + b·x), x = tanh(cp/400): a fair recalibration of Stockfish's eval to these games' results."""
    a = np.zeros(3); b = np.zeros(3); Y = np.eye(3)[y]
    for _ in range(iters):
        z = a + np.outer(x, b); z -= z.max(1, keepdims=True); p = np.exp(z); p /= p.sum(1, keepdims=True)
        g = p - Y; a -= lr * g.mean(0); b -= lr * (g * x[:, None]).mean(0)
    return a, b

def apply_cal(a, b, x):
    z = a + np.outer(x, b); z -= z.max(1, keepdims=True); p = np.exp(z); return p / p.sum(1, keepdims=True)

if __name__ == "__main__":
    t0 = time.time()
    pos = sample([20, 26, 32, 38], 1024, 2, 20261004)
    if LIMIT: pos = pos[:LIMIT]
    y = np.array([p["target"] for p in pos]); X = np.stack([p["x"] for p in pos])
    print(f"positions {len(pos)}  sampled in {time.time()-t0:.0f}s", flush=True)
    P = net_probs(os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Models/20261002-bench_v5s3_noSE_noReZero-replay-step33000.safetensors"), X)
    print(f"net done {time.time()-t0:.0f}s", flush=True)
    sf = sf_eval([p["fen"] for p in pos]); print(f"stockfish done {time.time()-t0:.0f}s", flush=True)
    W = np.array([s[0] for s in sf]); cp = np.array([s[1] for s in sf], float); xcp = np.tanh(cp / 400)
    # 2-fold cross-validated calibration (fit on one half, score the other)
    idx = np.arange(len(y)); fold = idx % 2; Pcal = np.zeros((len(y), 3))
    for k in (0, 1):
        a, b = fit_calibration(xcp[fold != k], y[fold != k]); Pcal[fold == k] = apply_cal(a, b, xcp[fold == k])
    base = np.bincount(y, minlength=3) / len(y)
    rows = []
    def bucket(name, mask):
        if mask.sum() == 0: return
        yy = y[mask]; bb = np.bincount(yy, minlength=3) / len(yy)
        rows.append(dict(bucket=name, n=int(mask.sum()), base=ce(np.tile(base, (mask.sum(), 1)), yy),
                         bucket_rate=float(-sum(q * np.log(q) for q in bb if q > 0)),
                         net=ce(P[mask], yy), sf_raw=ce(np.clip(W[mask], 1e-3, 1), yy), sf_cal=ce(Pcal[mask], yy)))
    ply = np.array([p["ply"] for p in pos]); left = np.array([p["left"] for p in pos]); mat = np.array([p["mat"] for p in pos])
    bucket("all", np.ones(len(y), bool))
    for lo, hi in [(0, 10), (11, 30), (31, 60), (61, 100), (101, 10**6)]: bucket(f"ply {lo}-{hi}", (ply >= lo) & (ply <= hi))
    for lo, hi in [(1, 10), (11, 30), (31, 60), (61, 10**6)]: bucket(f"plies left {lo}-{hi}", (left >= lo) & (left <= hi))
    for lo, hi in [(0, 0), (1, 2), (3, 5), (6, 99)]: bucket(f"|material| {lo}-{hi}", (np.abs(mat) >= lo) & (np.abs(mat) <= hi))
    json.dump(dict(nodes=NODES, positions=len(pos), rows=rows), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "value_stage.json"), "w"), indent=1)
    print(f"{'bucket':22s} {'n':>5s} {'base':>6s} {'bktrate':>7s} {'net':>6s} {'sf_raw':>6s} {'sf_cal':>6s}")
    for r in rows: print(f"{r['bucket']:22s} {r['n']:5d} {r['base']:6.3f} {r['bucket_rate']:7.3f} {r['net']:6.3f} {r['sf_raw']:6.3f} {r['sf_cal']:6.3f}")
    print(f"total {time.time()-t0:.0f}s")
