#!/usr/bin/env python3
"""DCM channel analysis: numpy forward reimplementation + feature correlation.

Model: 20261011-zlra-elitecyc-latest-step167000.safetensors (model_id 20261016-2-qGIj)
Arch (from file __metadata__): basic24 input, stem 5x5 conv -> BN (no act, pre-act tower),
  3 pre-activation residual blocks (128ch, conv 9x9/9x9, SiLU, clean-add, res_ln LayerNorm,
  NO SE, NO ReZero), tower_final_bn -> SiLU, policy intermediate_conv head, value WDL head.
Computed in fp32 (app runs bf16); validated against recorded top1Move + W/D/L.
"""
import json, struct, sys, math
import numpy as np
import chess

MODEL = "/Users/andrew/Library/Application Support/DrewsChessMachine/Models/20261011-zlra-elitecyc-latest-step167000.safetensors"
PROBE = "/private/tmp/claude-501/-Users-andrew-cursor-drews-chess-machine/5778c88d-3bf8-448a-ad1a-c4dca906ffbc/scratchpad/analyze_net/probe_positions.jsonl"
RES200 = "/Users/andrew/cursor/drews-chess-machine/DrewsChessMachine/DrewsChessMachine/Resources/lichess_probes_200.json"
RESWIDE = "/Users/andrew/cursor/drews-chess-machine/DrewsChessMachine/DrewsChessMachine/Resources/lichess_probes_wide.json"
OUTDIR = "/private/tmp/claude-501/-Users-andrew-cursor-drews-chess-machine/5778c88d-3bf8-448a-ad1a-c4dca906ffbc/scratchpad/analyze_net"
EPS = 1e-5
TARGET_CH = [13, 25, 48]

# ---------------------------------------------------------------- safetensors
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = f.tell()
        data = f.read()
    T = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        if v["dtype"] != "F32":
            continue
        a, b = v["data_offsets"]
        arr = np.frombuffer(data[a:b], dtype="<f4").astype(np.float32).copy()
        T[k] = arr.reshape(v["shape"])
    return T, hdr.get("__metadata__", {})

W, META = load_safetensors(MODEL)

# ---------------------------------------------------------------- forward ops
def conv2d(x, w, pad):
    # x [N,C,H,W], w [O,C,K,K] OIHW cross-correlation, returns [N,O,H,W]
    N, C, H, Wd = x.shape
    O, Ci, K, _ = w.shape
    assert Ci == C
    if pad:
        x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    Hp, Wp = x.shape[2], x.shape[3]
    Ho, Wo = Hp - K + 1, Wp - K + 1
    win = np.lib.stride_tricks.sliding_window_view(x, (K, K), axis=(2, 3))  # [N,C,Ho,Wo,K,K]
    win = win.transpose(0, 2, 3, 1, 4, 5).reshape(N, Ho * Wo, C * K * K)
    wm = w.reshape(O, C * K * K)
    out = win @ wm.T  # [N, Ho*Wo, O]
    return out.transpose(0, 2, 1).reshape(N, O, Ho, Wo)

def bn_inf(x, pfx):
    g = W[pfx + ".weight"].reshape(1, -1, 1, 1)
    b = W[pfx + ".bias"].reshape(1, -1, 1, 1)
    m = W[pfx + ".running_mean"].reshape(1, -1, 1, 1)
    v = W[pfx + ".running_var"].reshape(1, -1, 1, 1)
    return g * (x - m) / np.sqrt(v + EPS) + b

def silu(x):
    return x / (1.0 + np.exp(-x))

def layernorm_c(x, pfx):
    # over channel axis (axis=1), per (n,h,w); population variance; eps 1e-5
    g = W[pfx + ".weight"].reshape(1, -1, 1, 1)
    b = W[pfx + ".bias"].reshape(1, -1, 1, 1)
    m = x.mean(axis=1, keepdims=True)
    v = x.var(axis=1, keepdims=True)  # ddof=0 population
    return g * (x - m) / np.sqrt(v + EPS) + b

def forward(x):
    """x [N,24,8,8] -> dict of taps + policy logits [N,4864] + wdl [N,3]."""
    taps = {}
    # stem: conv5x5 pad2 -> bn -> (no act)
    h = conv2d(x, W["stem.conv.weight"], pad=2)
    h = bn_inf(h, "stem.bn")
    taps["block0_in"] = h  # input to blocks.0.bn1
    for i in range(3):
        x_in = h
        a = bn_inf(x_in, f"blocks.{i}.bn1")
        a = silu(a)
        a = conv2d(a, W[f"blocks.{i}.conv1.weight"], pad=4)
        a = bn_inf(a, f"blocks.{i}.bn2")
        a = silu(a)
        a = conv2d(a, W[f"blocks.{i}.conv2.weight"], pad=4)
        merged = x_in + a  # clean add (no SE, no ReZero)
        h = layernorm_c(merged, f"blocks.{i}.res_ln")
        if i == 0:
            taps["block1_in"] = h
        elif i == 1:
            taps["block2_in"] = h
        elif i == 2:
            taps["tower_final_in"] = h  # input to tower_final_bn
    # tower end
    t = bn_inf(h, "tower_final_bn")
    t = silu(t)
    # policy head: pre_conv 1x1 -> pre_bn -> silu -> conv 1x1 (+bias)
    p = conv2d(t, W["policy.pre_conv.weight"], pad=0)
    p = bn_inf(p, "policy.pre_bn")
    p = silu(p)
    p = conv2d(p, W["policy.conv.weight"], pad=0) + W["policy.conv.bias"].reshape(1, -1, 1, 1)
    N = x.shape[0]
    logits = p.reshape(N, -1)  # NCHW row-major -> idx = c*64 + r*8 + col
    # value head: conv 1x1 (128->24) -> bn -> silu -> flatten -> fc1 -> silu -> fc2 -> softmax
    vv = conv2d(t, W["value.conv.weight"], pad=0)
    vv = bn_inf(vv, "value.bn")
    vv = silu(vv)
    vv = vv.reshape(N, -1)  # [N,1536] channel-major
    vv = vv @ W["value.fc1.weight"].T + W["value.fc1.bias"]  # [out,in] -> x@W.T
    vv = silu(vv)
    vv = vv @ W["value.wdl_fc2.weight"].T + W["value.wdl_fc2.bias"]  # [N,3]
    vv = vv - vv.max(axis=1, keepdims=True)
    e = np.exp(vv)
    wdl = e / e.sum(axis=1, keepdims=True)  # [win,draw,loss]
    return taps, logits, wdl

# ---------------------------------------------------------------- encoding
PT_IDX = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}

def enc_row(app_row, flip):
    return (7 - app_row) if flip else app_row

def encode_basic24(board):
    x = np.zeros((24, 8, 8), dtype=np.float32)
    flip = (board.turn == chess.BLACK)
    persp = board.turn
    for sq, piece in board.piece_map().items():
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)   # 0 = rank1
        app_row = 7 - rank             # row0 = rank8
        r = enc_row(app_row, flip)
        c = file
        mine = (piece.color == persp)
        plane = (0 if mine else 6) + PT_IDX[piece.piece_type]
        x[plane, r, c] = 1.0
    # castling 12-15: my K, my Q, opp K, opp Q
    if flip:
        myK = board.has_kingside_castling_rights(chess.BLACK)
        myQ = board.has_queenside_castling_rights(chess.BLACK)
        opK = board.has_kingside_castling_rights(chess.WHITE)
        opQ = board.has_queenside_castling_rights(chess.WHITE)
    else:
        myK = board.has_kingside_castling_rights(chess.WHITE)
        myQ = board.has_queenside_castling_rights(chess.WHITE)
        opK = board.has_kingside_castling_rights(chess.BLACK)
        opQ = board.has_queenside_castling_rights(chess.BLACK)
    if myK: x[12, :, :] = 1.0
    if myQ: x[13, :, :] = 1.0
    if opK: x[14, :, :] = 1.0
    if opQ: x[15, :, :] = 1.0
    # ep plane 16
    if board.ep_square is not None:
        ef = chess.square_file(board.ep_square)
        er = chess.square_rank(board.ep_square)
        ar = 7 - er
        x[16, enc_row(ar, flip), ef] = 1.0
    # plane 17 halfmove clock
    norm = min(board.halfmove_clock, 99) / 99.0
    if norm > 0:
        x[17, :, :] = norm
    # planes 18-23: repetition -> 0 (no history in reconstructed position)
    return x

# policy index for a python-chess move (encoder frame)
QDIR = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
KJMP = [(-2,1),(-1,2),(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1)]
PROMO_PIECE = {chess.QUEEN: ("q", 73), chess.KNIGHT: ("n", 64 + 0*3), chess.ROOK: ("r", 64 + 1*3), chess.BISHOP: ("b", 64 + 2*3)}

def policy_index(move, flip):
    ff = chess.square_file(move.from_square); fr = chess.square_rank(move.from_square)
    tf = chess.square_file(move.to_square);   tr = chess.square_rank(move.to_square)
    from_app = 7 - fr; to_app = 7 - tr
    fR = enc_row(from_app, flip); tR = enc_row(to_app, flip)
    fC = ff; tC = tf
    dr = tR - fR; dc = tC - fC
    if move.promotion is not None:
        # dr must be -1 in encoder frame
        if dc == 0: di = 0
        elif dc == -1: di = 1
        elif dc == 1: di = 2
        else: return None
        if move.promotion == chess.QUEEN:
            ch = 73 + di
        elif move.promotion == chess.KNIGHT:
            ch = 64 + 0*3 + di
        elif move.promotion == chess.ROOK:
            ch = 64 + 1*3 + di
        elif move.promotion == chess.BISHOP:
            ch = 64 + 2*3 + di
        else: return None
        return ch * 64 + fR * 8 + fC
    if (dr, dc) in KJMP:
        ch = 56 + KJMP.index((dr, dc))
        return ch * 64 + fR * 8 + fC
    dist = max(abs(dr), abs(dc))
    sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
    di = QDIR.index((sdr, sdc))
    ch = di * 7 + (dist - 1)
    return ch * 64 + fR * 8 + fC

# ---------------------------------------------------------------- features
MAT = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

def features(board):
    persp = board.turn
    f = {}
    own_mat = opp_mat = 0
    counts = {}
    total = 0
    pawns_own = pawns_opp = 0
    nonpawn = 0
    for sq, pc in board.piece_map().items():
        total += 1
        mine = (pc.color == persp)
        side = "own" if mine else "opp"
        key = f"{side}_{chess.piece_name(pc.piece_type)}"
        counts[key] = counts.get(key, 0) + 1
        if mine: own_mat += MAT[pc.piece_type]
        else: opp_mat += MAT[pc.piece_type]
        if pc.piece_type == chess.PAWN:
            if mine: pawns_own += 1
            else: pawns_opp += 1
        if pc.piece_type != chess.PAWN and pc.piece_type != chess.KING:
            nonpawn += MAT[pc.piece_type]
    f["piece_total"] = total
    f["own_material"] = own_mat
    f["opp_material"] = opp_mat
    f["material_imbalance"] = own_mat - opp_mat
    f["abs_material_imbalance"] = abs(own_mat - opp_mat)
    f["own_pawns"] = pawns_own
    f["opp_pawns"] = pawns_opp
    f["total_pawns"] = pawns_own + pawns_opp
    f["nonpawn_material"] = nonpawn
    for t in ["knight", "bishop", "rook", "queen"]:
        f[f"own_{t}"] = counts.get(f"own_{t}", 0)
        f[f"opp_{t}"] = counts.get(f"opp_{t}", 0)
    f["queens_present"] = 1 if (counts.get("own_queen", 0) + counts.get("opp_queen", 0)) > 0 else 0
    f["in_check"] = 1 if board.is_check() else 0
    f["legal_count"] = board.legal_moves.count()
    # phase: endgame indicator by total pieces
    f["phase_endgame"] = 1 if total <= 12 else 0
    f["phase_opening"] = 1 if total >= 28 else 0
    return f

# ---------------------------------------------------------------- run
def main():
    # puzzle id -> fen
    fen = {}
    for path in (RES200, RESWIDE):
        d = json.load(open(path))
        for p in d["puzzles"]:
            fen[p["id"]] = p["fen"]

    probes = [json.loads(l) for l in open(PROBE)]
    print(f"probe lines: {len(probes)}", file=sys.stderr)

    # encode all
    X = np.zeros((len(probes), 24, 8, 8), dtype=np.float32)
    boards = []
    flips = []
    miss = 0
    for i, pr in enumerate(probes):
        pid = pr["puzzleId"]
        if pid not in fen:
            miss += 1
            boards.append(None); flips.append(False); continue
        b = chess.Board(fen[pid])
        boards.append(b)
        flips.append(b.turn == chess.BLACK)
        X[i] = encode_basic24(b)
    print(f"missing fen: {miss}", file=sys.stderr)

    # forward in batches
    N = len(probes)
    BS = 64
    tap_names = ["block0_in", "block1_in", "block2_in", "tower_final_in"]
    # per-channel per-tap: meanabs(mean over 64), maxabs
    ch_mean = {t: {c: np.zeros(N) for c in TARGET_CH} for t in tap_names}
    ch_maxabs = {t: {c: np.zeros(N) for c in TARGET_CH} for t in tap_names}
    top1_idx = np.full(N, -1)
    top1_prob = np.zeros(N)
    wdl_all = np.zeros((N, 3))
    for s in range(0, N, BS):
        e = min(N, s + BS)
        taps, logits, wdl = forward(X[s:e])
        wdl_all[s:e] = wdl
        for t in tap_names:
            A = taps[t]  # [n,128,8,8]
            for c in TARGET_CH:
                ch = A[:, c, :, :].reshape(A.shape[0], -1)
                ch_mean[t][c][s:e] = ch.mean(axis=1)
                ch_maxabs[t][c][s:e] = np.abs(ch).max(axis=1)
        # full temp-1 softmax over all logits for top1Prob
        lg = logits - logits.max(axis=1, keepdims=True)
        sm = np.exp(lg); sm /= sm.sum(axis=1, keepdims=True)
        for j in range(e - s):
            gi = s + j
            b = boards[gi]
            if b is None:
                continue
            best = None; bestlog = -1e30; bestidx = -1
            for mv in b.legal_moves:
                idx = policy_index(mv, flips[gi])
                if idx is None or idx < 0 or idx >= logits.shape[1]:
                    continue
                if logits[j, idx] > bestlog:
                    bestlog = logits[j, idx]; best = mv; bestidx = idx
            if best is not None:
                top1_idx[gi] = bestidx
                top1_prob[gi] = sm[j, bestidx]
                probes[gi]["_pred_uci"] = best.uci()
                probes[gi]["_pred_prob"] = float(sm[j, bestidx])

    # ---- self-check
    n_ok = n_cmp = 0
    verr = []
    pperr = []
    for i, pr in enumerate(probes):
        if "_pred_uci" not in pr:
            continue
        n_cmp += 1
        if pr["_pred_uci"] == pr["top1Move"]:
            n_ok += 1
        verr.append(abs(wdl_all[i, 0] - pr["valueWin"]))
        pperr.append(abs(top1_prob[i] - pr.get("top1Prob", np.nan)))
    verr = np.array(verr); pperr = np.array(pperr)
    selfcheck = {
        "n_compared": n_cmp,
        "top1_match": n_ok,
        "top1_match_pct": 100.0 * n_ok / max(1, n_cmp),
        "valueWin_mae": float(np.nanmean(verr)),
        "valueWin_max_abs_err": float(np.nanmax(verr)),
        "top1prob_mae": float(np.nanmean(pperr)),
    }
    print("SELF-CHECK", json.dumps(selfcheck, indent=2), file=sys.stderr)

    # ---- features
    feat_rows = []
    for i, pr in enumerate(probes):
        b = boards[i]
        if b is None:
            feat_rows.append(None); continue
        f = features(b)
        f["_i"] = i
        f["set"] = pr.get("set")
        feat_rows.append(f)
    feat_keys = [k for k in feat_rows[0].keys() if not k.startswith("_") and k != "set"] if feat_rows and feat_rows[0] else []

    # ---- correlations (Pearson + Spearman) on positions with fen
    def pearson(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if a.std() < 1e-12 or b.std() < 1e-12:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return pearson(ra, rb)

    valid = [i for i in range(N) if boards[i] is not None]
    corr = {}  # corr[ch][tap][metric][feat] = {pearson, spearman}
    std_report = {}
    for c in TARGET_CH:
        corr[c] = {}
        std_report[c] = {}
        for t in tap_names:
            corr[c][t] = {}
            std_report[c][t] = {
                "meanact_std": float(ch_mean[t][c][valid].std()),
                "meanact_mean": float(ch_mean[t][c][valid].mean()),
                "maxabs_std": float(ch_maxabs[t][c][valid].std()),
                "maxabs_mean": float(ch_maxabs[t][c][valid].mean()),
                "maxabs_cv": float(ch_maxabs[t][c][valid].std() / (abs(ch_maxabs[t][c][valid].mean()) + 1e-9)),
            }
            for metric, arr in (("mean", ch_mean[t][c]), ("maxabs", ch_maxabs[t][c])):
                corr[c][t][metric] = {}
                av = arr[valid]
                for fk in feat_keys:
                    fv = np.array([feat_rows[i][fk] for i in valid], float)
                    corr[c][t][metric][fk] = {"pearson": pearson(av, fv), "spearman": spearman(av, fv)}

    # ---- save JSON
    out = {
        "model": MODEL, "model_id": META.get("model_id"),
        "n_positions": N, "n_valid": len(valid),
        "selfcheck": selfcheck,
        "channel_stats": std_report,
        "correlations": corr,
    }
    with open(f"{OUTDIR}/results.json", "w") as f:
        json.dump(out, f, indent=2)

    # ---- per-position CSV
    with open(f"{OUTDIR}/per_position.csv", "w") as f:
        hdr = ["i", "puzzleId", "set", "pred_uci", "rec_top1", "match",
               "pred_prob", "rec_top1prob", "win", "draw", "loss", "rec_win"]
        for fk in feat_keys:
            hdr.append("F_" + fk)
        for c in TARGET_CH:
            for t in tap_names:
                hdr.append(f"c{c}_{t}_mean")
                hdr.append(f"c{c}_{t}_maxabs")
        f.write(",".join(hdr) + "\n")
        for i, pr in enumerate(probes):
            if boards[i] is None:
                continue
            row = [i, pr["puzzleId"], pr.get("set"), pr.get("_pred_uci", ""), pr["top1Move"],
                   int(pr.get("_pred_uci") == pr["top1Move"]),
                   f"{top1_prob[i]:.6f}", f"{pr.get('top1Prob', float('nan')):.6f}",
                   f"{wdl_all[i,0]:.5f}", f"{wdl_all[i,1]:.5f}", f"{wdl_all[i,2]:.5f}",
                   f"{pr['valueWin']:.5f}"]
            for fk in feat_keys:
                row.append(feat_rows[i][fk])
            for c in TARGET_CH:
                for t in tap_names:
                    row.append(f"{ch_mean[t][c][i]:.5f}")
                    row.append(f"{ch_maxabs[t][c][i]:.5f}")
            f.write(",".join(str(x) for x in row) + "\n")

    # ---- print concise report
    print("\n==== CHANNEL STD (across-position) ====", file=sys.stderr)
    for c in TARGET_CH:
        print(f"\nch{c}:", file=sys.stderr)
        for t in tap_names:
            s = std_report[c][t]
            print(f"  {t:16s} maxabs mean={s['maxabs_mean']:9.3f} std={s['maxabs_std']:9.4f} cv={s['maxabs_cv']:.4f} | meanact mean={s['meanact_mean']:9.3f} std={s['meanact_std']:8.4f}", file=sys.stderr)

    print("\n==== TOP FEATURE CORRELATIONS (maxabs @ tower_final_in) ====", file=sys.stderr)
    for c in TARGET_CH:
        items = sorted(corr[c]["tower_final_in"]["maxabs"].items(), key=lambda kv: -abs(kv[1]["pearson"]))
        print(f"\nch{c} (tower_final_in, maxabs):", file=sys.stderr)
        for fk, v in items[:8]:
            print(f"   {fk:24s} pearson={v['pearson']:+.3f} spearman={v['spearman']:+.3f}", file=sys.stderr)

    # growth block-to-block of maxabs (mean over positions)
    print("\n==== RESIDUAL-STREAM MAGNITUDE GROWTH (mean maxabs over valid positions) ====", file=sys.stderr)
    for c in TARGET_CH:
        vals = [std_report[c][t]["maxabs_mean"] for t in tap_names]
        print(f"  ch{c}: " + "  ".join(f"{t.split('_')[0]}={v:.2f}" for t, v in zip(tap_names, vals)), file=sys.stderr)

if __name__ == "__main__":
    main()
