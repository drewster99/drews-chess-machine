#!/usr/bin/env python3
"""Causal intervention study for channel 13 amplitude (DCM).

Metric: ch13 max-abs over the 64 squares at the tower_final_bn INPUT tap.
Reuses the validated numpy forward + encoder + weights from analyze_channels.py.
Does NOT write to the model file.
"""
import json, sys
import numpy as np
import chess
import analyze_channels as A  # module-level: loads weights, defines forward/encode

OUTDIR = "/private/tmp/claude-501/-Users-andrew-cursor-drews-chess-machine/5778c88d-3bf8-448a-ad1a-c4dca906ffbc/scratchpad/analyze_net"
RES200 = A.RES200; RESWIDE = A.RESWIDE; PROBE = A.PROBE
CH = 13
N_BASE = 350
rng = np.random.default_rng(0)

def ch13_tower(boards):
    """max-abs over 64 squares of ch13 at tower_final_in, for a list of boards."""
    N = len(boards)
    X = np.zeros((N, 24, 8, 8), dtype=np.float32)
    for i, b in enumerate(boards):
        X[i] = A.encode_basic24(b)
    out = np.zeros(N)
    BS = 128
    for s in range(0, N, BS):
        e = min(N, s + BS)
        taps, _, _ = A.forward(X[s:e])
        Aarr = taps["tower_final_in"][:, CH, :, :].reshape(e - s, -1)
        out[s:e] = np.abs(Aarr).max(axis=1)
    return out

def legal_quiet(b):
    """valid position AND side-to-move not in check (opponent-in-check caught by is_valid)."""
    try:
        return b.is_valid() and not b.is_check()
    except Exception:
        return False

def with_edits(base, removes=(), adds=()):
    b = base.copy(stack=False)
    b.ep_square = None
    for sq in removes:
        b.remove_piece_at(sq)
    for sq, pc in adds:
        b.set_piece_at(sq, pc)
    b.castling_rights = b.clean_castling_rights()
    return b

# ---- perturbation builders: return a perturbed board or None ----
def remove_one_pawn(base):
    for sq in sorted(base.pieces(chess.PAWN, chess.WHITE) | base.pieces(chess.PAWN, chess.BLACK)):
        b = with_edits(base, removes=[sq])
        if legal_quiet(b):
            return b
    return None

def remove_one_minor(base):
    sqs = (base.pieces(chess.KNIGHT, chess.WHITE) | base.pieces(chess.KNIGHT, chess.BLACK)
           | base.pieces(chess.BISHOP, chess.WHITE) | base.pieces(chess.BISHOP, chess.BLACK))
    for sq in sorted(sqs):
        b = with_edits(base, removes=[sq])
        if legal_quiet(b):
            return b
    return None

def add_one_pawn(base):
    occ = set(base.piece_map().keys())
    for sq in range(64):
        r = chess.square_rank(sq)
        if r == 0 or r == 7:      # pawns illegal on ranks 1/8
            continue
        if sq in occ:
            continue
        b = with_edits(base, adds=[(sq, chess.Piece(chess.PAWN, base.turn))])
        if legal_quiet(b):
            return b
    return None

def move_one_pawn(base):
    """relocate a pawn to another empty rank2-7 square (count fixed, structure changed)."""
    occ = set(base.piece_map().keys())
    pawns = sorted(base.pieces(chess.PAWN, chess.WHITE) | base.pieces(chess.PAWN, chess.BLACK))
    for psq in pawns:
        pc = base.piece_at(psq)
        for tsq in range(64):
            r = chess.square_rank(tsq)
            if r == 0 or r == 7 or tsq in occ or tsq == psq:
                continue
            b = with_edits(base, removes=[psq], adds=[(tsq, pc)])
            if legal_quiet(b):
                return b
    return None

def construct_check(base):
    """relocate side-to-move's king to an empty square where it is in check.
    Material + pawn structure identical; only the king square changes."""
    turn = base.turn
    ksq = base.king(turn)
    if ksq is None:
        return None
    occ = set(base.piece_map().keys())
    for tsq in range(64):
        if tsq in occ or tsq == ksq:
            continue
        b = base.copy(stack=False)
        b.ep_square = None
        b.remove_piece_at(ksq)
        b.set_piece_at(tsq, chess.Piece(chess.KING, turn))
        b.castling_rights = b.clean_castling_rights()
        try:
            if b.is_valid() and b.is_check():
                return b
        except Exception:
            continue
    return None

def main():
    fen = {}
    for p in (RES200, RESWIDE):
        for e in json.load(open(p))["puzzles"]:
            fen[e["id"]] = e["fen"]
    probes = [json.loads(l) for l in open(PROBE)]

    # base set: quiet (not in check), valid, in order; cap at N_BASE
    bases = []
    for pr in probes:
        b = chess.Board(fen[pr["puzzleId"]])
        if b.is_valid() and not b.is_check():
            bases.append(b)
        if len(bases) >= N_BASE:
            break

    perts = {
        "remove_pawn": remove_one_pawn,
        "remove_minor": remove_one_minor,
        "add_pawn": add_one_pawn,
        "move_pawn_fixed_count": move_one_pawn,
        "construct_check": construct_check,
    }
    # expected sign of Δ (perturbed - base)
    exp_sign = {"remove_pawn": -1, "remove_minor": -1, "add_pawn": +1,
                "move_pawn_fixed_count": 0, "construct_check": -1}

    results = {}
    for name, fn in perts.items():
        pairs = []  # (base, perturbed)
        for b in bases:
            pb = fn(b)
            if pb is not None:
                pairs.append((b, pb))
        base_boards = [p[0] for p in pairs]
        pert_boards = [p[1] for p in pairs]
        base_v = ch13_tower(base_boards)
        pert_v = ch13_tower(pert_boards)
        d = pert_v - base_v
        es = exp_sign[name]
        if es == 0:
            pct = float(100.0 * np.mean(np.abs(d) > 0.1))  # fraction with non-trivial change
            sign_desc = "% |Δ|>0.1"
        else:
            pct = float(100.0 * np.mean(np.sign(d) == es))
            sign_desc = f"% sign=={es:+d}"
        results[name] = {
            "n": len(pairs),
            "mean_dCh13": float(d.mean()),
            "std_dCh13": float(d.std()),
            "mean_abs_dCh13": float(np.abs(d).mean()),
            "pct_expected_or_change": pct,
            "sign_desc": sign_desc,
            "base_mean": float(base_v.mean()),
        }

    # ---- #4b matched-subset in-check vs not, from per_position.csv (material-matched)
    import csv
    rows = list(csv.DictReader(open(f"{OUTDIR}/per_position.csv")))
    def g(r, k): return r[k]
    inchk = [r for r in rows if r["F_in_check"] == "1"]
    noch = [r for r in rows if r["F_in_check"] == "0"]
    # match each in-check row to not-in-check rows with same total_pawns and piece_total
    from collections import defaultdict
    buck = defaultdict(list)
    for r in noch:
        buck[(r["F_total_pawns"], r["F_piece_total"])].append(float(r["c13_tower_final_in_maxabs"]))
    diffs = []
    for r in inchk:
        key = (r["F_total_pawns"], r["F_piece_total"])
        if buck.get(key):
            diffs.append(float(r["c13_tower_final_in_maxabs"]) - np.mean(buck[key]))
    matched = {
        "n_incheck_matched": len(diffs),
        "mean_dCh13_incheck_minus_matched": float(np.mean(diffs)) if diffs else None,
        "std": float(np.std(diffs)) if diffs else None,
        "pct_negative": float(100.0 * np.mean(np.array(diffs) < 0)) if diffs else None,
    }

    # ---- #5 count-vs-structure: eta^2 of ch13 maxabs explained by total_pawns (all probes)
    y = np.array([float(r["c13_tower_final_in_maxabs"]) for r in rows])
    tp = np.array([int(r["F_total_pawns"]) for r in rows])
    grand = y.mean()
    ss_tot = ((y - grand) ** 2).sum()
    ss_between = 0.0
    for v in np.unique(tp):
        m = tp == v
        ss_between += m.sum() * (y[m].mean() - grand) ** 2
    eta2_pawncount = float(ss_between / ss_tot)
    # also piece_total
    ptv = np.array([int(r["F_piece_total"]) for r in rows])
    ssb2 = sum((ptv == v).sum() * (y[ptv == v].mean() - grand) ** 2 for v in np.unique(ptv))
    eta2_piece_total = float(ssb2 / ss_tot)

    out = {
        "metric": "ch13 max-abs over 64 squares at tower_final_bn input",
        "n_base_candidates": len(bases),
        "interventions": results,
        "check_matched_subset": matched,
        "count_vs_structure": {
            "eta2_total_pawns": eta2_pawncount,
            "eta2_piece_total": eta2_piece_total,
            "note": "eta^2 = fraction of across-position ch13-maxabs variance explained by the scalar (one-way ANOVA)."
        },
    }
    with open(f"{OUTDIR}/intervention_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # concise print
    print(f"BASE candidates: {len(bases)}  (metric: ch13 maxabs @ tower_final_bn input)\n")
    print(f"{'perturbation':24s} {'n':>4s} {'meanΔ':>9s} {'stdΔ':>8s} {'mean|Δ|':>8s}  expected")
    for name, r in results.items():
        print(f"{name:24s} {r['n']:4d} {r['mean_dCh13']:+9.3f} {r['std_dCh13']:8.3f} {r['mean_abs_dCh13']:8.3f}  {r['pct_expected_or_change']:5.1f}{r['sign_desc']}")
    print("\n#4b in-check vs material-matched not-in-check:", json.dumps(matched))
    print("\n#5 count vs structure:")
    print(f"  eta2(total_pawns)={eta2_pawncount:.3f}  eta2(piece_total)={eta2_piece_total:.3f}")

if __name__ == "__main__":
    main()
