"""Per-channel statistics of every ReLU input, and of the pre-softmax policy and value
logits, for DCM corpus-replay checkpoints, computed by an fp32 numpy forward pass on the
same 4,097 positions the app's --analyze-numerics uses (the start position, then every
ply of the Lichess bot's filed games, oldest file first, capped at 4,096), or, with
POSITIONS=corpus, a seeded sample of the training corpus (corpus_positions; no start
position, so start_wdl is null there).

The forward pass mirrors ChessNetwork.swift in inference mode (pre-activation tower,
clean_add skip, LayerNorm block output, no SE, no ReZero), BoardEncoder.swift (basic30)
and PolicyEncoding.swift. It is checked against the app's own audit JSON for the same
checkpoint before any number is used.

Usage: relu_inputs.py <out.json> <label>=<model.safetensors> ..."""
import glob, json, math, os, struct, sys
import numpy as np
import chess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import dcm_arch

GAMES = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/LichessBot/Games")
CAP = 4096
EPS = 1e-5

# ---------------------------------------------------------------- model file
def load(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    h = json.loads(raw[8:8 + n]); md = h.pop("__metadata__"); base = 8 + n
    t = {}
    for k, v in h.items():
        s, e = v["data_offsets"]
        t[k] = np.frombuffer(raw[base + s:base + e], dtype=np.float32).reshape(v["shape"]).copy()
    return md, dcm_arch.norm_arch_md(md), t

# ---------------------------------------------------------------- positions
def row_col(sq):            # python-chess square -> DCM (row 0 = rank 8, col 0 = file a)
    return 7 - chess.square_rank(sq), chess.square_file(sq)

class Tracker:
    """ChessGameEngine's repetition bookkeeping: positionCounts / recentPositionKeys,
    both cleared on an irreversible move (halfmove clock 0)."""
    def __init__(self, board):
        self.counts = {self.key(board): 1}
        self.recent = []
        self.rep = 0; self.mask = 0
    @staticmethod
    def key(b):
        ep = b.ep_square  # set after every double push, as DCM's enPassantSquare
        return (b.board_fen(), b.turn, b.has_kingside_castling_rights(chess.WHITE), b.has_queenside_castling_rights(chess.WHITE),
                b.has_kingside_castling_rights(chess.BLACK), b.has_queenside_castling_rights(chess.BLACK), ep)
    def push(self, b, move):
        prior = self.key(b)
        b.push(move)
        if b.halfmove_clock == 0:
            self.counts = {}; self.recent = []
        else:
            self.recent.insert(0, prior); self.recent = self.recent[:10]
        k = self.key(b)
        visits = self.counts.get(k, 0) + 1; self.counts[k] = visits
        self.mask = sum(1 << i for i, rk in enumerate(self.recent) if rk == k)
        self.rep = min(visits - 1, 2)

def encode(b, tr):
    x = np.zeros((30, 8, 8), np.float32)
    me = b.turn; flip = me == chess.BLACK
    for sq, pc in b.piece_map().items():
        r, c = row_col(sq)
        r = 7 - r if flip else r
        plane = (0 if pc.color == me else 6) + (pc.piece_type - 1)  # pawn..king -> 0..5
        x[plane, r, c] = 1
    mk = b.has_kingside_castling_rights(me); mq = b.has_queenside_castling_rights(me)
    ok = b.has_kingside_castling_rights(not me); oq = b.has_queenside_castling_rights(not me)
    for p, v in ((12, mk), (13, mq), (14, ok), (15, oq)):
        if v: x[p] = 1
    if b.ep_square is not None:
        r, c = row_col(b.ep_square); r = 7 - r if flip else r
        x[16, r, c] = 1
    hm = min(b.halfmove_clock, 99) / 99.0
    if hm > 0: x[17] = hm
    x[18] = 1.0 if tr.rep >= 1 else 0.0
    x[19] = 1.0 if tr.rep >= 2 else 0.0
    for i in range(10):
        if (tr.mask >> i) & 1: x[20 + i] = 1
    return x

QDIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
KNIGHT = [(-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1)]
def policy_index(move, turn):
    flip = turn == chess.BLACK
    fr, fc = row_col(move.from_square); tr_, tc = row_col(move.to_square)
    if flip: fr, tr_ = 7 - fr, 7 - tr_
    dr, dc = tr_ - fr, tc - fc
    if move.promotion:
        d = {0: 0, -1: 1, 1: 2}[dc]
        ch = {chess.QUEEN: 73 + d, chess.KNIGHT: 64 + d, chess.ROOK: 67 + d, chess.BISHOP: 70 + d}[move.promotion]
    elif (dr, dc) in KNIGHT:
        ch = 56 + KNIGHT.index((dr, dc))
    else:
        dist = max(abs(dr), abs(dc)); step = (int(np.sign(dr)), int(np.sign(dc)))
        ch = QDIRS.index(step) * 7 + dist - 1
    return ch * 64 + fr * 8 + fc

def positions():
    out = []
    b = chess.Board(); tr = Tracker(b)
    out.append((encode(b, tr), [policy_index(m, b.turn) for m in b.legal_moves]))
    files = sorted(glob.glob(os.path.join(GAMES, "**", "*.json"), recursive=True))
    games = 0
    for f in files:
        if len(out) - 1 >= CAP: break
        rec = json.load(open(f))
        if rec["setup"]["variant"] != "standard": continue
        fen = rec["setup"]["initialFen"]
        b = chess.Board() if fen in ("startpos", chess.STARTING_FEN) else chess.Board(fen)
        tr = Tracker(b); took = 0
        for mv in rec["moves"]:
            if len(out) - 1 >= CAP: break
            legal = list(b.legal_moves)
            if not legal: break
            out.append((encode(b, tr), [policy_index(m, b.turn) for m in legal])); took += 1
            m = chess.Move.from_uci(mv["uciAsGiven"])
            if m not in legal:  # castling given king-takes-rook
                m = b.parse_uci(mv["uciAsGiven"])
            tr.push(b, m)
        if took: games += 1
    return out, games

CORPUS = os.path.expanduser("~/Library/Application Support/DrewsChessMachine/Corpora/20260624-192615-w3aA5b")

def read_shard_games(path):
    """Every game of a sealed shard: (outcome 0=white win 1=draw 2=black win, start FEN or None, [(from_sq, to_sq, promo)])
    in DCM square numbering (0 = a8). Framing: 256-byte front header, records len|payload|crc32, 64-byte trailer."""
    raw = open(path, "rb").read()
    assert raw[:8] == b"DCMGAME1" and raw[-64:-56] == b"DCMGSEAL", path
    games = []; i = 256; end = len(raw) - 64
    while i < end:
        n = struct.unpack_from("<I", raw, i)[0]; i += 4
        pay = raw[i:i + n]; i += n + 4
        flags, outcome, _reason = pay[0], pay[1], pay[2]
        mc = struct.unpack_from("<I", pay, 3)[0]
        moves = [struct.unpack_from("<H", pay, 7 + 2 * k)[0] for k in range(mc)]
        fen = None
        if flags & 1:
            fl = struct.unpack_from("<H", pay, 7 + 2 * mc)[0]
            fen = pay[9 + 2 * mc:9 + 2 * mc + fl].decode()
        games.append((outcome, fen, moves))
    assert i == end, (path, i, end)
    return games

def dcm_sq_to_chess(sq):    # DCM 0 = a8 -> python-chess 0 = a1
    return chess.square(sq % 8, 7 - sq // 8)

PROMO = {0: None, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK, 4: chess.QUEEN}

def corpus_positions(shards, games_per_shard, plies_per_game, seed):
    """A seeded sample from shards the runs never reached (they trained on shards 0-9):
    `games_per_shard` games per shard, `plies_per_game` distinct plies per game, each
    replayed from the start with DCM's repetition bookkeeping. Every move is checked
    legal. Each position carries its value target from the side to move's view
    (0 win, 1 draw, 2 loss)."""
    rng = np.random.default_rng(seed)
    out = []; used_games = 0
    for sh in shards:
        games = read_shard_games(os.path.join(CORPUS, f"shard-{sh:05d}.dcmgames"))
        for gi in rng.choice(len(games), size=games_per_shard, replace=False):
            outcome, fen, moves = games[gi]
            if len(moves) < 2: continue
            b = chess.Board(fen) if fen else chess.Board(); tr = Tracker(b)
            want = set(rng.choice(len(moves), size=min(plies_per_game, len(moves)), replace=False).tolist())
            for ply, pm in enumerate(moves):
                if ply in want:
                    legal = list(b.legal_moves)
                    stm_white = b.turn == chess.WHITE
                    target = 1 if outcome == 1 else (0 if (outcome == 0) == stm_white else 2)
                    out.append((encode(b, tr), [policy_index(m, b.turn) for m in legal], target))
                if ply >= max(want): break
                m = chess.Move(dcm_sq_to_chess((pm >> 6) & 63), dcm_sq_to_chess(pm & 63), PROMO[(pm >> 12) & 7])
                if m not in b.legal_moves:
                    raise ValueError(f"shard {sh} game {gi} ply {ply}: {m.uci()} is not legal")
                tr.push(b, m)
            used_games += 1
    return out, used_games

# ---------------------------------------------------------------- forward pass
def conv(x, w, bias=None):
    """x [N,C,8,8], w [O,C,k,k], same padding, stride 1."""
    k = w.shape[2]; p = k // 2; n = x.shape[0]
    xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
    cols = np.empty((n, 8, 8, x.shape[1], k, k), np.float32)
    for dy in range(k):
        for dx in range(k):
            cols[:, :, :, :, dy, dx] = xp[:, :, dy:dy + 8, dx:dx + 8].transpose(0, 2, 3, 1)
    y = cols.reshape(n * 64, -1) @ w.reshape(w.shape[0], -1).T
    y = y.reshape(n, 8, 8, -1).transpose(0, 3, 1, 2)
    if bias is not None: y = y + bias[None, :, None, None]
    return y

def bn(x, t, name):
    g, b = t[name + ".weight"], t[name + ".bias"]; m, v = t[name + ".running_mean"], t[name + ".running_var"]
    return (x - m[None, :, None, None]) / np.sqrt(v[None, :, None, None] + EPS) * g[None, :, None, None] + b[None, :, None, None]

def ln(x, t, name):
    mu = x.mean(1, keepdims=True); var = x.var(1, keepdims=True)
    return (x - mu) / np.sqrt(var + EPS) * t[name + ".weight"][None, :, None, None] + t[name + ".bias"][None, :, None, None]

def forward(x, arch, t, record, *, md):
    """md: the file's safetensors __metadata__; `arch` must be dcm_arch.norm_arch_md(md)
    (dcm_arch.require_architecture_of). Every site this forward applies, and every
    block main path, is modelled as ReLU (dcm_arch.require_relu)."""
    dcm_arch.require_architecture_of(md, arch, "relu_inputs.forward")
    groups = arch["block_groups"]
    if not all(g["activation_style"] == "pre" and g["se_style"] == "none" and not g["use_rezero"]
               and g["skip_merge"] == "clean_add" and g.get("output_norm") == "layer_norm" for g in groups):
        raise dcm_arch.ArchitectureError("relu_inputs.forward: unsupported architecture")
    if arch["policy_head_style"] != "intermediate_conv" or arch["value_head_style"] != "wdl_softmax":
        raise dcm_arch.ArchitectureError("relu_inputs.forward: only an intermediate_conv policy and a wdl_softmax value head are modelled")
    if arch["feature_skip_source"] != "none":
        raise dcm_arch.ArchitectureError("relu_inputs.forward: a feature skip is not modelled")
    dcm_arch.require_relu(md, "relu_inputs.forward", ("tower_end_activation", "policy_head_activation",
                                                      "value_head_conv_activation", "value_head_fc1_hidden_activation"),
                          block_main_path=True)
    h = conv(x, t["stem.conv.weight"]); record("tap:stem_bn_input", h)
    h = bn(h, t, "stem.bn")
    i = 0
    for g in groups:
        for _ in range(g["count"]):
            p = f"blocks.{i}"
            a = bn(h, t, p + ".bn1"); record(f"relu:{p}.bn1", a)
            a = conv(np.maximum(a, 0), t[p + ".conv1.weight"])
            a = bn(a, t, p + ".bn2"); record(f"relu:{p}.bn2", a)
            a = conv(np.maximum(a, 0), t[p + ".conv2.weight"])
            h = ln(h + a, t, p + ".res_ln"); i += 1
    a = bn(h, t, "tower_final_bn"); record("relu:tower_final_bn", a)
    trunk = np.maximum(a, 0)
    pp = bn(conv(trunk, t["policy.pre_conv.weight"]), t, "policy.pre_bn"); record("relu:policy.pre_bn", pp)
    logits = conv(np.maximum(pp, 0), t["policy.conv.weight"], t["policy.conv.bias"]).reshape(x.shape[0], -1)
    vv = bn(conv(trunk, t["value.conv.weight"]), t, "value.bn"); record("relu:value.bn", vv)
    f1 = np.maximum(vv, 0).reshape(x.shape[0], -1) @ t["value.fc1.weight"].T + t["value.fc1.bias"]
    record("relu:value.fc1", f1[:, :, None, None])
    vl = np.maximum(f1, 0) @ t["value.wdl_fc2.weight"].T + t["value.wdl_fc2.bias"]
    return logits, vl

# ---------------------------------------------------------------- statistics
class ChannelStats:
    def __init__(self):
        self.n = 0; self.s = None
    def add(self, a):  # a [N,C,H,W]
        c = a.shape[1]; flat = a.transpose(1, 0, 2, 3).reshape(c, -1).astype(np.float64)
        if self.s is None:
            self.s = np.zeros(c); self.ss = np.zeros(c); self.mn = np.full(c, np.inf); self.mx = np.full(c, -np.inf); self.neg = np.zeros(c); self.pos = np.zeros(c)
        self.s += flat.sum(1); self.ss += (flat ** 2).sum(1); self.mn = np.minimum(self.mn, flat.min(1)); self.mx = np.maximum(self.mx, flat.max(1))
        self.neg += (flat < 0).sum(1); self.pos += (flat > 0).sum(1); self.n += flat.shape[1]
    def summary(self):
        mean = self.s / self.n; std = np.sqrt(np.maximum(self.ss / self.n - mean ** 2, 0))
        return dict(channels=len(mean), values_per_channel=int(self.n),
                    mean=mean.tolist(), std=std.tolist(), min=self.mn.tolist(), max=self.mx.tolist(),
                    frac_negative=(self.neg / self.n).tolist(), frac_positive=(self.pos / self.n).tolist(),
                    overall_mean=float(self.s.sum() / (self.n * len(mean))), overall_min=float(self.mn.min()), overall_max=float(self.mx.max()))

def run(label, path, pos):
    md, arch, t = load(path)
    stats = {}; maxabs = {}
    def record(name, a):
        if name.startswith("tap:"):
            maxabs[name[4:]] = max(maxabs.get(name[4:], 0.0), float(np.abs(a).max())); return
        stats.setdefault(name[5:], ChannelStats()).add(a)
    X = np.stack([p[0] for p in pos])
    logits_all = []; vlog_all = []
    for i in range(0, len(X), 64):
        lg, vl = forward(X[i:i + 64], arch, t, record, md=md)
        logits_all.append(lg); vlog_all.append(vl)
    L = np.concatenate(logits_all); V = np.concatenate(vlog_all)
    legal_means = []; legal_spreads = []; legal_min = []; legal_max = []; all_means = []
    legal_vals = []
    for row, (_, idx, *_) in enumerate(pos):
        lv = L[row, idx]; legal_vals.append(lv)
        legal_means.append(lv.mean()); legal_spreads.append(lv.std()); legal_min.append(lv.min()); legal_max.append(lv.max())
        all_means.append(L[row].mean())
    illegal_mask = np.ones_like(L, bool)
    for row, (_, idx, *_) in enumerate(pos): illegal_mask[row, idx] = False
    lv_cat = np.concatenate(legal_vals); il = L[illegal_mask]
    pct = lambda a: [float(np.percentile(a, q)) for q in (1, 25, 50, 75, 99)]
    ex = V - V.max(1, keepdims=True); P = np.exp(ex) / np.exp(ex).sum(1, keepdims=True)
    shared = V.mean(1)
    targets = [p[2] if len(p) > 2 else None for p in pos]
    tgt_rows = [(i, t) for i, t in enumerate(targets) if t is not None]
    value_ce = float(np.mean([-np.log(max(P[i, t], 1e-12)) for i, t in tgt_rows])) if tgt_rows else None
    base = np.bincount([t for _, t in tgt_rows], minlength=3) / len(tgt_rows) if tgt_rows else None
    base_ce = float(-sum(b * np.log(b) for b in base if b > 0)) if tgt_rows else None
    out = dict(label=label, model_id=md["model_id"], training_step=md.get("training_step"), positions=len(pos),
               relu_inputs={k: s.summary() for k, s in stats.items()},
               tap_maxabs=maxabs,
               policy=dict(all_min=float(L.min()), all_max=float(L.max()), all_mean=float(L.mean()),
                           legal_min=float(lv_cat.min()), legal_max=float(lv_cat.max()), legal_mean=float(lv_cat.mean()),
                           illegal_min=float(il.min()), illegal_max=float(il.max()), illegal_mean=float(il.mean()),
                           per_position_legal_mean_pct=pct(legal_means), per_position_legal_spread_median=float(np.median(legal_spreads)),
                           per_position_legal_max_pct=pct(legal_max), per_position_legal_min_pct=pct(legal_min),
                           per_position_all_mean_median=float(np.median(all_means)),
                           legal_top_minus_illegal_max_pct=pct([L[r, idx].max() - L[r][illegal_mask[r]].max() for r, (_, idx, *_) in enumerate(pos)]),
                           maxabs=float(np.abs(L).max()),
                           legal_top_minus_illegal_max_min=float(min(L[r, idx].max() - L[r][illegal_mask[r]].max() for r, (_, idx, *_) in enumerate(pos))),
                           illegal_softmax_mass_mean=float(np.mean([np.exp(L[r][illegal_mask[r]] - L[r].max()).sum() / np.exp(L[r] - L[r].max()).sum() for r in range(len(pos))])),
                           illegal_softmax_mass_max=float(np.max([np.exp(L[r][illegal_mask[r]] - L[r].max()).sum() / np.exp(L[r] - L[r].max()).sum() for r in range(len(pos))]))),
               value=dict(slot_min=V.min(0).tolist(), slot_max=V.max(0).tolist(), slot_mean=V.mean(0).tolist(),
                          all_min=float(V.min()), all_max=float(V.max()), all_mean=float(V.mean()),
                          shared_pct=pct(shared), start_wdl=(P[0].tolist() if os.environ.get("POSITIONS") != "corpus" else None),
                          value_ce=value_ce, base_rate_ce=base_ce, target_rates=(base.tolist() if base is not None else None), prob_mean=P.mean(0).tolist(), maxabs=float(np.abs(V).max())))
    return out

if __name__ == "__main__":
    if os.environ.get("POSITIONS") == "corpus":
        pos, games = corpus_positions(shards=[20, 26, 32, 38], games_per_shard=1024, plies_per_game=2, seed=20261004)
    else:
        pos, games = positions()
    print(f"positions {len(pos)} from {games} games", file=sys.stderr)
    results = [run(*a.split("=", 1), pos) for a in sys.argv[2:]]
    json.dump(dict(positions=len(pos), games=games, results=results), open(sys.argv[1], "w"))
    for r in results:
        print(r["label"], "start WDL", (None if r["value"]["start_wdl"] is None else [round(x, 4) for x in r["value"]["start_wdl"]]), "legal-mean pct", [round(x, 3) for x in r["policy"]["per_position_legal_mean_pct"]],
              "spread med", round(r["policy"]["per_position_legal_spread_median"], 4), "all-mean med", round(r["policy"]["per_position_all_mean_median"], 4),
              "shared pct", [round(x, 3) for x in r["value"]["shared_pct"]], "tap", {k: round(v, 3) for k, v in r["tap_maxabs"].items()}, "plog maxabs", round(r["policy"]["maxabs"], 3), "vlog maxabs", round(r["value"]["maxabs"], 3))
