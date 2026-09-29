"""Position set for the Ejp0 step-681000 bf16 study.

- lichess: every position (both sides) of every Lichess bot game played by
  model 20260727-1-Ejp0 (identified from the game record's `generations`),
  labeled from the game result; our own moves carry the engine's recorded
  decision (W/D/L + top-5 policy) as ground truth for the real bf16 graph.
- corpus: shard 45 of corpus w3aA5b. The first 300 usable games with
  random.seed(1) reproduce the prior survey's 900 positions exactly
  (`c900`); sampling continues to 900 games for a larger set.
"""
import sys, os, json, glob, struct, random, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(HERE)
sys.path.insert(0, SP); sys.path.insert(0, os.path.join(SP, 'pylib'))
import chess, numpy as np
from fwd import game_states, encode

QD = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
KJ = [(-2,1),(-1,2),(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1)]
def pidx(mv, wtm):
    fr = 7-chess.square_rank(mv.from_square); fc = chess.square_file(mv.from_square)
    tr = 7-chess.square_rank(mv.to_square); tc = chess.square_file(mv.to_square)
    if not wtm: fr = 7-fr; tr = 7-tr
    dr = tr-fr; dc = tc-fc
    if mv.promotion:
        assert dr == -1
        d = {0:0,-1:1,1:2}[dc]
        ch = 73+d if mv.promotion == chess.QUEEN else 64+{chess.KNIGHT:0,chess.ROOK:1,chess.BISHOP:2}[mv.promotion]*3+d
        return ch*64+fr*8+fc
    if (dr,dc) in KJ: return (56+KJ.index((dr,dc)))*64+fr*8+fc
    dist = max(abs(dr),abs(dc)); sd = (0 if dr == 0 else dr//abs(dr), 0 if dc == 0 else dc//abs(dc))
    return (QD.index(sd)*7+dist-1)*64+fr*8+fc

MODEL_ID = '20260727-1-Ejp0'
P = []
G = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/LichessBot/Games')
ngames = 0
for f in sorted(glob.glob(G+'/**/*.json', recursive=True)):
    d = json.load(open(f))
    if not all(g['modelID'] == MODEL_ID and g['trainingStep'] == 681000 for g in d['generations']): continue
    ngames += 1
    ucis = [m['uciAsGiven'] for m in d['moves']]
    wscore = {'1-0':1,'0-1':-1,'1/2-1/2':0}[d['outcome']['pgnResult']]
    sts = game_states(ucis); b = chess.Board()
    for i, m in enumerate(d['moves']):
        st, rep, mask = sts[i]; assert m['ply'] == i
        wtm = b.turn == chess.WHITE; assert (st['stm'] == 'w') == wtm
        legal = list(b.legal_moves); li = [pidx(x, wtm) for x in legal]
        mv = chess.Move.from_uci(m['uciAsGiven']); assert mv in legal
        z = wscore if wtm else -wscore
        P.append(dict(src='lichess', game=d['gameID'], ply=i, fen=b.fen(), x=encode(st,rep,mask).astype(np.float32),
                      legal=np.array(li), legal_uci=[x.uci() for x in legal], target=li[legal.index(mv)],
                      label={1:0,0:1,-1:2}[z], ours=bool(m.get('ours') and m.get('decision')),
                      obs=(m['decision'] if (m.get('ours') and m.get('decision')) else None), c900=False))
        b.push(mv)
nl = len(P)
S = os.path.expanduser('~/Library/Application Support/DrewsChessMachine/Corpora/20260624-192615-w3aA5b/shard-00045.dcmgames')
data = open(S,'rb').read(120_000_000); off = 256; games = []
while len(games) < 900:
    ln = struct.unpack_from('<I', data, off)[0]; p = data[off+4:off+4+ln]; off += 8+ln
    flags, outc = p[0], p[1]; n = struct.unpack_from('<I', p, 3)[0]
    if flags & 1: continue
    mv = [struct.unpack_from('<H', p, 7+2*i)[0] for i in range(n)]
    def u(m):
        to = m & 63; fr = (m >> 6) & 63; pr = (m >> 12) & 7; s = lambda q: chr(97+q % 8)+str(8-q//8)
        return s(fr)+s(to)+('','n','b','r','q')[pr]
    games.append(([u(m) for m in mv], outc))
random.seed(1); bad = 0
for gi, (ucis, outc) in enumerate(games):
    if not ucis: continue
    b = chess.Board(); ok = True
    for uu in ucis:
        mv = chess.Move.from_uci(uu)
        if mv not in b.legal_moves: ok = False; break
        b.push(mv)
    if not ok: bad += 1; continue
    sts = game_states(ucis); b = chess.Board(); boards = []
    for uu in ucis: boards.append(b.copy()); b.push(chess.Move.from_uci(uu))
    for i in random.sample(range(len(ucis)), min(3, len(ucis))):
        st, rep, mask = sts[i]; bb = boards[i]; wtm = bb.turn == chess.WHITE
        legal = list(bb.legal_moves); li = [pidx(x, wtm) for x in legal]; mv = chess.Move.from_uci(ucis[i])
        z = 0 if outc == 1 else (1 if (outc == 0) == wtm else -1)
        P.append(dict(src='corpus', game=gi, ply=i, fen=bb.fen(), x=encode(st,rep,mask).astype(np.float32), legal=np.array(li),
                      legal_uci=[x.uci() for x in legal], target=li[legal.index(mv)], label={1:0,0:1,-1:2}[z],
                      ours=False, obs=None, c900=gi < 300))
print('ejp0 games', ngames, 'lichess positions', nl, 'of which ours', sum(p['ours'] for p in P),
      'corpus positions', len(P)-nl, 'c900', sum(p['c900'] for p in P), 'rejected', bad,
      'corpus label mix', np.bincount([p['label'] for p in P[nl:]], minlength=3))
pickle.dump(P, open(os.path.join(HERE, 'posset_ejp0.pkl'), 'wb'))
