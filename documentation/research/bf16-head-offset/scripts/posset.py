import sys,os,json,glob,struct,random,pickle
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'pylib'))
import chess, numpy as np
from fwd import game_states, encode
QD=[(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
KJ=[(-2,1),(-1,2),(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1)]
def pidx(mv,white_to_move):
    fr=7-chess.square_rank(mv.from_square); fc=chess.square_file(mv.from_square)
    tr=7-chess.square_rank(mv.to_square); tc=chess.square_file(mv.to_square)
    if not white_to_move: fr=7-fr; tr=7-tr
    dr=tr-fr; dc=tc-fc
    if mv.promotion and mv.promotion!=chess.QUEEN or (mv.promotion==chess.QUEEN):
        pass
    if mv.promotion:
        assert dr==-1
        d={0:0,-1:1,1:2}[dc]
        ch = 73+d if mv.promotion==chess.QUEEN else 64+{chess.KNIGHT:0,chess.ROOK:1,chess.BISHOP:2}[mv.promotion]*3+d
        return ch*64+fr*8+fc
    if (dr,dc) in KJ: return (56+KJ.index((dr,dc)))*64+fr*8+fc
    dist=max(abs(dr),abs(dc)); sd=(0 if dr==0 else dr//abs(dr), 0 if dc==0 else dc//abs(dc))
    return (QD.index(sd)*7+dist-1)*64+fr*8+fc
P=[]  # dicts
# ---- Lichess bot games: every position, both sides ----
G=os.path.expanduser('~/Library/Application Support/DrewsChessMachine/LichessBot/Games')
for f in sorted(glob.glob(G+'/**/*.json',recursive=True)):
    if f.endswith('.journal.jsonl'): continue
    d=json.load(open(f)); ucis=[m['uciAsGiven'] for m in d['moves']]
    res=d['outcome']['pgnResult']; wscore={'1-0':1,'0-1':-1,'1/2-1/2':0}[res]
    sts=game_states(ucis); b=chess.Board()
    for i,m in enumerate(d['moves']):
        st,rep,mask=sts[i]; assert m['ply']==i, (m['ply'],i)
        wtm=b.turn==chess.WHITE; assert (st['stm']=='w')==wtm
        legal=list(b.legal_moves); li=[pidx(x,wtm) for x in legal]
        mv=chess.Move.from_uci(m['uciAsGiven']); assert mv in legal
        z=wscore if wtm else -wscore
        P.append(dict(src='lichess',game=d['gameID'],ply=i,x=encode(st,rep,mask).astype(np.float32),legal=np.array(li),legal_uci=[x.uci() for x in legal],
                      target=li[legal.index(mv)],label={1:0,0:1,-1:2}[z],ours=bool(m.get('ours')),obs=(m['decision'] if m.get('ours') else None)))
        b.push(mv)
nl=len(P)
# ---- corpus sample ----
S='/Users/andrew/Library/Application Support/DrewsChessMachine/Corpora/20260624-192615-w3aA5b/shard-00045.dcmgames'
data=open(S,'rb').read(40_000_000); off=256; games=[]
while len(games)<300:
    ln=struct.unpack_from('<I',data,off)[0]; p=data[off+4:off+4+ln]; off+=8+ln
    flags,outc=p[0],p[1]; n=struct.unpack_from('<I',p,3)[0]
    if flags&1: continue
    mv=[struct.unpack_from('<H',p,7+2*i)[0] for i in range(n)]
    def u(m):
        to=m&63; fr=(m>>6)&63; pr=(m>>12)&7; s=lambda q: chr(97+q%8)+str(8-q//8)
        return s(fr)+s(to)+('','n','b','r','q')[pr]
    games.append(([u(m) for m in mv],outc))
random.seed(1); bad=0
for gi,(ucis,outc) in enumerate(games):
    if not ucis: continue
    b=chess.Board(); ok=True
    for u in ucis:
        mv=chess.Move.from_uci(u)
        if mv not in b.legal_moves: ok=False; break
        b.push(mv)
    if not ok: bad+=1; continue
    sts=game_states(ucis); b=chess.Board(); boards=[]
    for u in ucis: boards.append(b.copy()); b.push(chess.Move.from_uci(u))
    for i in random.sample(range(len(ucis)),min(3,len(ucis))):
        st,rep,mask=sts[i]; bb=boards[i]; wtm=bb.turn==chess.WHITE
        legal=list(bb.legal_moves); li=[pidx(x,wtm) for x in legal]; mv=chess.Move.from_uci(ucis[i])
        z = 0 if outc==1 else (1 if (outc==0)==wtm else -1)
        P.append(dict(src='corpus',game=gi,ply=i,x=encode(st,rep,mask).astype(np.float32),legal=np.array(li),legal_uci=[x.uci() for x in legal],target=li[legal.index(mv)],label={1:0,0:1,-1:2}[z],ours=False,obs=None))
print('lichess positions',nl,'corpus positions',len(P)-nl,'corpus games rejected',bad, 'label mix corpus',np.bincount([p['label'] for p in P[nl:]],minlength=3))
pickle.dump(P,open('posset.pkl','wb'))
