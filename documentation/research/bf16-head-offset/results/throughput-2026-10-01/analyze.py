import re,sys
def hms(t): h,m,s=t.split(':'); return int(h)*3600+int(m)*60+float(s)
def replay(log):
    pts=[]
    for line in open(log,errors='replace'):
        m=re.match(r'(\d+:\d+:[\d.]+)\s+\[REPLAY\] step=(\d+)',line)
        if m: pts.append((int(m.group(2)),hms(m.group(1))))
    a=[p for p in pts if p[0]>=200]
    return (a[-1][1]-a[0][1])/(a[-1][0]-a[0][0])*1000
def selfplay(log):
    pts=[]
    for line in open(log,errors='replace'):
        if '[STATS] elapsed=' not in line: continue
        e=re.search(r'elapsed=(\d+):(\d+):(\d+)',line); mv=re.search(r'spMoves=(\d+)',line); st=re.search(r'steps=(\d+)',line)
        if e and mv: pts.append((int(e.group(1))*3600+int(e.group(2))*60+int(e.group(3)),int(mv.group(1)),int(st.group(1)) if st else 0))
    a=[p for p in pts if p[0]>=120]
    if len(a)<2: return None
    dt=a[-1][0]-a[0][0]
    return (a[-1][1]-a[0][1])/dt*3600, (a[-1][2]-a[0][2])/dt, dt
for line in open(sys.argv[1]):
    tag=line.split()[0]; log=line.split('log=')[1].strip()
    if tag.startswith('sp'): print(tag, 'plies/hour %.0f, train steps/s %.3f over %ds'%selfplay(log))
    else: print(tag, 'ms/step %.1f'%replay(log))
