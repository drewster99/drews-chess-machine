import re, sys, statistics as st
def hms(t): h, m, s = t.split(':'); return int(h) * 3600 + int(m) * 60 + float(s)
def replay(log):
    pts = []
    for line in open(log, errors='replace'):
        m = re.match(r'(\d+:\d+:[\d.]+)\s+\[REPLAY\] step=(\d+) .*? ms=([\d.]+)', line)
        if m: pts.append((int(m.group(2)), hms(m.group(1)), float(m.group(3))))
    a = [p for p in pts if 200 <= p[0] <= 550]
    t = a[-1][1] - a[0][1]
    if t < 0: t += 86400
    return t / (a[-1][0] - a[0][0]) * 1000, st.median(p[2] for p in a)
runs = {}
for line in open(sys.argv[1]):
    if 'log=' not in line: continue
    tag = line.split()[0]; rc = re.search(r'rc=(\d+)', line).group(1)
    if rc != '0': print(tag, 'FAILED rc', rc); continue
    runs[tag] = replay(line.split('log=')[1].strip())
    print(tag, 'wall ms/step %.1f  sampled step ms median %.1f' % runs[tag])
for net in ('r7', 'se', 'v4'):
    arms = {a: runs.get(f'{net}-{a}') for a in ('fp32A', 'mixedA', 'mixedB', 'fp32B')}
    if None in arms.values(): continue
    f = (arms['fp32A'][0] + arms['fp32B'][0]) / 2; m = (arms['mixedA'][0] + arms['mixedB'][0]) / 2
    print(f'{net}: fp32 mean {f:.1f}  mixed mean {m:.1f}  change {m - f:+.1f} ms ({100 * (m - f) / f:+.1f}%)')
