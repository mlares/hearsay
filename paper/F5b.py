from itertools import product as pp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
 
from hearsay import hearsay

# Figura 5

# (b) Variation of tau_awakening for fixed tau_survive
#-----------------------------------------------------

ta = [5000, 10000, 30000, 50000, 70000, 100000, 500000]
ts = [100000]
td = [40000]


z = pp(ta, ts, td)

tau_a = []
tau_s = []
d_max = []
fname = []

for k, i in enumerate(z):
    tau_a.append(i[0])
    tau_s.append(i[1])
    d_max.append(i[2])
    fname.append(f"../out/F5b/{str(k).zfill(5)}_001.pk")

df = pd.DataFrame(list(zip(tau_a, tau_s, d_max, fname)), 
               columns =['tau_awakening', 'tau_survive', 'D_max',
               'filename'])

df.to_csv('F5b.csv')

conf = hearsay.Parser('F5b.ini')
conf.load_config()
G = hearsay.C3Net(conf)
G.set_parameters(df)
G.run()

R = hearsay.Results(G)
R.load()
res = R.redux()
FirstContactTimes = res['lH']
 
fig = plt.figure()
ax = fig.add_subplot()

mx = 2.e5
mn = 0

for k, c1 in enumerate(FirstContactTimes):
    if len(c1)==0:
        continue
    imax = max(c1)
    imin = min(c1)
    if imax < imin+1.e-4:
        continue
    breaks = np.linspace(mn, mx, 50)
    hy, hx = np.histogram(c1, breaks, density=True)

    hx = (breaks[:-1] + breaks[1:])/2

    lbl = (f"A={R.params.iloc[k]['tau_awakening']},"
           f"S={R.params.iloc[k]['tau_survive']}")
    ax.step(hx, hy, label=lbl)

ax.set_yscale('log')
ax.legend()
#ax.set_xlim(0, 1.e3)
fig.savefig('F5b.png')
