from itertools import product as pp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
 
from hearsay import hearsay

# Figura 5

# (a) Variation of tau_survive for fixed tau_awakening
#-----------------------------------------------------

ta = [50000]
ts = [10000, 50000, 100000, 200000, 300000, 400000]
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
    fname.append(f"../out/F5a/{str(k).zfill(5)}_001.pk")

df = pd.DataFrame(list(zip(tau_a, tau_s, d_max, fname)), 
               columns =['tau_awakening', 'tau_survive', 'D_max',
               'filename'])

df.to_csv('F5a.csv')

conf = hearsay.Parser('F5a.ini')
conf.load_config()
G = hearsay.C3Net(conf)
G.set_parameters(df)
G.run()

R = hearsay.Results(G)
R.load()
res = R.redux()
FirstContactTimes = res['lF']
 
fig = plt.figure()
ax = fig.add_subplot()
for k, c1 in enumerate(FirstContactTimes):
    if len(c1)==0:
        continue
    imax = max(c1)
    imin = min(c1)
    if imax < imin+1.e-4:
        continue
    breaks = np.linspace(imin, imax, 30)
    hy, hx = np.histogram(c1, breaks, density=True)

    hx = (breaks[:-1] + breaks[1:])/2

    lbl = (f"A={R.params.iloc[k]['tau_awakening']},"
           f"S={R.params.iloc[k]['tau_survive']}")
    ax.step(hx, hy, label=lbl)

ax.set_yscale('log')
ax.legend()
ax.set_xlim(0, 7.e5)
fig.savefig('F5a.png')
