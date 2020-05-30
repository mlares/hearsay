from itertools import product as pp
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from hearsay import hearsay

# Figura 5

# (a) Variation of tau_survive for fixed tau_awakening
# -----------------------------------------------------

ta = [10000]
ts = [5000, 10000, 20000, 50000, 500000]
td = [32615]

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
                  columns=['tau_awakening', 'tau_survive', 'D_max',
                  'filename'])

df.to_csv('F5a.csv')
df = pd.read_csv('F5a.csv')

conf = hearsay.Parser('F5a.ini')
conf.load_config()
G = hearsay.C3Net(conf)
G.set_parameters(df)

print('RUN simulation')
G.run()

print('REDUCE simulation')
R = hearsay.Results(G)
R.load()
res = R.redux()
FirstContactTimes = res['lF']
 
minval = 9999.
maxval = -9999.
for c1 in FirstContactTimes:
    imax = max(c1)
    imin = min(c1) 
    minval = min(minval, imin)
    maxval = max(maxval, imax)

fig = plt.figure()
ax = fig.add_subplot()
for k, c1 in enumerate(FirstContactTimes):
    if len(c1) == 0:
        continue
    imax = max(c1)
    imin = min(c1)
    if imax < imin+1.e-4:
        continue
    breaks = np.linspace(minval, maxval, 200)
    hy, hx = np.histogram(c1, breaks, density=False)

    lbl = (f"A={R.params.iloc[k]['tau_awakening']},"
           f"S={R.params.iloc[k]['tau_survive']}")
    hy = np.append(hy, hy[-1])
    ax.step(breaks, hy, where='post', label=lbl)

ax.set_yscale('log')
ax.set_xlim(0, 6.e5)
ax.legend()
fig.savefig('F5a.png')
fig.savefig('F5a.pdf')
plt.close()
