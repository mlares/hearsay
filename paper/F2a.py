import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from itertools import product as pp
from hearsay import hearsay

# Figura 2a

# (a) Variation of tau_survive for fixed tau_awakening
# -----------------------------------------------------


# 1) Generate points in the paramter space to sample :::::::::::::::::::

ta = [100, 500, 1000, 2000, 4000, 5000, 10000, 50000]
ts = [10000]
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
    fname.append(f"../out/F2a/{str(k).zfill(5)}_001.pk")

df = pd.DataFrame(list(zip(tau_a, tau_s, d_max, fname)),
                  columns=['tau_awakening', 'tau_survive', 'D_max',
                  'filename'])

df.to_csv('F2a.csv')


# 1) Correr las simulaciones :::::::::::::::::::

# ....(a)
df = pd.read_csv('F2a.csv')
config = hearsay.Parser('F2a.ini')
config.load_config()
G = hearsay.C3Net(config)
G.set_parameters(df)

G.run()

# 2) Leer las simulaciones :::::::::::::::::::::

# ....(a)
dfa = pd.read_csv('F2a.csv')
config = hearsay.Parser('F2a.ini')
config.load_config()
G = hearsay.C3Net(config)
G.set_parameters(dfa)

R = hearsay.Results(G)
R.load()
res = R.redux()
ib = res['lI']

fig = plt.figure()
ax = fig.add_subplot()
for k, inbox in enumerate(ib):
    imax = max(inbox)
    breaks = np.array(range(imax+1)) + 0.5
    hy, hx = np.histogram(inbox, breaks, density=True)

    xx = breaks[:-1] + 0.5
    yy = np.cumsum(hy)

    lbl = (f"A={R.params.iloc[k]['tau_awakening']}, "
           f"D={R.params.iloc[k]['D_max']}")
    ax.step(xx, yy, label=lbl)

ax.set_xscale('log')
ax.legend()
fig.savefig('F2a.png')
