from hearsay import hearsay
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from itertools import product as pp
 
from hearsay import hearsay


#----   T a b l a     2   -------

# 1) Generate points in the paramter space to sample :::::::::::::::::::

ta = [1000, 10000]
ts = [10000, 50000]
td = [3261, 22830]

z = pp(ta, ts, td)

tau_a = []
tau_s = []
d_max = []
fname = []

for k, i in enumerate(z):
    tau_a.append(i[0])
    tau_s.append(i[1])
    d_max.append(i[2])
    fname.append(f"../out/T2/{str(k).zfill(5)}_001.pk")

df = pd.DataFrame(list(zip(tau_a, tau_s, d_max, fname)), 
               columns =['tau_awakening', 'tau_survive', 'D_max',
               'filename'])

df.to_csv('T2.csv')
 


# 2) Correr las simulaciones :::::::::::::::::::

df = pd.read_csv('T2.csv')
config = hearsay.Parser('T2.ini')
config.load_config()
G = hearsay.C3Net(config)
G.set_parameters(df)

G.run()
 

df = pd.read_csv('T2.csv')
config = hearsay.Parser('T2.ini')
config.load_config()
G = hearsay.C3Net(config)
G.set_parameters(df)

R = hearsay.Results(G)
R.load()
res = R.redux()


# aca calcular las cosas de la tabla

# Fraccion de nodos que hacen mas de X contactos

ib = res['lI']

for inbox in ib:
    inbox = np.array(inbox)
    f1 = sum(inbox>1)/len(inbox)
    f10 = sum(inbox>10)/len(inbox)
    print(f1, f10)

