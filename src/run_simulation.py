# Formas de cargar el modulo interactivo:
# En la terminal de ipython poner:
# %load_ext autoreload
# %autoreload 2

from hearsay import hearsay
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# TUTORIAL 1: experiment from ini file

conf = hearsay.Parser('experiment.ini')
G = hearsay.C3Net(conf)
G.set_parameters()
net = G.run(interactive=True)
R = hearsay.Results(G)
R.load()
res = R.redux_1d()
plt.hist(res['A'])
plt.show()


# TUTORIAL 2: CORRER UNA SIMULACION

conf = hearsay.Parser()
G = hearsay.C3Net(conf)
conf.load_config(['nran'], ['7'])
tau_awakening = 20000
tau_survive = 20000
D_max = 20000
directory = ''.join([G.config.filenames.dir_output, G.config.filenames.exp_id])
filename = ''.join([directory, 'test.pk'])
pars = [[tau_awakening, tau_survive, D_max, filename]]
df = pd.DataFrame(pars, columns=['tau_awakening', 'tau_survive',
                                 'D_max', 'filename'])
G.set_parameters(df)
res = G.run(interactive=True)
G.show_single_ccns(res[0])





# TUTORIAL 3: experiment from ini file masked

conf = hearsay.Parser()
conf.load_config()
G = hearsay.C3Net(conf)
G.set_parameters()

c1 = np.array((G.params['tau_awakening']-2000.) < 1.e-6)
c2 = np.array((G.params['tau_survive']-2330.) < 1.e-6)
c3 = np.array((G.params['D_max']-20000.) < 1.e-6)
c4 = np.array([s[-6:-3] in '001002003' for s in G.params['filename']])

cond = c1 & c2 & c3 & c4

subset = G.params[cond]

G.set_parameters(subset)

res = G.run(interactive=True)
G.show_single_ccns(res[0])


# TUTORIAL 4: experiment with custom parameters
# Ver el numero de contactos en funcion de A:

conf = hearsay.Parser()
conf.load_config(['nran'], ['3'])
G = hearsay.C3Net(conf)
G.set_parameters()
A = [20000, 30000, 40000]
S = [10000]
D = [10000]

s = []
for a in A:
    pars = [[a], S, D]
    G.set_parameters(pars)
    res = G.run(interactive=True)
    R = hearsay.Results(G)
    stat = R.redux_1d()
    s.append(stat)





# TUTORIAL 5: experiment with custom parameters

conf = hearsay.Parser()
conf.load_config(['nran'], ['1'])
G = hearsay.C3Net(conf)
G.set_parameters()
A = [20000, 30000, 40000]
G.set_parameters(A=A)
res = G.run(interactive=True)
G.show_single_ccns(res[0])


# TUTORIAL 5: experiment with custom parameters

conf = hearsay.Parser()
conf.load_config(['nran', 't_max'], ['1','1.e7'])
G = hearsay.C3Net(conf)
G.set_parameters()

A = [5000, 10000, 20000]
S = [20000]
D = [20000]
G.set_parameters(A=A, S=S, D=D)
net = G.run(interactive=True)
R = hearsay.Results(G)
R.load()

res = R.redux()

res1 = R.redux([True, False, False])
res2 = R.redux([False, True, False])
res3 = R.redux([False, False, True])






# wake up  despertarse (awaken) number if ccns
# hold on  mantener
# wait in  esperar que algo llegue (w) # time elapsed from awakening to contact

# speak_up
# get in
# call_in
# cut_out
# hang out: to spend a lot of time in a particular place, or to spend a lot of time with someone 
# call out: to say something in a loud voice, especially in order to get someone's attention
# 
# awaken  = wake_up  : times of awakenings
# waiting = wait_in  : waiting times from awakening to first contact
# hangon  = hang_on  : period of two way communication
# 
# firstc  = come_out : time of first contact (incoming)
# index   = keep_up  : index 
# inbox   = sign_up  : number of contacts per node
# ncetis  = count_up : number of nodes in a simulation
# 
#           distance
#           n_nodes
#           n_contacts
#           t_wake
#           t_first
#           dt_twoway
#           dt_wait



# class ccn():
#     """Class for causal contact nodes.
# 
#     methods:
#         init:
#             creates a node
#         __len__:
#             None
#         __repr__:
#             None
#         __str__:
#             None
#     """
# 
#     def __init__(self):
#         """Initialize.
# 
#         Args:
#             None
#         """
#         self.state = 'pre-awakening'
#         self.received = 0
#         self.delivered = 0
#         self.twoway = 0
#         self.t_awakening = 0.
#         self.t_doomsday = 0.
#         self.n_listening = 0.
#         self.n_listened = 0.
# 
#     def __len__(self):
#         """Get the number of contacts for this node.
# 
#         Args:
#             None
#         """
#         return self.received
# 
#     def __repr__(self):
#         """Representation for print.
# 
#         Args:
#             None
#         """
#         return 'Causal contact node in state {!s}, having\
#                 {!i} received signals and {!i} times listened'.format(
#             self.state, self.received, self.delivered)
# 
#     def __str__(self):
#         """Show the node as a string.
# 
#         Args:
#             None
#         """
#         return 'Causal contact node in state {!s}, having\
#                 {!i} received signals and {!i} times listened'.format(
#             self.state, self.received, self.delivered)
#  


# Verify lifetimes are exponential

conf = hearsay.Parser()
conf.load_config(['nran', 't_max'], ['500', '1.e7'])
G = hearsay.C3Net(conf)
G.set_parameters()

A = [500000]
S = [200000]
D = [40000]
G.set_parameters(A=A, S=S, D=D)
net = G.run(interactive=True)
R = hearsay.Results(G)
R.load()

res = R.redux()
x = res['lP'] 
flat_list = [item for sublist in x for item in sublist]

plt.hist(flat_list, bins=100)
plt.yscale('log')
plt.show()


# Analyze fraction of nodes that make contact as a function of tau_survive

conf = hearsay.Parser()
conf.load_config(['nran', 't_max'], ['10', '1.e7'])
G = hearsay.C3Net(conf)
G.set_parameters()

A = [50000]
S = np.linspace(1000, 1.e6, 10)
D = [40000]
G.set_parameters(A=A, S=S, D=D)
net = G.run(interactive=True)
R = hearsay.Results(G)
R.load()

frac = []
for s in S:

    sample = abs(R.params['tau_survive'] - s) < 1.e-3
    res = R.redux(sample)
    x = res['lI']
    x_flat = [item for sublist in x for item in sublist]
    fr = (len(x_flat) - x_flat.count(0)) / len(x_flat)
    frac.append(fr)

plt.scatter(S, np.array(frac))
plt.xlabel('tau_survive')
plt.ylabel('fraction of CCN that make contacts')
plt.show()


# Multiplicity of contacts

conf = hearsay.Parser()
conf.load_config(['nran', 't_max'], ['200', '1.e7'])
G = hearsay.C3Net(conf)
G.set_parameters()

A = [180000]
S = [400000]
D = [40000]
G.set_parameters(A=A, S=S, D=D)
net = G.run(interactive=True)
R = hearsay.Results(G)
R.load()
res = R.redux()
x = res['lI']
inbox = [item for sublist in x for item in sublist]
inbox = np.array(inbox)

imax = max(inbox)
breaks = np.array(range(imax+1)) + 0.5
hy, hx = np.histogram(inbox, breaks)

xx = breaks[:-1] + 0.5
yy = np.cumsum(hy)

plt.step(xx, yy)
plt.xscale('log')
plt.show()


# FIG. 6 :::::::::::::::::::
