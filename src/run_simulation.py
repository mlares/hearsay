# Formas de cargar el modulo interactivo:
# En la terminal de ipython poner:
# %load_ext autoreload
# %autoreload 2

import hearsay
import pandas as pd
# from sys import argv
          
# TUTORIAL 0: CORRER UNA SIMULACION

conf = hearsay.parser()
G = hearsay.GalacticNetwork(conf)
tau_awakening = 20000
tau_survive = 20000
D_max = 20000
name = 'file.dat'
pars = [[tau_awakening, tau_survive, D_max, name]]
df = pd.DataFrame(pars, columns=['tau_awakening', 'tau_survive',
                            'D_max', 'name'])
G.set_parameters(df)
res = G.run(interactive=True)
G.show_single_ccns(res[0])


# TUTORIAL 1: experiment from ini file

conf = hearsay.parser('experiment.ini')
G = hearsay.GalacticNetwork(conf)
G.set_parameters()
df, net = G.run(interactive=True)
R = hearsay.results(conf)
R.load()
res = R.redux_1d()
from matplotlib import pyplot as plt
plt.hist(res['A'])
plt.show()



# TUTORIAL 2: experiment from ini file masked

conf = hearsay.parser('experiment.ini')
G = hearsay.GalacticNetwork(conf)

subset = G.param_set['tau_awakening'] < 10000. 
pars_df = G.param_set.loc[subset]

df, net = G.run_suite(pars, interactive=True)   

#G.show_single_ccns(net[0])

R = hearsay.results(conf)
R.load()

# todos los parametros:
res = R.redux_1d()

plt.hist(res['A'])
plt.show()












              


# TUTORIAL 2: CORRER VARIAS REALIZACIONES DE UNA SIMULACION



# TUTORIAL 3: CORRER VARIAS SIMULACIONES PARA EXPLORAR UN PARÁMETRO

# G = hearsay.GalacticNetwork(conf)
# tau_awakening = 2000
# tau_survive = 20000
# D_max = 20000
# pars = [(tau_awakening, tau_survive, D_max), (tau_awakening, tau_survive, D_max)]
# 
# mpl = G.run_suite(pars, interactive=True)   
# 
 


# TUTORIAL 4: CORRER VARIAS SIMULACIONES PARA EXPLORAR DOS PARÁMETROS








# G.run_experiment(spars=pars)

# R = hearsay.results(conf)
# R.load()
# res = R.redux_1d()

#G.run_simulation(conf.p, pars)
#MPL1 = G.MPL


#G.run_simulation_II(conf.p, pars, 10, 1)


# 
# import hearsay
# 
# conf = hearsay.parser('../set/experiment.ini', 
#         keys=['t_max', 'nran'], values=['1.e6', '1'])
# 
# print(conf.p.nran)
# 
# G = hearsay.GalacticNetwork(conf)
# 
# tau_awakening = 2000
# tau_survive = 20000
# D_max = 20000
# pars = [(tau_awakening, tau_survive, D_max), (tau_awakening, tau_survive, D_max)]
# 
# mpl = G.run_suite(pars, interactive=True)
# 
# #mpl = G.run()
# 
# 

#G.run_experiment(spars=pars)
#
#R = hearsay.results(conf)
#R.load()
#res = R.redux_1d()
#
##G.run_simulation(conf.p, pars)
##MPL1 = G.MPL
##G.run_simulation_II(conf.p, pars, 10, 1)



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








