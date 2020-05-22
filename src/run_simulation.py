from hearsay import hearsay
from sys import argv

conf = hearsay.parser(argv, nran=100)

G = hearsay.GalacticNetwork(conf)

tau_awakening = 20000
tau_survive = 20000
D_max = 20000
pars = [(tau_awakening, tau_survive, D_max)]

G.run_experiment_params(pars, [1])

# G.run_experiment(spars=pars)

# R = hearsay.results(conf)
# R.load()
# res = R.redux_1d()

#G.run_simulation(conf.p, pars)
#MPL1 = G.MPL


#G.run_simulation_II(conf.p, pars, 10, 1)



import hearsay

conf = hearsay.parser('../set/experiment.ini', 
        keys=['t_max', 'nran'], values=['1.e6', '1'])

print(conf.p.nran)

G = hearsay.GalacticNetwork(conf)

tau_awakening = 2000
tau_survive = 20000
D_max = 20000
pars = [(tau_awakening, tau_survive, D_max), (tau_awakening, tau_survive, D_max)]

mpl = G.run_suite(pars, interactive=True)

#mpl = G.run()



#G.run_experiment(spars=pars)
#
#R = hearsay.results(conf)
#R.load()
#res = R.redux_1d()
#
##G.run_simulation(conf.p, pars)
##MPL1 = G.MPL
##G.run_simulation_II(conf.p, pars, 10, 1)
