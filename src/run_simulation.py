import hearsay
from sys import argv

conf = hearsay.parser(argv, nran=100)

G = hearsay.GalacticNetwork(conf)

tau_awakening = 20000
tau_survive = 20000
D_max = 20000
pars = [tau_awakening, tau_survive, D_max]

G.run_experiment(spars=pars)


#G.run_simulation(conf.p, pars)
#MPL1 = G.MPL


#G.run_simulation_II(conf.p, pars, 10, 1)

