# %load_ext autoreload
# %autoreload 2

import ccn

# python run_profile.py ../set/experiment.ini

import numpy as np
import time
import sys
from sys import argv


conf = ccn.parser()
conf.check_file(argv)
conf.read_config_file()
#conf.load_filenames()
conf.load_parameters()

print(conf.p.tau_a_nbins)

G = ccn.GalacticNetwork()

G.run_experiment(conf.p)


# tau_awakening = 20000
# tau_survive = 20000
# D_max = 20000
# pars = [tau_awakening, tau_survive, D_max]
# G.run_simulation(conf.p, pars)


