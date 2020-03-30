# %load_ext autoreload
# %autoreload 2

import ccn

# python run_profile.py ../set/config_small.ini

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



#G.run_simulation(conf.p)


