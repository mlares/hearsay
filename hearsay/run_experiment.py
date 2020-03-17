# %load_ext autoreload
# %autoreload 2

import ccn

 

# python run_profile.py ../set/config_small.ini

import numpy as np
import time
import sys
from configparser import ConfigParser

#_________________________________________
# Parse parameters from configuration file

filename = ccn.check_file(['','../set/experiment.ini'])
#filename = ccn.check_file(sys.argv)
config = ConfigParser()
config.read(filename)

G = ccn.GalacticNetwork()
G.set_parameters(config._sections)

G.run_simulation()
