# %load_ext autoreload
# %autoreload 2

import hearsay
'''
python run_profile.py ../set/experiment.ini
'''

from sys import argv

conf = hearsay.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

G = hearsay.GalacticNetwork()

tau_awakening = 20000
tau_survive = 20000
D_max = 20000
pars = [tau_awakening, tau_survive, D_max]
G.run_simulation(conf.p, pars)

G.run_simulation_II(conf.p, pars, 10, 2)

G.show_single_ccns()


