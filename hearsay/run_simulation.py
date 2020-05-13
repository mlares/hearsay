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

print(conf.p.tau_a_nbins)

G = hearsay.GalacticNetwork()

tau_awakening = 20000
tau_survive = 20000
D_max = 20000
pars = [tau_awakening, tau_survive, D_max]
G.run_simulation(conf.p, pars)
print(len(G.MPL))

hearsay.ShowCCNs(G.MPL)
