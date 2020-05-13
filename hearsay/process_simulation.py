import hearsay
import pandas
from sys import argv

conf = hearsay.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

G = hearsay.GalacticNetwork(conf)
R = hearsay.results(G)
R.load()
res = R.redux()

R.show_single_ccns()

# Mostrar una simulacion en particular:
# R.show_ccns(2)

# VER: make_matrices y plots en plt

