import hearsay
from sys import argv

conf = hearsay.parser()
conf.check_file(argv)
conf.read_config_file()
conf.load_filenames()
conf.load_parameters()

G = hearsay.GalacticNetwork(conf)

G.run_experiment()
