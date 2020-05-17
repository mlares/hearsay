from hearsay import hearsay
from sys import argv

if len(argv) > 1:
    conf = hearsay.parser(argv[1])
else:
    conf = hearsay.parser()

G = hearsay.GalacticNetwork(conf)

G.run_experiment(parallel=conf.p.run_parallel)
