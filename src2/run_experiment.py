import ccn
import sys

inifile = str(sys.argv[1])

G = ccn.GalacticNetwork()

G.load_parameters('../set/experiment.ini')

G.run_simulation()

