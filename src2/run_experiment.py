import ccn

# In this script we make many simulations of a galaxy filled with MPLs
# and the correspnding network of contacts.

import configparser
simu_parameters = configparser.ConfigParser()
simu_parameters.read('../set/config.ini')
 

G = ccn.Galaxy()

G.set_parameters(simu_parameters)

G.run_experiment()

