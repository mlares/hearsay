from hearsay import hearsay
import pickle
import pandas as pd


####################################################
# Figura 3
####################################################

df = pd.read_csv('F3.csv')
config = hearsay.Parser('F3.ini')
config.load_config()
G = hearsay.C3Net(config)
G.set_parameters(df)

R = hearsay.Results(G)
R.load()

m1, m2 = R.redux_2d()

fn = R.config.filenames
fname = fn.dir_output + fn.exp_id
fname1 = fname + '/m1.pk'
fname2 = fname + '/m2.pk'

with open(fname1, 'w') as pickle_file:
    pickle.dump(m1, pickle_file)
with open(fname2, 'w') as pickle_file:
    pickle.dump(m2, pickle_file)
