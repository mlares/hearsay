from hearsay import hearsay
import pickle
import pandas as pd
from sys import argv

####################################################
# Figura 3
####################################################

if len(argv) > 1:
    conf = hearsay.Parser(argv[1])
else:
    conf = hearsay.Parser()

G = hearsay.C3Net(conf)
G.set_parameters() 


#df = pd.read_csv('F2a.csv')
#config = hearsay.Parser('F2a.ini')
#config.load_config()
#G = hearsay.C3Net(config)
#G.set_parameters(df)

R = hearsay.Results(G)
R.load()

m1, m2 = R.redux_2d()

fn = R.config.filenames
fname = fn.dir_output + fn.exp_id
fname1 = fname + '/m1.pk'
fname2 = fname + '/m2.pk'


f1 = open(fname1, 'wb')
pickle.dump(m1, f1)
f1.close()

f2 = open(fname2, 'wb')
pickle.dump(m2, f2)
f2.close()
