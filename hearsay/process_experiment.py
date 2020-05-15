import hearsay
import pandas
from sys import argv

if len(argv) > 1:
    conf = hearsay.parser(argv[1])
else:
    conf = hearsay.parser()

R = hearsay.results(conf)

R.load()

res = R.redux2()



 
 
# # Mostrar una simulacion en particular:
# # R.show_ccns(2)
# 
# # VER: make_matrices y plots en plt
# 
