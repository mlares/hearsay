from hearsay import hearsay
from sys import argv

if len(argv) > 1:
    conf = hearsay.Parser(argv[1])
else:
    conf = hearsay.Parser()

G = hearsay.C3Net(conf)
G.set_parameters()
G.run()
