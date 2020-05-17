import hearsay
from sys import argv

if len(argv) > 1:
    conf = hearsay.parser(argv[1])
else:
    conf = hearsay.parser()

R = hearsay.results(conf)
R.load()
res = R.redux_2d()