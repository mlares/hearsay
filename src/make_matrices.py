# coding: utf-8

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import seaborn as sns
import sys


from mpl_toolkits.axes_grid1 import make_axes_locatable
from ceti_exp import redux, reddux

D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.csv')



### MATRIX1: RATE OF NO CONTACT VS. TAU_A AND TAU_S

A = D['tau_awakening'].unique()
A.sort()
N1 = len(A)

S = D['tau_survive'].unique()
S.sort()
N2 = len(S)

m1_d1=np.zeros((N1,N2))
m1_d2=np.zeros((N1,N2))
m1_d3=np.zeros((N1,N2))
m1_d4=np.zeros((N1,N2))

m2_d1=np.zeros((N1,N2))
m2_d2=np.zeros((N1,N2))
m2_d3=np.zeros((N1,N2))
m2_d4=np.zeros((N1,N2))

l0_d1 = D['D_max']==1000.
l0_d2 = D['D_max']==10000.
l0_d3 = D['D_max']==40000.
l0_d4 = D['D_max']==80000.

toolbar_width = 40

for i, a in enumerate(A):
    print("%2.2d/%2.2d" % (i, N1))
    l1 = D['tau_awakening']==a


    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1))
    for j, s in enumerate(S):
        sys.stdout.write("-")
        sys.stdout.flush()

        l2 = D['tau_survive']==s

        l = l0_d1 & l1 & l2
        D_d1 = D[l]

        l = l0_d2 & l1 & l2
        D_d2 = D[l]

        l = l0_d3 & l1 & l2
        D_d3 = D[l]

        l = l0_d4 & l1 & l2
        D_d4 = D[l]


        if len(D_d1)>0:
            awaken, inbox, distancias, hangon, waiting, count, index,\
            firstc, ncetis, x, y = redux(D_d1)
            m1_d1[i][j] = inbox.count(0)/max(len(inbox), 1)
            m2_d1[i][j] = firstc.count(0.)/max(len(firstc),1)
        else:
            m1_d1[i][j] = 0.
            m2_d1[i][j] = 0.
 

        if len(D_d2)>0:
            awaken, inbox, distancias, hangon, waiting, count, index,\
            firstc, ncetis, x, y = redux(D_d2)
            m1_d2[i][j] = inbox.count(0)/max(len(inbox), 1)
            m2_d2[i][j] = firstc.count(0.)/max(len(firstc),1)
        else:
            m1_d2[i][j] = 0.
            m2_d2[i][j] = 0.      

 
        if len(D_d3)>0:
            awaken, inbox, distancias, hangon, waiting, count, index,\
            firstc, ncetis, x, y = redux(D_d3)
            m1_d3[i][j] = inbox.count(0)/max(len(inbox), 1)
            m2_d3[i][j] = firstc.count(0.)/max(len(firstc),1)
        else:
            m1_d3[i][j] = 0.
            m2_d3[i][j] = 0.      
 

        if len(D_d4)>0:
            awaken, inbox, distancias, hangon, waiting, count, index,\
            firstc, ncetis, x, y = redux(D_d4)
            m1_d4[i][j] = inbox.count(0)/max(len(inbox), 1)
            m2_d4[i][j] = firstc.count(0.)/max(len(firstc),1)
        else:
            m1_d4[i][j] = 0.
            m2_d4[i][j] = 0.      

    sys.stdout.write("]\n") # this ends the progress bar
 
# end: for i, a in enumerate(A)



m1_d1 = np.transpose(m1_d1)
m2_d1 = np.transpose(m2_d1)

m1_d2 = np.transpose(m1_d2)
m2_d2 = np.transpose(m2_d2)

m1_d3 = np.transpose(m1_d3)
m2_d3 = np.transpose(m2_d3)

m1_d4 = np.transpose(m1_d4)
m2_d4 = np.transpose(m2_d4)


pickle.dump( m1_d1, open('../dat/SKRU_07/matrix1_d1_SKRU_07.pkl', 'wb'))
pickle.dump( m1_d2, open('../dat/SKRU_07/matrix1_d2_SKRU_07.pkl', 'wb'))
pickle.dump( m1_d3, open('../dat/SKRU_07/matrix1_d3_SKRU_07.pkl', 'wb'))
pickle.dump( m1_d4, open('../dat/SKRU_07/matrix1_d4_SKRU_07.pkl', 'wb'))

pickle.dump( m2_d1, open('../dat/SKRU_07/matrix2_d1_SKRU_07.pkl', 'wb'))
pickle.dump( m2_d2, open('../dat/SKRU_07/matrix2_d2_SKRU_07.pkl', 'wb'))
pickle.dump( m2_d3, open('../dat/SKRU_07/matrix2_d3_SKRU_07.pkl', 'wb'))
pickle.dump( m2_d4, open('../dat/SKRU_07/matrix2_d4_SKRU_07.pkl', 'wb'))













for k in range(5):
    print(k)


    for i in range(toolbar_width):
        time.sleep(0.1) # do real work here
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()







la = D['tau_awakening'].isin([108000])
ls = D['tau_survive'].isin([340000])
ld = D['D_max'].isin([80000])
l = la & ls & ld
d = D[l]

for i in range(len(d)):
    print(i)
    #r = reddux(d[i])
