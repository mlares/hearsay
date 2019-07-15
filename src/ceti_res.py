# coding: utf-8

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import seaborn as sns

#===============================================================================
def redux(D):

    index = []
    firstc = []
    ncetis = []
    awaken = []     # lapso de tiempo que esta activa
    waiting = []    # lapso de tiempo que espera hasta el primer contacto
    inbox = []      # cantidad de cetis que esta escuchando
    distancias = [] # distancias a las cetis contactadas
    hangon = []     # lapso de tiempo que esta escuchando otra CETI
    x = []
    y = []
    N = len(D)
    kcross = 0
    
    for filename in D['name']:        

        CETIs = pickle.load( open(filename, "rb") )

        M = len(CETIs)
        ncetis.append(M)
    
        for i in range(M): # experiments
    
            k = len(CETIs[i]) # CETIs resulting from the experiment
            inbox.append(k-1)
            awaken.append(CETIs[i][0][5] - CETIs[i][0][4])
            index.append(kcross)
            x.append(CETIs[i][0][2])
            y.append(CETIs[i][0][3])

            firstcontact = 1.e8

            for l in range(1,k):  # traverse contacts

                earlier = CETIs[i][l][4] - CETIs[i][0][4]
                firstcontact = min(earlier, firstcontact)
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) - 
                    np.array(CETIs[i][l][2:4]))**2).sum())

                waiting.append(earlier)
                distancias.append(Dx)
                hangon.append(CETIs[i][l][5] - CETIs[i][l][4])
    
            if(k>1): firstc.append(firstcontact)

        kcross+=1
            
    N = 12
    count = [0]*N
    for i in range(N):
        count[i] = inbox.count(i)
 
    return(awaken, inbox, distancias, hangon, waiting, count, index,
            firstc, ncetis, x, y)
#===============================================================================
    
D = pandas.read_csv('../dat/SKRU_01/params.csv')

#lens = []
#for i in list(redux(D)): 
#    lens.append(len(i))
 

# size = the number of cetis
#awaken # lifetime of a CETI
#inbox  # number of contacts in the life of a CETI
#index  # ID identifier

# size = the number of cetis that make contact
#firstc  # time elapsed between awakening and first contact

# size = range in the number of Ncetis (12, fixed)
#count # (mean?) number of contacts per ceti

# size = Number of contacts
# distancias
# hangon
# waiting

# Number of experiments
# ncetis



# D['tau_survive'].unique()
# array([10000., 20000., 30000., 40000., 50000.])
# D['tau_awakening'].unique()
# array([ 4800.,  9600., 14400., 19200., 24000.])
# D['D_max'].unique()
# array([ 1000.,  5000., 10000., 20000., 40000.])
  
#--------------------------------------------------------


# samples:
l1s = (D['tau_awakening']<15000) & (D['tau_survive']<15000) & (D['D_max']<=5000.)
l2s = (D['tau_awakening']>20000) & (D['tau_survive']>50000) & (D['D_max']<=5000.)
l3s = (D['tau_awakening']<5000) & (D['tau_survive']>15000) & (D['D_max']<=5000.)
l4s = (D['tau_awakening']>15000) & (D['tau_survive']<26000) & (D['D_max']<=5000.)
# 800, 1050, 1350, 1200

l1l = (D['tau_awakening']<15000) & (D['tau_survive']<15000) & (D['D_max']>=20000.)
l2l = (D['tau_awakening']>20000) & (D['tau_survive']>50000) & (D['D_max']>=20000.)
l3l = (D['tau_awakening']<5000) & (D['tau_survive']>15000) & (D['D_max']>=20000.)
l4l = (D['tau_awakening']>15000) & (D['tau_survive']<26000) & (D['D_max']>=20000.)
# 1600, 2100, 2721, 2400







# Distribucion del numero de contactos

bins = np.arange(-1,12)+0.5
A = D['tau_awakening'].unique()
S = D['tau_survive'].unique()
                                                      

d = D[l1s]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
plt.hist(inbox, bins=bins, histtype='step', align='mid', color='teal',
        label='l1s', linewidth=2)

d = D[l2s]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
plt.hist(inbox, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l2s', linewidth=2)

d = D[l3s]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
plt.hist(inbox, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l3s', linewidth=2)

d = D[l4s]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
plt.hist(inbox, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l4s', linewidth=2)

plt.xticks(np.arange(0, 12, 2))
plt.legend()
plt.show()






d = D[l1l]
awaken, inbox1, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)
d = D[l2l]
awaken, inbox2, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)
d = D[l3l]
awaken, inbox3, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
d = D[l4l]
awaken, inbox4, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 

plt.hist(inbox1, bins=bins, histtype='step', align='mid', color='teal',
        label='l1s', linewidth=2, density=True)
plt.hist(inbox2, bins=bins, histtype='step', align='mid',color='slateblue',
        label='l2s', linewidth=2, density=True)
plt.hist(inbox3, bins=bins, histtype='step', align='mid',color='firebrick',
        label='l3s', linewidth=2, density=True)
plt.hist(inbox4, bins=bins, histtype='step', align='mid',color='tomato',
        label='l4s', linewidth=2, density=True)

plt.xticks(np.arange(0, 12, 2))
plt.yscale('log')
plt.legend()
plt.show()
          

PROBAR CON LA ACUMULADA (R:ECDF)












bins = np.arange(0, 150000, 5000)

l = (D['tau_awakening']==A[0]) & (D['tau_survive']==S[10]) & (D['D_max']<=5000.)
d = D[l]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 
plt.hist(firstc, bins=bins, histtype='step', align='mid', color='teal',
        label='Dmax=40000', linewidth=2)

l = (D['tau_awakening']==A[0]) & (D['tau_survive']==S[10]) & (D['D_max']<=20000.)
d = D[l]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d) 

plt.hist(firstc, bins=bins, histtype='step', align='mid',color='crimson',
        label='Dmax=10000', linewidth=2)

#plt.xticks(np.arange(0, 12, 2))
plt.legend()
plt.yscale('log')
plt.show()
 











import matplotlib.cm as cm

### RATE OF NO CONTACT VS. TAU_A AND TAU_S

A = D['tau_awakening'].unique()
S = D['tau_survive'].unique()
N1 = len(A)
N2 = len(S)
m=np.zeros((N1,N2))
           
for i, a in enumerate(A):
    for j, s in enumerate(S):
        l = (D['tau_awakening']==a) & (D['tau_survive']==s) & (D['D_max']==10000.)
        d = D[l]
        awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)
        m[i][j] = inbox.count(0)/len(inbox)

plt.imshow(m, interpolation='nearest',
        cmap=cm.viridis)

plt.xticks(range(N1), [str(int(a)) for a in A])
plt.yticks(range(N2), [str(int(a)) for a in S])
plt.title('rate of no contact')
plt.xlabel('awakening rate')
plt.ylabel('survival rate')
plt.colorbar(orientation='vertical')
plt.show()


### RATE OF FIRST CONTACT AT AWAKENING VS. TAU_A AND TAU_S
# de todas las cetis que hicieron contacto alguna vez, 
# cuales lo hicieron en el awakening

A = D['tau_awakening'].unique()
S = D['tau_survive'].unique()
N1 = len(A)
N2 = len(S)
m=np.zeros((N1,N2))
           
for i, a in enumerate(A):
    for j, s in enumerate(S):
        l = (D['tau_awakening']==a) & (D['tau_survive']==s) & (D['D_max']==10000.)
        d = D[l]
        awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)
        m[i][j] = firstc.count(0.)/max(len(firstc),1)

plt.imshow(m, interpolation='nearest',
        cmap=cm.viridis)

plt.xticks(range(N1), [str(int(a)) for a in A])
plt.yticks(range(N2), [str(int(a)) for a in S])
plt.title('rate of contact at awakening')
plt.xlabel('awakening rate')
plt.ylabel('survival rate')
plt.colorbar(orientation='vertical')
plt.show()

 
######################################################

l = (D['tau_awakening']< 20000) &  (D['tau_survive']> 20000) & (D['D_max']==10000.)
d = D[l]
awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis, x, y = redux(d)

inbox = np.array(inbox)

l0 = inbox == 0
l1 = inbox == 1
l2 = inbox == 2
l3 = inbox == 3

x = np.array(x)
y = np.array(y)

plt.scatter(x[l1],y[l1],color='slateblue', s=0.1)
plt.scatter(x[l2],y[l2],color='crimson', alpha=0.6, s=0.4)
plt.scatter(x[l3],y[l3],color='forestgreen',alpha=0.2, s=1)
plt.show()




dist0 = np.sqrt(x[l0]**2 + y[l0]**2)
dist1 = np.sqrt(x[l1]**2 + y[l1]**2)
dist2 = np.sqrt(x[l2]**2 + y[l2]**2)
dist3 = np.sqrt(x[l3]**2 + y[l3]**2)


plt.hist(dist0, bins=30, histtype='step', density=True, align='mid',color='slategrey',linewidth=3, label='all')
plt.hist(dist1, bins=30, histtype='step', density=True, align='mid',color='slateblue', label='1')
plt.hist(dist2, bins=30, histtype='step', density=True, align='mid',color='crimson', label='2')
plt.hist(dist3, bins=30, histtype='step', density=True, align='mid',color='forestgreen', label='3')
plt.title('radial distribution of cetis with contact')
plt.xlabel('gactocentric distance (lyr)')
plt.ylabel('PDE')
plt.legend(loc=2)
plt.show()







