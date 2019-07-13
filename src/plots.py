 # coding: utf-8

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import seaborn as sns

def redux(D):

    index = []
    firstc = []
    ncetis = []
    awaken = []     # lapso de tiempo que esta activa
    waiting = []    # lapso de tiempo que espera hasta el primer contacto
    inbox = []      # cantidad de cetis que esta escuchando
    distancias = [] # distancias a las cetis contactadas
    hangon = []     # lapso de tiempo que esta escuchando otra CETI
    N = len(D)
    kcross = 0
    
    for filename in D['name']:        

        CETIs = pickle.load( open(filename, "rb") )

        M = len(CETIs)
        ncetis.append(M)
    
        for i in range(M): # number of realizations
    
            k = len(CETIs[i]) # number of contacts
            inbox.append(k)
            awaken.append(CETIs[i][0][5] - CETIs[i][0][4])
            index.append(kcross)

            firstcontact = 1.e8

            for l in range(1,k):
                earlier = CETIs[i][l][4] - CETIs[i][0][4]
                firstcontact = min(earlier, firstcontact)
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) - 
                    np.array(CETIs[i][l][2:4]))**2).sum())

                waiting.append(earlier)
                distancias.append(Dx)
                hangon.append(CETIs[i][l][5] - CETIs[i][l][4])
    
            if(k>0) firstc.append(firstcontact)

        kcross+=1
            
    N = 12
    count = [0]*N
    for i in range(N):
        count[i] = inbox.count(i)
 
    return(awaken, inbox, distancias, hangon, waiting, count, index,
            firstc, ncetis)

    # size = the number of cetis
    #awaken
    #inbox
    #index
    #firstc

    # size = max number of Ncetis (12)
    #count

    # size = Number of contacts
    # distancias
    # hangon
    # waiting

    # Number of experiments
    # ncetis


D = pandas.read_csv('../dat/SKRU_01/params.csv')

#awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis = redux(D)
#l = [index, awaken, inbox, firstc, distancias, hangon, waiting, count, ncetis]

l = list(redux(D))

lens = []
for i in l: 
    lens.append(len(i))


# MAKE SOME PLOTS ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# awaken: distribution

plt.hist(awaken, bins=50, color='teal')
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['font.size'] = 16

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.rcParams['figure.figsize'] = [56, 56]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
plt.tight_layout()

plt.savefig('../plt/SKRU_awaken.pdf', format='pdf') 



# distancias: distribution
     
plt.hist(distancias, bins=50, color='teal')
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 'small'
plt.rcParams['font.size'] = 16

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.rcParams['figure.figsize'] = [56, 56]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
plt.tight_layout()

plt.savefig('../plt/SKRU_distances.pdf', format='pdf') 


# inbox: distribution

plt.bar(range(N), count, color='teal')
 
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['font.size'] = 36

plt.rcParams['axes.xmargin'] = 0.05
plt.rcParams['axes.ymargin'] = 0.05
plt.rcParams['figure.figsize'] = [26, 26]
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'

plt.tight_layout()

plt.savefig('../plt/SKRU_inbox.pdf', format='pdf') 

            



## MAKE SOME PLOTS
#
#fig, axes = plt.subplots(2, 2, figsize=(8,8))
#
#axes[0,0].hist(awaken, bins=50, color='teal')
#axes[0,0].set_title('AWAKEN')
#axes[0,0].set_xlabel('time awaken [yr]')
#axes[0,0].set_ylabel('N')
#
#axes[0,1].hist(distancias, bins=20, color='teal')
#axes[0,1].set_title('DISTANCES')
#axes[0,1].set_xlabel('distance [lyr]')
#axes[0,1].set_ylabel('N')
# 
#mx = max(waiting)
#axes[1,0].hist(waiting, bins=np.linspace(0,mx,20), color='lightseagreen')
#axes[1,0].hist(hangon, bins=np.linspace(0,mx,20), color='teal', alpha=0.8)
#axes[1,0].set_title('WAITING + HANGON')
#axes[1,0].set_xlabel('time waiting first contact [yr]')
#axes[1,0].set_ylabel('N')
#
#N = 5
#count = [0]*N
#for i in range(N):
#   count[i] = inbox.count(i)
#
#axes[1,1].bar(range(N), count, color='teal')
#axes[1,1].set_title('Number of contacts')
#axes[1,1].set_xlabel('time hanging on [yr]')
#axes[1,1].set_ylabel('N')
# 
#plt.rcParams['legend.fontsize'] = 40
#plt.rcParams['lines.linewidth'] = 2
#plt.rcParams['axes.labelsize'] = 'large'
#plt.rcParams['font.size'] = 36
#
##plt.rcParams['axes.xmargin'] = 0.05
##plt.rcParams['axes.ymargin'] = 0.05
##plt.rcParams['figure.figsize'] = [26, 26]
##plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
### https://matplotlib.org/users/customizing.html for further customization
##
#plt.tight_layout()
#
#plt.savefig('../plt/plot.pdf', format='pdf')

 
##################################################################

l1 = (D['tau_awakening'] == 1000.)  & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.)
l2 = (D['tau_awakening'] == 10500.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.)
l3 = (D['tau_awakening'] == 20000.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.)
d1 = D[l1]
d2 = D[l2]
d3 = D[l3]
 
awaken1, inbox1, distancias1, hangon1, waiting1, count1, index1, firstc1, ncetis1 = redux(d1)
awaken2, inbox2, distancias2, hangon2, waiting2, count2, index2, firstc2, ncetis2 = redux(d2)
awaken3, inbox3, distancias3, hangon3, waiting3, count3, index3, firstc3, ncetis3 = redux(d3)


sns.distplot(awaken1, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=1000')
sns.distplot(awaken2, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=10500')
sns.distplot(awaken3, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=20000')
plt.show()

##################################################################

l1 = (D['tau_awakening'] == 1000.)  & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
l2 = (D['tau_awakening'] == 10500.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
l3 = (D['tau_awakening'] == 20000.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
d1 = D[l1]
d2 = D[l2]
d3 = D[l3]
 
awaken1, inbox1, distancias1, hangon1, waiting1, count1, index1, firstc1, ncetis1 = redux(d1)
awaken2, inbox2, distancias2, hangon2, waiting2, count2, index2, firstc2, ncetis2 = redux(d2)
awaken3, inbox3, distancias3, hangon3, waiting3, count3, index3, firstc3, ncetis3 = redux(d3)

sns.distplot(distancias1, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=1000')
sns.distplot(distancias2, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=10500')
sns.distplot(distancias3, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=20000')
plt.show()
 

##################################################################

l1 = (D['tau_awakening'] == 1000.)  & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
l2 = (D['tau_awakening'] == 10500.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
l3 = (D['tau_awakening'] == 20000.) & (D['tau_survive'] == 5050.) & (D['D_max'] == 10500.) 
d1 = D[l1]
d2 = D[l2]
d3 = D[l3]
 
awaken1, inbox1, distancias1, hangon1, waiting1, count1, index1, firstc1, ncetis1 = redux(d1)
awaken2, inbox2, distancias2, hangon2, waiting2, count2, index2, firstc2, ncetis2 = redux(d2)
awaken3, inbox3, distancias3, hangon3, waiting3, count3, index3, firstc3, ncetis3 = redux(d3)

sns.distplot(waiting1, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=1000')
sns.distplot(waiting2, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=10500')
sns.distplot(waiting3, hist = False, kde = True,
             kde_kws = {'shade': True, 'linewidth': 3},
             label='tau_a=20000')
plt.show()
 

# D['tau_survive'].unique()
# array([10000., 20000., 30000., 40000., 50000.])
# 
# D['tau_awakening'].unique()
# array([ 4800.,  9600., 14400., 19200., 24000.])
# 
# D['D_max'].unique()
# array([ 1000.,  5000., 10000., 20000., 40000.])
  

l = (D['tau_awakening']==19200.) & (D['tau_survive']==40000.) & (D['D_max']==10000.)
d = D[l]

awaken, inbox, distancias, hangon, waiting, count, index, firstc, ncetis = redux(d)








#probar plotnine

#https://medium.com/@gscheithauer/data-visualization-in-python-like-in-rs-ggplot2-bc62f8debbf5

#https://plotnine.readthedocs.io/en/stable/



# Analysis

#D.groupby('tau_awakening').mean()

#D['tau_awakening'].unique()

# https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html



df = pd.DataFrame({'distancias':distancias, 'hangon':hangon, 'waiting':waiting})
df = pd.DataFrame({'awaken':awaken, 'inbox':inbox, 'index':index, 'firstc':firstc})
df.groupby('index').max()

 
