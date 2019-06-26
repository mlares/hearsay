# coding: utf-8
#
# key:  A: Awakening, B: Blackout, C: Contact, D: Doomsday
#
#
# CONTENTS OF THE <CETIS> PICKLE:
# . . . . . . . . . . . . . . . . . . . . . . . .
# 0. ID of emitting CETI
# 1. ID of emitting [receiving] CETI
# 2. x position in the galaxy
# 3. y position in the galaxy
# 4. time of Awakening [Contact]
# 5. time of Doomsday  [Blackout]
# . . . . . . . . . . . . . . . . . . . . . . . .
#
# This structure is always for the A-D cycle of each emitting CETI,
# and repeats for each contact (C-B).
#
# The time span of a CETI is t_D - t_A for CETIs[k][0]
# The time span of a contact is t_B - t_C for CETIs[k][i], with i>0
#
#
# age: time elapsed from A to a given time
# ceti_e: ceti emitting signals (emiter)
# ceti_r: ceti receiving signals (receiver)
# ceti_c: ceti that listen at least another ceti (citizen)
# ceti_h: ceti that is lestened by at least another ceti
#

# for Check purposes
#-------------------
# duration of a civilization (exponential distribution by costruction)
# time from the appeareance of a CETI to the next
#
# 
# derived quantities
#-------------------
# [time]
# time span of a ceti listening another
# time span of a ceti being listened by another
# duration of two-way communication channels
# waiting time until the first contact
# fraction of awaken time a ceti is listening at least another ceti
# age of contacted cetis_e at first C
# distribution of time to wait until next C
# fraction of cetis where the first contact is given at the A

# [numbers]
# distribution of the number of contacts for each ceti
# distribution of the maximum number of contacts for each ceti
# distribution of the number of contacts as a function of age
# number of contacts as a function of time in the galaxy
# rate of cetis that succeed in contact 
# fraction of contacts that admit a response
#
# [distances]
# distribution of distances between contacted cetis
#
# [correlations] relation between {RB}
# RB distance to ceti and time of double communication
# RB distance to ceti and age of contacted ceti
# RB the age of a ceti and the maximum number of contacted cetis before D
# RB lifespan of a ceti and max number of contacts
#
# distribution in the galaxy of cetis that reach contact

# ... everything as a function of parameters!


 

##########  PLOTS: dependencia de valores medios y parametros

# 

#----------------------------------------------------------------------

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas

D = pandas.read_csv('../dat/params_SKRU.csv')
N = len(D)

awaken = []     # lapso de tiempo que esta activa
waiting = []    # lapso de tiempo que espera hasta el primer contacto
inbox = []      # cantidad de cetis que esta escuchando
distancias = [] # distancias a las cetis contactadas
hangon = []     # lapso de tiempo que esta escuchando otra CETI

for experiment in range(N):

    filename = '../dat/CETIs_SKRU_' + str(experiment+1).zfill(7) + '.dat'
    CETIs = pickle.load( open(filename, "rb") )

    for i in range(len(CETIs)):

        k = len(CETIs[i])
        for l in range(1,k):
            earlier = CETIs[i][l][4] - CETIs[i][0][4]
            waiting.append(earlier)
            Dx = np.sqrt(((
                np.array(CETIs[i][0][2:4]) - 
                np.array(CETIs[i][l][2:4]))**2).sum())
            distancias.append(Dx)
            hangon.append(CETIs[i][l][5] - CETIs[i][l][4])

        inbox.append(k)
        awaken.append(CETIs[i][0][5] - CETIs[i][0][4])

N = 12
count = [0]*N
for i in range(N):
   count[i] = inbox.count(i)



# MAKE SOME PLOTS ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# awaken

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



# distancias
     
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


# inbox

N = 20
count = [0]*N
for i in range(N):
   count[i] = inbox.count(i)
                             
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

            

l1 = D['tau_awakening'] == 1000.
l2 = D['tau_awakening'] == 10500.
l2 = D['tau_awakening'] == 20000.
d1 = D[l1]
d2 = D[l2]
d3 = D[l3]





#
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




#probar plotnine

#https://medium.com/@gscheithauer/data-visualization-in-python-like-in-rs-ggplot2-bc62f8debbf5

#https://plotnine.readthedocs.io/en/stable/



# Analysis

# D.groupby('tau_awakening').mean()

# https://jakevdp.github.io/PythonDataScienceHandbook/03.08-aggregation-and-grouping.html
