# coding: utf-8

# Contents of the CETIs pickle:

# 0. ID of emitting CETI
# 1. ID of emitting [receiving] CETI
# 2. x
# 3. y
# 4. time of Awakening [Contact]
# 5. time of Doomsday  [Blackout]
# This structure is always for the A-D cycle of each emitting CETI,
# and repeats for each contact.
#
# The time span of a CETI is t_D - t_A for CETIs[k][0]
# The time span of a contact is t_B - t_C for CETIs[k][i], with i>0


# derived quantities:
#-------------------
# duration of a civilization (exponential distribution by costruction)
# duration of a civilization listening another
# duration of a civilization being listened by another
# duration of two-way communication channels
# number of contacts as a function of time
# rate of cetis that succeed in contact
# number of contacts a single ceti can make


# PLOTS: distribuciones de:

# numero total de contactos por unidad de tiempo
# (por ej. por cada 1000 años en toda la galaxia)

# duracion de los contactos

# fraccion de tiempo que esta activa la CETI y que tiene 
# contacto con otra

# fraccion de contactos que admiten una respuesta

# tiempo que hay que esperar para el proximo contacto

# cantidad de contactos que hace una unica CETI

# distribucion de las distancias entre CETIs que entran en contacto

# fraccion de las CETIs que al despertar ya ven algo

##########  PLOTS: relaciones entre:

# duracion de actividad y cantidad de contactos

# 

##########  PLOTS: dependencia de valores medios y parametros

# 

#----------------------------------------------------------------------

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas

D = pandas.read_csv('../dat/experiment_params.csv')

awaken = []
waiting = []
inbox = []
distancias = []
hangon = []  # lapso de tiempo que esta escuchando

for experiment in range(6250):

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

N = 5
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
