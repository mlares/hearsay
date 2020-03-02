import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas
import itertools
from os import makedirs, path

exp_ID = 'SKRU_07'  # version de alta resolucion
D = pandas.read_csv('../dat/SKRU_07/params_SKRU_07.dat')

#tau_awakeningS = np.linspace(0, 200000, 51)[1:]
#tau_surviveS = np.linspace(0, 500000, 51)[1:]
#D_maxS = [500, 1000., 10000., 20000., 40000, 80000]
#Nran = 50  

esta = []
for filename in D['name']:
    if(path.isfile(filename.lstrip())):
        esta.append(True)
    else:
        esta.append(False) 

esta = np.array(esta)
len(esta)
sum(esta)

falta = D['name'][~esta]









#-------------------------------------
#esta = []
#k=0; l=0
#for tau_awakening, tau_survive, D_max in itertools.product(tau_awakeningS, tau_surviveS, D_maxS):
#
#   print(tau_awakening, tau_survive, D_max)
#   k+=1; i=0 
#   for experiment in range(Nran):
#
#       i+=1; l+=1
#
#       dirName = '../dat/'+exp_ID + '/D' +str(int(D_max))+'/'
#       filename = dirName + str(k).zfill(5) + '_' + str(i).zfill(3) + '.dat'
#
#       if(path.isfile(filename)):
#           esta.append(True)
#       else:
#           esta.append(False)
#-------------------------------------





# Si se corta el programa experiments.py, deinicia el data frame df, y
# luego escribe el archivo final solo de una parte.
# Para corregirlo se puede generar nuevamente el archivo "params.dat":

import numpy as np
import pandas
                    
tau_a = np.linspace(0, 200000, 51)[1:]
tau_s = np.linspace(0, 500000, 51)[1:]
Dmax = [500, 1000., 10000., 20000., 40000, 80000]
rans = range(50)


n_tau_a = len(tau_a)
n_tau_s = len(tau_s)
n_Dmax = len(Dmax)
n_rans = len(rans)

C4 = list(rans)*n_tau_a*n_tau_s*n_Dmax

C3 = list(np.repeat(Dmax, n_rans))* n_tau_a*n_tau_s

C2 = list(np.repeat(tau_s, n_Dmax*n_rans))* n_tau_a

C1 = list(np.repeat(tau_a, n_Dmax*n_tau_s*n_rans))

C5 = np.repeat(list(range(n_tau_a*n_tau_s*n_Dmax)), n_rans)+1


# 1)tau_awakening, 2)tau_survive, 3)dmax, 4)ran, 5)ids
dict = {'a':C1, 'b':C2, 'c':C3, 'd':C4, 'e':C5}

df = pandas.DataFrame(data=dict)

df.to_csv('check_params2.csv', index=False)


awk -F, 'NR<10{printf"%7d, %7d, %6d, %2d, %d, ../dat/SKRU_07/D%d/%5.5d_%3.3d.dat\n", $1, $2, $3, $4, $5,   $3, $5, $4+1}' check_params2.csv > params_SKRU_07.data

awk -F, '{printf"%7d, %7d, %6d, %2d, %d, ../dat/SKRU_07/D%d/%5.5d_%3.3d.dat\n", $1, $2, $3, $4, $5,   $3, $5, $4+1}' check_params2.csv > params_SKRU_07.data

# borrar la primera linea de params_SKRU_07.data
cat params_SKRU_07.hdr params_SKRU_07.data > params_SKRU_07.dat




