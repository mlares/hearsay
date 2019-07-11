# coding: utf-8

from ceti_exp import ceti_exp
import pickle
import numpy as np
import pandas
import itertools
from os import makedirs

##########  PARAMETROS
# tiempo medio de aparicion de una nueva CETI (tau_awakening)
# duracion media de una CETI (tau_survice)
# tamaño de la GHZ (GHZ_inner, GHZ_outer)
# maximo alcance de señal (D_max)
 
# PARAMETERS :::

# radio interno de la zona galactica habitable, años luz
GHZ_inner = 20000.
# radio interno de la zona galactica habitable, años luz
GHZ_outer = 60000.  

# maximo tiempo para simular
t_max = 700000.

# experiment ID
exp_ID = 'SKRU_01'
 
tau_awakeningS = np.linspace(0, 24000, 25)[1:]
tau_surviveS =   np.linspace(0, 50000, 26)[1:]

# Separate data in directories according to D_max
#D_maxS = np.linspace(0, 40000, 11)[1::2]
D_maxS = [1000., 5000., 10000, 20000., 40000.]

try:
    dirName = '../dat/'+exp_ID+''
    makedirs(dirName)
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
        print("Directory " , dirName ,  " already exists") 
for d in D_maxS:
    dirName = '../dat/'+exp_ID + '/D' +str(int(d))
    try:
        makedirs(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")


df = pandas.DataFrame(columns=['tau_awakening','tau_survive','D_max','name'])

k=0; l=0
for tau_awakening, tau_survive, D_max in itertools.product(tau_awakeningS, tau_surviveS, D_maxS):

   #e = (GHZ_inner, GHZ_outer,tau_awakening, tau_survive, D_max, t_max)
   print(tau_awakening, tau_survive, D_max)
   k+=1; i=0 
   for experiment in range(100):

       i+=1; l+=1

       #CETIs = ceti_exp(*e)
       dirName = '../dat/'+exp_ID + '/D' +str(int(D_max))+'/'
       filename = dirName + str(k).zfill(5) + '_' + str(i).zfill(3) + '.dat'
       df.loc[l] = [tau_awakening, tau_survive, D_max, filename]

       #pickle.dump( CETIs, open( filename, "wb" ) )


df.to_csv('../dat/' + exp_ID + '/params.csv', index=False)


