# coding: utf-8

from ceti_exp import ceti_exp
import pickle
import numpy as np
import pandas
import itertools

##########  PARAMETROS
# tiempo medio de aparicion de una nueva CETI (tau_awakening)
# duracion media de una CETI (tau_survice)
# tamaño de la GHZ (GHZ_inner, GHZ_outer)
# maximo alcance de señal (D_max)
 
# PARAMETERS :::

# radio interno de la zona galactica habitable, años luz
GHZ_inner     = 20000.
# radio interno de la zona galactica habitable, años luz
GHZ_outer = 60000.  

# maximo tiempo para simular
t_max = 500000.
 

tau_awakeningS = np.linspace(1000, 20000, 5)
tau_surviveS = np.linspace(100, 10000, 5)
D_maxS = np.linspace(1000, 20000, 5)

df = pandas.DataFrame(columns=['tau_awakening','tau_survive','D_max','name'])

i = 0
for tau_awakening, tau_survive, D_max in itertools.product(tau_awakeningS, tau_surviveS, D_maxS):

   print(tau_awakening, tau_survive, D_max)

   for experiment in range(10):

       i = i + 1

       CETIs = ceti_exp(GHZ_inner, GHZ_outer, 
                               tau_awakening, tau_survive, D_max, t_max)

       filename = '../dat/CETIs_SKRU_' + str(i).zfill(7) + '.dat'
       df.loc[i] = [tau_awakening, tau_survive, D_max, filename]

       pickle.dump( CETIs, open( filename, "wb" ) )

df.to_csv('../dat/experiment_params.csv', index=False)


