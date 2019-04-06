# coding: utf-8

from ceti_sim import ceti_experiment
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

# tiempo medio, en años, que hay que esperar para que 
# aparezca otra CETI en la galaxia
tau_awakening = 5000.

# Tiempo medio, en años, durante el cual una CETI esta activa
tau_survive = 3000.   

# Maxima distancia, en años luz, a la cual una CETI puede 
# enviar o recibir mensajes
D_max = 20000.   

# maximo tiempo para simular
t_max = 500000.
 

tau_awakeningS = np.linspace(1000, 20000, 5)
tau_surviveS = np.linspace(100, 10000, 5)
D_maxS = np.linspace(1000, 20000, 5)

df = pandas.DataFrame(columns=['tau_awakening','tau_survive','D_max','name'])

i = 0
for tau_awakening, tau_survive, D_max in itertools.product(tau_awakeningS, tau_surviveS, D_maxS):

   print(tau_awakening, tau_survive, D_max)

   for experiment in range(50):

       i = i + 1

       CETIs = ceti_experiment(GHZ_inner, GHZ_outer, 
                               tau_awakening, tau_survive, D_max, t_max)

       filename = '../dat/CETIs_' + str(i).zfill(7) + '.dat'
       df.loc[i] = [tau_awakening, tau_survive, D_max, filename]

       pickle.dump( CETIs, open( filename, "wb" ) )

df.to_csv('../dat/experiment_params.csv')
