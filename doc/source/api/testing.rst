***********
Testing
***********

Make several tests to reach a good code coverage and verify
  if results are as expected.

Proposed test to develop
========================

  * 


Testing tools and procedures
============================

In order to make testing, we should use any of the following tools:

* `pytest <https://docs.pytest.org/en/latest/>`_
* hypothesis


**pytest examples**

"pytest will run all files of the form test_\*.py or \*_test.py in the current directory and its subdirectories."
So, simply go to the tst directory, and run pytest.

In the environment:

.. code-block::

   pip install pytest


In the code (example from pytest documentation):

.. code-block::

   def inc(x):
       return x + 1


   def test_answer():
       assert inc(3) == 5

How to run test:

From the CLI, write:

.. code-block::

   pytest




Experiment series
=================


SERIES:   MCU ALIEN RACES

_________________________________________________________
FLERKEN [FLKN]

First set of interactive experiments, carried out in order to
debug possible inconsistencies.  Based on ceti_test.py.


KREE [KREE]

CELESTIALS [CLTL]

SKRULLS [SKRU]

CHITAURI [CHTR]

INHUMANS [INHM]






key:  A: Awakening, B: Blackout, C: Contact, D: Doomsday

CONTENTS OF THE <CETIS> PICKLE:


 0. ID of emitting CETI
 1. ID of emitting [receiving] CETI
 2. x position in the galaxy
 3. y position in the galaxy
 4. time of Awakening [Contact]
 5. time of Doomsday  [Blackout]


 This structure is always for the A-D cycle of each emitting CETI,
 and repeats for each contact (C-B).

- The time span of a CETI is t_D - t_A for CETIs[k][0]
- The time span of a contact is t_B - t_C for CETIs[k][i], with i>0

- age: time elapsed from A to a given time
- ceti_e: ceti emitting signals (emiter)
- ceti_r: ceti receiving signals (receiver)
- ceti_c: ceti that listen at least another ceti (citizen)
- ceti_h: ceti that is lestened by at least another ceti

for Check purposes
-------------------
 duration of a civilization (exponential distribution by costruction)
 time from the appeareance of a CETI to the next

 
derived quantities
-------------------
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
#----------------------------------------------------------------------
 
 
# awaken = t_doomsday - t_Awakening de cada ceti
# inbox = nro. de cetis que escucha cada ceti
# firstc = tiempo de espera hasta el primer contacto
# index = indice de la lista de parametros
# 
# waiting = tiempo de espera entre A y todos los C de una ceti
# hangon = lapso de tiempo que dura la escucha (t_blackout - t_contact)
# distancias = distancias entre cetis que alguna vez entraron en contacto
 

##########  PARAMETROS
# tiempo medio de aparicion de una nueva CETI (tau_awakening)
# duracion media de una CETI (tau_survice)
# tamaño de la GHZ (GHZ_inner, GHZ_outer)
# maximo alcance de señal (D_max)
 



PLOTS
============================================================================

distribucion de la cantidad de contactos



IDEAS
==========================================================================

Hacer un video de la simulacion

Calcular el área de la galaxia que está ocupada por signals

Hacer el codigo paralelo: paralelo en python o en fortran
El problema de fortran es el balltree.

