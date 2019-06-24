# coding: utf-8
#
#def ceti_exp(
#    GHZ_inner, GHZ_outer, tau_awakening, tau_survive, D_max, t_max
#):
##{{{
#
#    import ceti_tools as ct
#
#    random.seed(420)
#    np.random.seed(420)
#
#    # lista de CETIs alguna vez activas
#    CETIs = dict()
#    
#    # lista de CETIs actualmente activas
#    CHATs = []
#    CHATs_idx = []
#    
#    # inicializacion del tiempo: scalar
#    t_now = 0
#    # inicializacion del ID: index
#    ID = 0
#    
#    # lista de tiempos de eventos futuros: ordered list
#    # [time, ID_emit, ID_receive, case]
#    t_forthcoming = ct.OrderedList()
#    
#    # estructura de arbol para buscar vecinos
#    try:
#        del tree
#    except NameError:
#        pass
#    
#    # INITIALIZATION
#    # Simulation starts when the first CETI appears:
#    next_event = [0., 0, None, 1]
#    t_forthcoming.add(next_event)
#    
#    # SIMULATION LOOP OVER TIME
#    while (t_now < t_max):
#    
#        t_now, ID_emit, ID_hear, case = t_forthcoming.head.getData()
#    
#        if case == 1:
#    
#            # print('desaparece CETI con id:%d' % ID_emit)
#            ID_new = ID_emit
#            ID_next = ID_new + 1
#            t_new_hola = t_now
#    
#            # sortear el lugar donde aparece dentro de la GHZ
#            r = np.sqrt(random.random()*GHZ_outer**2 + GHZ_inner**2)
#            o = random.random()*2.*np.pi
#            x = r * np.cos(o)  # X position on the galactic plane
#            y = r * np.sin(o)  # Y position on the galactic plane
#    
#            # sortear el tiempo de actividad
#            t_new_active = np.random.exponential(tau_survive, 1)[0]
#            t_new_chau = t_new_hola + t_new_active
#    
#            # agregar el tiempo de desaparición a la lista de tiempos
#            next_event = [t_new_chau, ID_new, None, 2]
#            t_forthcoming.add(next_event)
#    
#            # agregar la CETI a la lista histórica
#            CETIs[ID_new] = list()
#            CETIs[ID_new].append(
#                (ID_new, ID_new, x, y, t_new_hola, t_new_chau))
#    
#            # sortear el tiempo de aparición de la próxima CETI
#            t_next_awakening = np.random.exponential(tau_awakening, 1)[0]
#            t_next_awakening = t_new_hola + t_next_awakening
#            if t_next_awakening < t_max:
#                next_event = [t_next_awakening, ID_next, None, 1]
#                t_forthcoming.add(next_event)
#    
#            if len(CHATs_idx) > 0:
#    
#                # if there are other CETIs, compute contacts:
#                # encontrar todas (los IDs de) las CETIs dentro del radio D_max
#                query_point = [x, y]
#                idx = tree.query_ball_point(query_point, r=D_max)
#    
#                # traverse all CETIs within reach
#                for k in idx:
#    
#                    ID_old = CHATs_idx[k]
#    
#                    Dx = np.sqrt(
#                                ((np.array(query_point) -
#                                 np.array(CETIs[ID_old][0][2:4]))**2).sum()
#                                )
#    
#                    t_old_hola, t_old_chau = CETIs[ID_old][0][4:6]
#    
#                    # check if contact is possible
#                    new_can_see_old = (
#                                    (Dx < D_max) &
#                                    (t_new_hola < t_old_chau + Dx) &
#                                    (t_new_chau > t_old_hola + Dx))
#    
#                    if new_can_see_old:  # (·) new sees old
#    
#                        # :start (type 3 event)
#                        t_new_see_old_start = max(t_old_hola + Dx, t_new_hola)
#                        next_event = [t_new_see_old_start, ID_new, ID_old, 3]
#                        t_forthcoming.add(next_event)
#    
#                        # :end (type 4 event)
#                        t_new_see_old_end = min(t_old_chau + Dx, t_new_chau)
#                        next_event = [t_new_see_old_end, ID_new, ID_old, 4]
#                        t_forthcoming.add(next_event)
#    
#                        contact = (ID_new, ID_old,
#                                   CETIs[ID_old][0][2], CETIs[ID_old][0][3],
#                                   t_new_see_old_start, t_new_see_old_end)
#                        CETIs[ID_new].append(contact)
#    
#                    # check if contact is possible
#                    old_can_see_new = (
#                        (Dx < D_max) & (t_new_hola+Dx > t_old_hola) &
#                        (t_new_hola+Dx < t_old_chau))
#    
#                    if old_can_see_new:  # (·) old sees new
#    
#                        # :start (type 3 event)
#                        t_old_see_new_start = t_new_hola + Dx
#                        next_event = [t_old_see_new_start, ID_old, ID_new, 3]
#                        t_forthcoming.add(next_event)
#    
#                        # :end (type 4 event)
#                        t_old_see_new_end = min(t_new_chau+Dx, t_old_chau)
#                        next_event = [t_old_see_new_end, ID_old, ID_new, 4]
#                        t_forthcoming.add(next_event)
#    
#                        contact = (ID_old, ID_new,
#                                   CETIs[ID_new][0][2], CETIs[ID_new][0][3],
#                                   t_old_see_new_start, t_old_see_new_end)
#                        CETIs[ID_old].append(contact)
#    
#            # agregar la CETI a la lista de posiciones de CETIs activas (CHATs)
#            CHATs.append([x, y])
#            CHATs_idx.append(ID_new)
#    
#            # rehacer el árbol
#            tree = sp.cKDTree(data=CHATs)
#    
#        if case == 2:
#            ID_bye = ID_emit
#    
#            # eliminar la CETI a la lista de CETIs activas
#            # [ID, x, y, t_new_hola, t_new_chau]
#            try:
#                id_loc = CHATs_idx.index(ID_bye)
#                del CHATs[id_loc]
#                del CHATs_idx[id_loc]
#    
#            except TypeError:
#                pass
#    
#            # rehacer el árbol
#            if len(CHATs) > 0:
#                tree = sp.cKDTree(data=CHATs)
#    
#        if case == 3: pass
#        if case == 4: pass
#    
#        # eliminar el tiempo actual
#        t_forthcoming.remove_first()
#        # salir si no queda nada para hacer:
#        if t_forthcoming.size() < 1:
#            break
#        t_forthcoming.show()
#    
#    return(CETIs) 
##}}}
#


import ceti_exp

import numpy as np
import random
import pandas as pd
from scipy import spatial as sp
import time
import sys
import pickle
import ceti_tools as ct

# para recargar el modulo si se hacen modificaciones:
from importlib import reload
ct = reload(ct)

# para guardar la salida
from io import StringIO

# DEBUGING
# import pdb; pdb.set_trace()

# CASES:
# 1 - awakening
# 2 - doomsday
# 3 - contact
# 4 - blackout

random.seed(420)
np.random.seed(420)

message = "PRESS%ENTER%TO%CONTINUE"
message = "~~~~~~~~~~~~~~~~~~~~~~~"
message = "\n"+message*3+"\n"
message = ""
state = ['awakening','doomsday','contact ','blackout']

# FIXED AND SIMULATION VARIABLES
# {{{

# PARAMETERS :::

# radio interno de la zona galactica habitable, años luz
GHZ_inner = 1000.0

# radio interno de la zona galactica habitable, años luz
GHZ_outer = 5000.0

# tiempo medio, en años, que hay que esperar para que
# aparezca otra CETI en la galaxia
tau_awakening = 500.

# Tiempo medio, en años, durante el cual una CETI esta activa
tau_survive = 1000.

# Maxima distancia, en años luz, a la cual una CETI puede
# enviar o recibir mensajes
D_max = 3000.

# maximo tiempo para simular
t_max = 5000.

# flag to set interactive session
interactive = True

# }}}


# {{{

# VARIABLES :::

# lista de CETIs alguna vez activas: dictionary
CETIs = dict()

# lista de CETIs actualmente activas: ndarray
CHATs = []
CHATs_idx = []

# inicializacion del tiempo: scalar
t_now = 0

# inicializacion del ID: index
ID = 0

# lista de tiempos de eventos futuros: ordered list
# [time, ID_emit, ID_receive, case]
t_forthcoming = ct.OrderedList()

# estructura de arbol para buscar vecinos
try:
    del tree
except NameError:
    pass

# INITIALIZATION
# Simulation starts when the first CETI appears:
next_event = [0., 0, None, 1]
t_forthcoming.add(next_event)


if not interactive:
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result


# SIMULATION LOOP OVER TIME
while (t_now < t_max):

    t_now, ID_emit, ID_hear, case = t_forthcoming.head.getData()

    if interactive: wait = input(message)

    print('<<< t = %f >>> %10s of %3d' % (t_now, state[case-1], ID_emit))

    #print('[ case:%d | id:%d ]      <<< t = %f >>>' % (case, ID_emit, t_now))
    #print('   active CETIs', CHATs_idx)
    print(CHATs_idx)
    #t_forthcoming.show()

    # sys.stdout.write("\rTime: %f  (max=%f)\n" % (t_now, t_max))
    # sys.stdout.flush()
    # print('tiempo: %9.3f | ID_emit: %5d|CASE %1d'%(t_now,ID_emit,case))

    if case == 1:
        # -------------------------------------
        # NEW CETI
        # -------------------------------------
        # {{{

        # print('desaparece CETI con id:%d' % ID_emit)
        ID_new = ID_emit
        ID_next = ID_new + 1
        t_new_hola = t_now

        # sortear el lugar donde aparece dentro de la GHZ
        r = np.sqrt(random.random()*GHZ_outer**2 + GHZ_inner**2)
        o = random.random()*2.*np.pi
        x = r * np.cos(o)  # X position on the galactic plane
        y = r * np.sin(o)  # Y position on the galactic plane
        #print('posiciones para el ID %5d   :::: %8.3f %8.3f' % (ID_new, x, y))

        # sortear el tiempo de actividad
        t_new_active = np.random.exponential(tau_survive, 1)[0]
        t_new_chau = t_new_hola + t_new_active
        #print('tiempos para el ID %2d   :::: %8.3f %8.3f' % (ID_new, t_new_active, t_new_chau))

        # agregar el tiempo de desaparición a la lista de tiempos
        next_event = [t_new_chau, ID_new, None, 2]
        t_forthcoming.add(next_event)

        # agregar la CETI a la lista histórica
        CETIs[ID_new] = list()
        CETIs[ID_new].append(
            (ID_new, ID_new, x, y, t_new_hola, t_new_chau))

        # sortear el tiempo de aparición de la próxima CETI
        t_next_awakening = np.random.exponential(tau_awakening, 1)[0]
        t_next_awakening = t_new_hola + t_next_awakening
        if t_next_awakening < t_max:
            next_event = [t_next_awakening, ID_next, None, 1]
            t_forthcoming.add(next_event)

        if len(CHATs_idx) > 0:

            #print('entering loop to compute contacts')

            # if there are other CETIs, compute contacts:
            # encontrar todas (los IDs de) las CETIs dentro del radio D_max
            query_point = [x, y]
            idx = tree.query_ball_point(query_point, r=D_max)

            # traverse all CETIs within reach
            for k in idx:

                #print(k)

                ID_old = CHATs_idx[k]

                Dx = np.sqrt(
                            ((np.array(query_point) -
                             np.array(CETIs[ID_old][0][2:4]))**2).sum()
                            )

                t_old_hola, t_old_chau = CETIs[ID_old][0][4:6]

                # check if contact is possible
                new_can_see_old = (
                                (Dx < D_max) &
                                (t_new_hola < t_old_chau + Dx) &
                                (t_new_chau > t_old_hola + Dx))

                if new_can_see_old:  # (·) new sees old

                    # :start (type 3 event)
                    t_new_see_old_start = max(t_old_hola + Dx, t_new_hola)
                    next_event = [t_new_see_old_start, ID_new, ID_old, 3]
                    t_forthcoming.add(next_event)

                    # :end (type 4 event)
                    t_new_see_old_end = min(t_old_chau + Dx, t_new_chau)
                    next_event = [t_new_see_old_end, ID_new, ID_old, 4]
                    t_forthcoming.add(next_event)

                    contact = (ID_new, ID_old,
                               CETIs[ID_old][0][2], CETIs[ID_old][0][3],
                               t_new_see_old_start, t_new_see_old_end)
                    CETIs[ID_new].append(contact)

                # check if contact is possible
                old_can_see_new = (
                    (Dx < D_max) & (t_new_hola+Dx > t_old_hola) &
                    (t_new_hola+Dx < t_old_chau))

                if old_can_see_new:  # (·) old sees new

                    # :start (type 3 event)
                    t_old_see_new_start = t_new_hola + Dx
                    next_event = [t_old_see_new_start, ID_old, ID_new, 3]
                    t_forthcoming.add(next_event)

                    # :end (type 4 event)
                    t_old_see_new_end = min(t_new_chau+Dx, t_old_chau)
                    next_event = [t_old_see_new_end, ID_old, ID_new, 4]
                    t_forthcoming.add(next_event)

                    contact = (ID_old, ID_new,
                               CETIs[ID_new][0][2], CETIs[ID_new][0][3],
                               t_old_see_new_start, t_old_see_new_end)
                    CETIs[ID_old].append(contact)

        # agregar la CETI a la lista de posiciones de CETIs activas (CHATs)
        CHATs.append([x, y])
        CHATs_idx.append(ID_new)

        # rehacer el árbol
        tree = sp.cKDTree(data=CHATs)

        # }}}

    if case == 2:
        # -------------------------------------
        # END CETI
        # -------------------------------------
        # {{{
        # print('desaparece CETI con id:%d' % ID_emit)

        ID_bye = ID_emit

        # eliminar la CETI a la lista de CETIs activas
        # [ID, x, y, t_new_hola, t_new_chau]
        try:
            id_loc = CHATs_idx.index(ID_bye)
            del CHATs[id_loc]
            del CHATs_idx[id_loc]

        except TypeError:
            pass

        # rehacer el árbol
        if len(CHATs) > 0:
            tree = sp.cKDTree(data=CHATs)

        # }}}

    if case == 3:
        # -------------------------------------
        # NEW CONTACT
        # -------------------------------------
        # {{{
        pass
        # print('primer contacto entre %d y %d' % (ID_emit, ID_hear))

        # }}}

    if case == 4:
        # -------------------------------------
        # END CONTACT
        # -------------------------------------
        # {{{
        pass
        # print('ultimo contacto entre %d y %d' % (ID_emit, ID_hear))
        # }}}

    # eliminar el tiempo actual
    t_forthcoming.remove_first()

    # salir si no queda nada para hacer:
    if t_forthcoming.size() < 1:
        break

    t_forthcoming.show()
    print(CHATs_idx)




print('\n\n')
ct.ShowCETIs(CETIs)

if not interactive:
    sys.stdout = old_stdout
    result_string = result.getvalue()

    i = input('experiment code: ').zfill(3)

    fname = 'experiment_' + i + '.dat'
    savef = open(fname,'w')
    savef.write(result_string)
    savef.close()


# El experimento consiste es correr interactivamente hasta que se
# corrobora que anda bien.  Luego puse la funcion en un modulo, la 
# llamo con la misma semilla y anda igual
#
#fCETIs = ceti_exp.ceti_exp(
#    GHZ_inner, GHZ_outer, tau_awakening, tau_survive, D_max, t_max
#)
#ct.ShowCETIs(fCETIs)
