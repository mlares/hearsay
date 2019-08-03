
def ceti_exp(GHZ_inner, GHZ_outer, tau_awakening, tau_survive, D_max, t_max):

    #{{{

    import numpy as np
    import random
    from scipy import spatial as sp
    import ceti_tools as ct

    random.seed()
    np.random.seed()

    # lista de CETIs alguna vez activas
    CETIs = dict()
    
    # lista de CETIs actualmente activas
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
    
    # SIMULATION LOOP OVER TIME
    while (t_now < t_max):
    
        t_now, ID_emit, ID_hear, case = t_forthcoming.head.getData()
    
        if case == 1:
    
            # print('desaparece CETI con id:%d' % ID_emit)
            ID_new = ID_emit
            ID_next = ID_new + 1
            t_new_hola = t_now
    
            # sortear el lugar donde aparece dentro de la GHZ
            r = np.sqrt(random.random()*GHZ_outer**2 + GHZ_inner**2)
            o = random.random()*2.*np.pi
            x = r * np.cos(o)  # X position on the galactic plane
            y = r * np.sin(o)  # Y position on the galactic plane
    
            # sortear el tiempo de actividad
            t_new_active = np.random.exponential(tau_survive, 1)[0]
            t_new_chau = t_new_hola + t_new_active
    
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
    
                # if there are other CETIs, compute contacts:
                # encontrar todas (los IDs de) las CETIs dentro del radio D_max
                query_point = [x, y]
                idx = tree.query_ball_point(query_point, r=D_max)
    
                # traverse all CETIs within reach
                for k in idx:
    
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
    
        if case == 2:
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
    
        if case == 3: pass
        if case == 4: pass
    
        # eliminar el tiempo actual
        t_forthcoming.remove_first()
        # salir si no queda nada para hacer:
        if t_forthcoming.size() < 1:
            break
        #t_forthcoming.show()
    
    return(CETIs) 
    #}}}



def redux(D):

    import pickle
    import numpy as np
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

        try:
            CETIs = pickle.load( open(filename, "rb") )
        except EOFError:
            CETIs = []

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
 
