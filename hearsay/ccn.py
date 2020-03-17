
# pensar un nombre para el proyecto

# gossip: galactic observations of signals from intelligent probes

import numpy as np

# Resources:
# http://interactivepython.org/courselib/static/pythonds/BasicDS/ImplementinganOrderedList.html

# NODE AND LINKED LIST CLASSES
class Node:
    # {{{
    def __init__(self, data):
        self.data = data
        self.next = None

    def getData(self):
        return self.data

    def getNext(self):
        return self.next

    def setNext(self, newnext):
        self.next = newnext
    #}}}

class OrderedList:
    #{{{
    def __init__(self):
        self.head = None
        self.last = None

    def show(self):
        state = ['awakening','doomsday','contact ','blackout']
        current = self.head
        while current is not None:

            if current.getData()[2] is None:
                print("   %6.2f  %14s - %i " %
                        tuple([current.getData()[0],
                            state[current.getData()[3]-1], 
                            current.getData()[1]]))
            else:
                print("   %6.2f  %14s - %i <    %i" % 
                        tuple([current.getData()[0], 
                            state[current.getData()[3]-1], 
                            current.getData()[1], 
                            current.getData()[2]]))

#            print(current.getData())


            current = current.getNext()        

    def add(self, data):
        current = self.head
        previous = None
        stop = False
        while current is not None and not stop:
            if current.getData()[0] > data[0]:
                stop = True
            else:
                previous = current
                current = current.getNext() 
        temp = Node(data)
        if previous is None:
            temp.setNext(self.head)
            self.head = temp
        else:
            temp.setNext(current)
            previous.setNext(temp)

    def remove_first(self):
        self.head = self.head.getNext()

    def isEmpty(self):
        return self.head is None

    def size(self):
        current = self.head
        count = 0
        while current is not None:
            count = count + 1
            current = current.getNext()
        return count
    # }}}

class ccn():
    # {{{
    '''
    class ccn: causal contact node
    methods:
        init: creates a node        
    '''

    def __init__(self):
        self.state = 'pre-awakening'
        self.received = 0
        self.delivered = 0
        self.twoway = 0
        self.t_awakening = 0.
        self.t_doomsday = 0.
        self.n_listening = 0.
        self.n_listened = 0.


    def __len__(self):
        return self.received

    def __repr__(self):
        return 'Causal contact node in state {!s}, having\
                {!i} received signals and {!i} times listened'.format(\
            self.state, self.received, self.delivered)

    def __str__(self):
        return 'Causal contact node in state {!s}, having\
                {!i} received signals and {!i} times listened'.format(\
            self.state, self.received, self.delivered)
    #}}}

def check_file(sys_args):
        import sys
        from os.path import isfile

        if len(sys_args) == 2:    
            filename = sys_args[1]
            if isfile(filename):
                msg = "Loading configuration parameters from {}"
                print(msg.format(filename) )
            else:
                print("Input argument is not a valid file")
                raise SystemExit(1) 
                
        else:
            print('Configuration file expected (just 1 argument)')
            print('example:  python run_correlation.py ../set/config.ini')
            raise SystemExit(1) 
        
        return filename


def unwrap_simulation_self(arg, **kwarg):
    return GalacticNetwork.run_simulation(*arg, **kwarg)

class GalacticNetwork():
    # {{{
    '''
    class GalacticNetwork: network of contacts from CCNs.
    methods:
        load: 
    '''

    def __init__(self):
        #{{{
        '''
        Instantiate Galaxy object
        '''
        self.params = dict()
#        self.GHZ_inner = 0.
#        self.GHZ_outer = 1.
#        self.t_max = 1.e6
#        self.tau_awakening = 5000.
#        self.tau_survive = 5000.
#        self.D_max = 5000.
        #}}}

    def __len__(self):
        return self.ccns

    def __repr__(self):
        print('message')

    def __str__(self):
        print('message')
 
    def set_parameters(self, params):
        #{{{
        '''
        Set paramenters for the simulation
        '''
        self.params = params  # check if this is a dictionary
        #}}}
 
    def get_parameters(self):
        #{{{
        '''
        Get paramenters for the simulation
        '''
        print(self.params)
        #}}}
 
    def load_parameters(self):
        #{{{
        '''
        Parse paramenters for the simulation from a .ini file
        '''

        from collections import namedtuple

        ghz_inner = float(self.params['simu']['ghz_inner'])
        ghz_outer = float(self.params['simu']['ghz_outer'])
        t_max = float(self.params['simu']['t_max'])
        exp_id = self.params['simu']['exp_id']

        tau_a_min = float(self.params['simu']['tau_a_min']) 
        tau_a_max = float(self.params['simu']['tau_a_max'])
        tau_a_nbins = int(self.params['simu']['tau_a_nbins'])

        tau_s_min = float(self.params['simu']['tau_s_min']) 
        tau_s_max = float(self.params['simu']['tau_s_max'])
        tau_s_nbins = int(self.params['simu']['tau_s_nbins'])

        d_max = float(self.params['simu']['d_max'])
        nran = int(self.params['simu']['nran'])

        parset = namedtuple('pars', 'ghz_inner ghz_outer t_max exp_id \
                tau_a_min tau_a_max tau_a_nbins \
                tau_s_min tau_s_max tau_s_nbins \
                d_max nran')
        res = parset(ghz_inner, ghz_outer, t_max, exp_id, \
                tau_a_min, tau_a_max, tau_a_nbins, \
                tau_s_min, tau_s_max, tau_s_nbins, \
                d_max, nran)

        return(res)

        #}}}

    def show_params(self):
        #{{{
        '''
        Show paramenters for the simulation
        '''
        #}}}


    def run_simulation(self):
        #{{{
        '''
        Make experiment


        aca corremos un experimento o varios ??????


        '''

        import numpy as np
        import random
        from scipy import spatial as sp

        p = self.load_parameters()

        random.seed()
        np.random.seed()

        # list of all MPLs
        MPL = dict()
        
        # list of active MPLs
        CHATs = []
        CHATs_idx = []
        
        # inicializacion del tiempo: scalar
        t_now = 0
        # inicializacion del ID: index
        ID = 0
        
        # lista de tiempos de eventos futuros: ordered list
        # [time, ID_emit, ID_receive, case]
        t_forthcoming = OrderedList()
        
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
        while (t_now < p.t_max):
        
            t_now, ID_emit, ID_hear, case = t_forthcoming.head.getData()
        
            if case == 1:
        
                # print('desaparece CETI con id:%d' % ID_emit)
                ID_new = ID_emit
                ID_next = ID_new + 1
                t_new_hola = t_now

                # sortear el lugar donde aparece dentro de la GHZ
                r = np.sqrt(random.random()*p.ghz_outer**2 + \
                        p.ghz_inner**2)
                o = random.random()*2.*np.pi
                x = r * np.cos(o)  # X position on the galactic plane
                y = r * np.sin(o)  # Y position on the galactic plane
        
                # sortear el tiempo de actividad
                t_new_active = np.random.exponential(self.tau_survive, 1)[0]
                t_new_chau = t_new_hola + t_new_active
        
                # agregar el tiempo de desaparición a la lista de tiempos
                next_event = [t_new_chau, ID_new, None, 2]
                t_forthcoming.add(next_event)
        
                # agregar la CETI a la lista histórica
                MPL[ID_new] = list()
                MPL[ID_new].append(
                    (ID_new, ID_new, x, y, t_new_hola, t_new_chau))
        
                # sortear el tiempo de aparición de la próxima CETI
                t_next_awakening = np.random.exponential(self.tau_awakening, 1)[0]
                t_next_awakening = t_new_hola + t_next_awakening
                if t_next_awakening < t_max:
                    next_event = [t_next_awakening, ID_next, None, 1]
                    t_forthcoming.add(next_event)
        
                if len(CHATs_idx) > 0:
        
                    # if there are other MPL, compute contacts:
                    # encontrar todas (los IDs de) las MPL dentro del radio D_max
                    query_point = [x, y]
                    idx = tree.query_ball_point(query_point, r=self.D_max)
        
                    # traverse all MPL within reach
                    for k in idx:
        
                        ID_old = CHATs_idx[k]
        
                        Dx = np.sqrt(
                                    ((np.array(query_point) -
                                     np.array(MPL[ID_old][0][2:4]))**2).sum()
                                    )
        
                        t_old_hola, t_old_chau = MPL[ID_old][0][4:6]
        
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
                                       MPL[ID_old][0][2], MPL[ID_old][0][3],
                                       t_new_see_old_start, t_new_see_old_end)
                            MPL[ID_new].append(contact)
        
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
                                       MPL[ID_new][0][2], MPL[ID_new][0][3],
                                       t_old_see_new_start, t_old_see_new_end)
                            MPL[ID_old].append(contact)
        
                # agregar la CETI a la lista de posiciones de MPL activas (CHATs)
                CHATs.append([x, y])
                CHATs_idx.append(ID_new)
        
                # rehacer el árbol
                tree = sp.cKDTree(data=CHATs)
        
            if case == 2:
                ID_bye = ID_emit
        
                # eliminar la CETI a la lista de MPL activas
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
        
            self.MPL = MPL
        #}}}

    def run_simulation_II(self, Nrealizations):
        #{{{
        """run_simulation_II(self) : computes the simulations
        in parallel.

        Tasks:
        1. traverse all ccns?

        Args:

        Raises:
            errors?

        Returns:
        """                            
        results = []

        results = Parallel(n_jobs=njobs, verbose=5, backend="threading")\
            (delayed(unwrap_simulation_self)(i) 
                    for i in zip([self]*len(Nrealizations),
                        Nrealizations))
        #}}}


    def redux(D):
        #{{{
        '''
        redux experiment
        '''

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
     
        return(awaken,      # lifetime of the CETI
               inbox,       # number of contact a single CETI makes
               distancias,  # distance between communicating CETIs
               hangon,      # duration of the contact
               waiting,     # time elapsed from awakening to contact
               count,       # distribution of the multiplicity of contacts
               index,       # ID if the CETI in the simulation run
               firstc,      # time of the first contact (measured from awakening)
               ncetis,      # total number of CETIs in each simulated points
               x,           # x position in the galaxy
               y)           # y position in the galaxy
    
        #}}}

    def reddux(D):
        #{{{
        '''
        reddux experiment
        '''

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
     
        return({
               # all CETIs
               'A':awaken,    # lifetime of the CETI                  
               'inbox':inbox, # number of contact a single CETI makes 
               'index':index, # ID if the CETI in the simulation run  
               'x':x,         # x position in the galaxy              
               'y':y,         # y position in the galaxy              
               #
               # all pairs of CETIs that make contact
               'dist':distancias, # distance between communicating CETIs                
               'hangon':hangon,   # duration of the contact                             
               'w':waiting,       # time elapsed from awakening to contact              
               'c1':firstc,       # time of the first contact (measured from awakening) 
               #
               # all simulated points in the parameter space
               'n':ncetis,  # total number of CETIs in each simulated points 
               #
               # chosen integer bins in multiplicity
               'count':count}) # distribution of the multiplicity of contacts
                
        #}}}
     
    def ShowCETIs(CETIs):
        #{{{
        for i in range(len(CETIs)):
            print('%2d         (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr' %
                     (CETIs[i][0][1], CETIs[i][0][4],
                      CETIs[i][0][5], CETIs[i][0][2], CETIs[i][0][3]))

            k = len(CETIs[i]) - 1
            for l in range(k):
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) - 
                    np.array(CETIs[i][l+1][2:4]))**2).sum())

                print('%2d sees %2d (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr distance=%f' % (CETIs[i][l+1][0], CETIs[i][l+1][1], CETIs[i][l+1][4], CETIs[i][l+1][5], CETIs[i][l+1][2], CETIs[i][l+1][3], Dx))

        #}}}

    #}}}

