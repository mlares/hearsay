
# pensar un nombre para el proyecto

# gossip: galactic observations of signals from intelligent probes

import numpy as np
from configparser import ConfigParser

# Resources:
# http://interactivepython.org/courselib/static/pythonds/BasicDS/ImplementinganOrderedList.html


class parser(ConfigParser):
    # {{{
    """
    parser class
    manipulation of parser from ini files
    """
    def check_file(self, sys_args):
        '''
        chek_file(args):
        Parse paramenters for the simulation from a .ini file

        Args:
            filename (str): the file name of the map to be read

        Raises:

        Returns:
        '''

        from os.path import isfile

        if len(sys_args) == 2:
            filename = sys_args[1]

            if isfile(filename):
                msg = "Loading configuration parameters from {}"
                print(msg.format(filename))
            else:
                print("Input argument is not a valid file")
                print("Using default configuration file instead")
                filename = '../set/experiment.ini'
                # raise SystemExit(1)
        else:
            print('Configuration file expected (just 1 argument)')
            print('example:  python run_correlation.py ../set/experiment.ini')
            print("Using default configuration file")
            # raise SystemExit(1)
            filename = '../set/experiment.ini'

        self.filename = filename

    def read_config_file(self):
        '''
        chek_file(args):
        Parse paramenters for the simulation from a .ini file

        Args:

        Raises:

        Returns:
        '''

        self.read(self.filename)

    def load_filenames(self):
        '''
        load_filenames(self):
        make filenames based on info in config file

        Args:
            None

        Raises:

        Returns:
            list of filenames
        '''

        from collections import namedtuple

        # Experiment settings
        # -----------------------

        exp_id = self['experiment']['exp_id']

        dir_plots = self['output']['dir_plots']
        dir_output = self['output']['dir_output']

        plot_fname = self['output']['plot_fname']
        plot_ftype = self['output']['plot_ftype']

        fname = dir_plots + plot_fname + '_' + exp_id + plot_ftype

        names = 'exp_id \
                 dir_plots \
                 dir_output \
                 plot_fname \
                 plot_ftype \
                 fname'

        parset = namedtuple('pars', names)

        res = parset(exp_id,
                     dir_plots,
                     dir_output,
                     plot_fname,
                     plot_ftype,
                     fname)

        self.filenames = res

    def load_parameters(self):
        '''
        load_parameters(self):
        load parameters from config file

        Args:
            None

        Raises:

        Returns:
            list of parameters as a named tuple

        '''
        print('loading parameters...')
        from collections import namedtuple

        ghz_inner = float(self['simu']['ghz_inner'])
        ghz_outer = float(self['simu']['ghz_outer'])

        t_max = float(self['simu']['t_max'])

        tau_a_min = float(self['simu']['tau_a_min'])
        tau_a_max = float(self['simu']['tau_a_max'])
        tau_a_nbins = int(self['simu']['tau_a_nbins'])

        tau_s_min = float(self['simu']['tau_s_min'])
        tau_s_max = float(self['simu']['tau_s_max'])
        tau_s_nbins = int(self['simu']['tau_s_nbins'])

        d_max_min = float(self['simu']['d_max_min'])
        d_max_max = float(self['simu']['d_max_max'])
        d_max_nbins = int(self['simu']['d_max_nbins'])

        nran = int(self['simu']['nran'])

        # Experiment settings
        # -----------------------

        exp_id = self['experiment']['exp_id']

        dir_plots = self['output']['dir_plots']
        dir_output = self['output']['dir_output']

        plot_fname = self['output']['plot_fname']
        plot_ftype = self['output']['plot_ftype']

        fname = dir_plots + plot_fname + '_' + exp_id + plot_ftype

        # ----

        names = 'ghz_inner ghz_outer t_max tau_a_min tau_a_max tau_a_nbins \
        tau_s_min tau_s_max tau_s_nbins d_max_min d_max_max d_max_nbins nran \
        exp_id dir_plots dir_output plot_fname plot_ftype fname'

        parset = namedtuple('pars', names)

        res = parset(ghz_inner, ghz_outer, t_max, tau_a_min, tau_a_max,
                     tau_a_nbins,
                     tau_s_min, tau_s_max, tau_s_nbins, d_max_min, d_max_max,
                     d_max_nbins, nran,
                     exp_id, dir_plots, dir_output,
                     plot_fname, plot_ftype, fname)

        self.p = res

    def show_params(self):
        # {{{
        '''
        Show paramenters for the simulation
        '''
        # }}}
    # }}}


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
    # }}}


class OrderedList:
    # {{{
    def __init__(self):
        self.head = None
        self.last = None

    def show(self):
        state = ['awakening', 'doomsday', 'contact ', 'blackout']
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
                {!i} received signals and {!i} times listened'.format(
            self.state, self.received, self.delivered)

    def __str__(self):
        return 'Causal contact node in state {!s}, having\
                {!i} received signals and {!i} times listened'.format(
            self.state, self.received, self.delivered)
    # }}}


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
        # {{{
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
        # }}}

    def __len__(self):
        return self.ccns

    def __repr__(self):
        print('message')

    def __str__(self):
        print('message')

    def run_experiment(self, p):
        # {{{
        '''
        Make experiment
        (single value of parameters)
        Writes output on a file

        Args:
            p (tuple): A named tuple containing all parameters for the
            simulation

        Raises:

        Returns:
            None

        '''
        from os import makedirs, path
        import itertools
        import pandas
        import pickle

        tau_awakeningS = np.linspace(p.tau_a_min, p.tau_a_max, p.tau_a_nbins)
        tau_surviveS = np.linspace(p.tau_s_min, p.tau_s_max, p.tau_s_nbins)
        D_maxS = np.linspace(p.d_max_min, p.d_max_max, p.d_max_nbins)

        try:
            dirName = p.dir_output + p.exp_id+''
            makedirs(dirName)
            print("Directory ", dirName,  " Created ")
        except FileExistsError:
            print("Directory ", dirName,  " already exists")
        for d in D_maxS:
            dirName = p.dir_output + p.exp_id + '/D' + str(int(d))
            try:
                makedirs(dirName)
                print("Directory ", dirName,  " Created ")
            except FileExistsError:
                print("Directory ", dirName,  " already exists")

        df = pandas.DataFrame(columns=['tau_awakening', 'tau_survive',
                                       'D_max', 'name'])

        k = 0
        l = 0
        itt = itertools.product(tau_awakeningS, tau_surviveS, D_maxS)
        for tau_awakening, tau_survive, D_max in itt:

            pars = [tau_awakening, tau_survive, D_max]
            print(tau_awakening, tau_survive, D_max)
            k += 1
            i = 0
            for experiment in range(p.nran):

                i += 1
                l += 1

                dirName = p.dir_output+p.exp_id + '/D' + str(int(D_max))+'/'
                filename = dirName + str(k).zfill(5) + '_'
                filename = filename + str(i).zfill(3) + '.pk'
                if(path.isfile(filename)):
                    continue

                self.run_simulation(p, pars)

                df.loc[l] = [tau_awakening, tau_survive, D_max, filename]
                pickle.dump(self.MPL, open(filename, "wb"))

        # df.to_csv('../dat/' + exp_ID + '/params.csv', index=False)

        # }}}

    def run_simulation(self, p, pars):
        # {{{
        '''
        Make experiment
        (single value of parameters)

        Args:
            p ():
            pars ():

        Raises:

        Returns:
            list of parameters as a named tuple

        '''
        tau_awakening = pars[0]
        tau_survive = pars[1]
        D_max = pars[2]

        import numpy as np
        import random
        from scipy import spatial as sp

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
            tree = 0
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
                r = np.sqrt(random.random()*p.ghz_outer**2 +
                            p.ghz_inner**2)
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
                MPL[ID_new] = list()
                MPL[ID_new].append(
                    (ID_new, ID_new, x, y, t_new_hola, t_new_chau))

                # sortear el tiempo de aparición de la próxima CETI
                t_next_awakening = np.random.exponential(tau_awakening, 1)[0]
                t_next_awakening = t_new_hola + t_next_awakening
                if t_next_awakening < p.t_max:
                    next_event = [t_next_awakening, ID_next, None, 1]
                    t_forthcoming.add(next_event)

                if len(CHATs_idx) > 0:

                    # if there are other MPL, compute contacts:
                    # encontrar todas (los IDs de) las MPL dentro de D_max
                    query_point = [x, y]
                    idx = tree.query_ball_point(query_point, r=D_max)

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
                            t_new_see_old_start = max(t_old_hola + Dx,
                                                      t_new_hola)
                            next_event = [t_new_see_old_start,
                                          ID_new, ID_old, 3]
                            t_forthcoming.add(next_event)

                            # :end (type 4 event)
                            t_new_see_old_end = min(t_old_chau + Dx,
                                                    t_new_chau)
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
                            next_event = [t_old_see_new_start,
                                          ID_old, ID_new, 3]
                            t_forthcoming.add(next_event)

                            # :end (type 4 event)
                            t_old_see_new_end = min(t_new_chau+Dx, t_old_chau)
                            next_event = [t_old_see_new_end, ID_old, ID_new, 4]
                            t_forthcoming.add(next_event)

                            contact = (ID_old, ID_new,
                                       MPL[ID_new][0][2], MPL[ID_new][0][3],
                                       t_old_see_new_start, t_old_see_new_end)
                            MPL[ID_old].append(contact)

                # agregar la CETI a la lista de posiciones
                # de MPL activas (CHATs)
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

            if case == 3:
                pass
            if case == 4:
                pass

            # eliminar el tiempo actual
            t_forthcoming.remove_first()
            # salir si no queda nada para hacer:
            if t_forthcoming.size() < 1:
                break
            # t_forthcoming.show()

            self.MPL = MPL
        # }}}

    def run_simulation_II(self, Nrealizations, njobs):
        # {{{
        """run_simulation_II(self) : computes the simulations
        in parallel.

        Tasks:
        1. traverse all ccns?

        Args:

        Raises:
            errors?

        Returns:
        """
        from joblib import Parallel, delayed

        results = []

        results = Parallel(n_jobs=njobs, verbose=5, backend="threading")\
            (delayed(unwrap_simulation_self)(i)
             for i in zip([self]*len(Nrealizations), Nrealizations))
        return results
        # }}}

    def redux(D):
        # {{{
        '''
        redux experiment
        '''

        import pickle
        import numpy as np
        index = []
        firstc = []
        ncetis = []
        awaken = []      # lapso de tiempo que esta activa
        waiting = []     # lapso de tiempo que espera hasta el primer contacto
        inbox = []       # cantidad de cetis que esta escuchando
        distancias = []  # distancias a las cetis contactadas
        hangon = []      # lapso de tiempo que esta escuchando otra CETI
        x = []
        y = []
        N = len(D)
        kcross = 0

        for filename in D['name']:

            try:
                CETIs = pickle.load(open(filename, "rb"))
            except EOFError:
                CETIs = []

            M = len(CETIs)
            ncetis.append(M)

            for i in range(M):  # experiments

                k = len(CETIs[i])  # CETIs resulting from the experiment
                inbox.append(k-1)
                awaken.append(CETIs[i][0][5] - CETIs[i][0][4])
                index.append(kcross)
                x.append(CETIs[i][0][2])
                y.append(CETIs[i][0][3])

                firstcontact = 1.e8

                for l in range(1, k):  # traverse contacts

                    earlier = CETIs[i][l][4] - CETIs[i][0][4]
                    firstcontact = min(earlier, firstcontact)
                    Dx = np.sqrt(((
                        np.array(CETIs[i][0][2:4]) -
                        np.array(CETIs[i][l][2:4]))**2).sum())

                    waiting.append(earlier)
                    distancias.append(Dx)
                    hangon.append(CETIs[i][l][5] - CETIs[i][l][4])

                if k > 1:
                    firstc.append(firstcontact)

            kcross += 1

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
               firstc,      # time of the first contact (from awakening)
               ncetis,      # total number of CETIs in each simulated points
               x,           # x position in the galaxy
               y)           # y position in the galaxy

        # }}}

    def reddux(D):
        # {{{
        '''
        reddux experiment
        '''

        import pickle
        import numpy as np
        index = []
        firstc = []
        ncetis = []
        awaken = []      # lapso de tiempo que esta activa
        waiting = []     # lapso de tiempo que espera hasta el primer contacto
        inbox = []       # cantidad de cetis que esta escuchando
        distancias = []  # distancias a las cetis contactadas
        hangon = []      # lapso de tiempo que esta escuchando otra CETI
        x = []
        y = []
        N = len(D)
        kcross = 0

        for filename in D['name']:

            try:
                CETIs = pickle.load(open(filename, "rb"))
            except EOFError:
                CETIs = []

            M = len(CETIs)
            ncetis.append(M)

            for i in range(M):  # experiments

                k = len(CETIs[i])  # CETIs resulting from the experiment
                inbox.append(k-1)
                awaken.append(CETIs[i][0][5] - CETIs[i][0][4])
                index.append(kcross)
                x.append(CETIs[i][0][2])
                y.append(CETIs[i][0][3])

                firstcontact = 1.e8

                for l in range(1, k):  # traverse contacts

                    earlier = CETIs[i][l][4] - CETIs[i][0][4]
                    firstcontact = min(earlier, firstcontact)
                    Dx = np.sqrt(((
                        np.array(CETIs[i][0][2:4]) -
                        np.array(CETIs[i][l][2:4]))**2).sum())

                    waiting.append(earlier)
                    distancias.append(Dx)
                    hangon.append(CETIs[i][l][5] - CETIs[i][l][4])

                if k > 1:
                    firstc.append(firstcontact)

            kcross += 1

        N = 12
        count = [0]*N
        for i in range(N):
            count[i] = inbox.count(i)

        return({
               # all CETIs
               'A': awaken,     # lifetime of the CETI
               'inbox': inbox,  # number of contact a single CETI makes
               'index': index,  # ID if the CETI in the simulation run
               'x': x,          # x position in the galaxy
               'y': y,          # y position in the galaxy
               #
               # all pairs of CETIs that make contact
               'dist': distancias,  # distance between communicating CETIs
               'hangon': hangon,    # duration of the contact
               'w': waiting,        # time elapsed from awakening to contact
               'c1': firstc,        # time of the first contact

               # all simulated points in the parameter space
               'n': ncetis,  # total number of CETIs in each simulated points
               #
               # chosen integer bins in multiplicity
               'count': count})  # distribution of the multiplicity of contacts

        # }}}

    def ShowCETIs(self):
        # {{{
        CETIs = self.MPL
        for i in range(len(CETIs)):
            print('%2d         (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr' %
                  (CETIs[i][0][1], CETIs[i][0][4],
                   CETIs[i][0][5], CETIs[i][0][2], CETIs[i][0][3]))

            k = len(CETIs[i]) - 1
            for l in range(k):
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) -
                    np.array(CETIs[i][l+1][2:4]))**2).sum())

                print('%2d sees %2d (%5.0f, %5.0f) yr      \
                      <%5.0f, %5.0f> lyr distance=%f' % (CETIs[i][l+1][0],
                                                         CETIs[i][l+1][1],
                                                         CETIs[i][l+1][4],
                                                         CETIs[i][l+1][5],
                                                         CETIs[i][l+1][2],
                                                         CETIs[i][l+1][3], Dx))

        # }}}

    # }}}
