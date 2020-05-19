"""HEARSAY.

description
"""

import numpy as np
from configparser import ConfigParser
import pandas as pd
import pickle
import sys
from tqdm import tqdm


class parser(ConfigParser):
    """parser class.

    manipulation of parser from ini files
    """

    def __init__(self, argv=None, *args, **kwargs):
        """Initialize a parser.

        Args:
            None
        Returns:
            None
        Raises:
            None
        """
        super().__init__()
        self.message = None
        self.check_file(argv)
        self.read_config_file()

        self.load_filenames()
        self.load_parameters(*args, **kwargs)
        self.check_settings()

    def check_file(self, sys_args=""):
        """Parse paramenters for the simulation from a .ini file.

        Args:
            filename (str): the file name of the map to be read

        Raises:
            None

        Returns:
            None
        """
        from os.path import isfile

        mess = ("Configuration file expected:"
                "\n\t filename or CLI input"
                "\n\t example:  python run_correlation.py"
                "\n\t ../set/experiment.ini"
                "\n\t Using default configuration file")
        if isinstance(sys_args, str):
            if isfile(sys_args):
                msg = f"Loading configuration parameters from {sys_args}"
                self.message = msg
                filename = sys_args
            else:
                self.message = "Input argument is not a valid file\
                                Using default configuration file instead"
                filename = '../set/experiment.ini'

        elif isinstance(sys_args, list):

            if len(sys_args) == 2:
                filename = sys_args[1]

                if isfile(filename):
                    msg = f"Loading configuration parameters from {filename}"
                    self.message = msg
                else:
                    self.message = mess
                    filename = '../set/experiment.ini'
            else:
                self.message = mess
                filename = '../set/experiment.ini'

        else:
            self.message = mess
            filename = '../set/experiment.ini'

        self.filename = filename

    def read_config_file(self):
        """Parse paramenters for the simulation from a .ini file.

        Args:
            None

        Raises:
            None

        Returns:
            None
        """
        self.read(self.filename)

    def load_filenames(self):
        """Make filenames based on info in config file.

        Args:
            None

        Raises:
            None

        Returns:
            list of filenames
        """
        from collections import namedtuple

        # Experiment settings
        exp_id = self['experiment']['exp_id']
        dir_plots = self['output']['dir_plots']
        pars_root = self['output']['pars_root']
        progress_root = self['output']['progress_root']
        dir_output = self['output']['dir_output']
        plot_fname = self['output']['plot_fname']
        plot_ftype = self['output']['plot_ftype']

        fname = dir_plots + plot_fname + '_' + exp_id + plot_ftype

        names = 'exp_id \
                 dir_plots \
                 dir_output \
                 pars_root \
                 progress_root \
                 plot_fname \
                 plot_ftype \
                 fname'

        parset = namedtuple('pars', names)

        res = parset(exp_id,
                     dir_plots,
                     dir_output,
                     pars_root,
                     progress_root,
                     plot_fname,
                     plot_ftype,
                     fname)

        self.filenames = res

    def load_parameters(self, nran=None):
        """Load parameters from config file.

        Args:
            None

        Raises:
            None

        Returns:
            list of parameters as a named tuple
        """
        choice = self['UX']['verbose']
        if choice.lower() in 'yesitrue':
            verbose = True
        elif choice.lower() in 'nofalse':
            verbose = False
        else:
            print('warning in .ini file: UX: verbose')
            verbose = False

        if verbose:
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

        if nran is None:
            nran = int(self['simu']['nran'])

        choices = self['simu']['run_parallel']
        if choices.lower() in 'yesitrue':
            run_parallel = True
        elif choices.lower() in 'nofalse':
            run_parallel = False
        else:
            print('Error in .ini file: simu: run_parallel must be Y/N'
                  'Exiting.')
            exit()

        # Experiment settings
        exp_id = self['experiment']['exp_id']
        njobs = int(self['simu']['njobs'])
        dir_plots = self['output']['dir_plots']
        dir_output = self['output']['dir_output']
        pars_root = self['output']['pars_root']
        plot_fname = self['output']['plot_fname']
        plot_ftype = self['output']['plot_ftype']
        fname = dir_plots + plot_fname + '_' + exp_id + plot_ftype

        choice = self['UX']['show_progress']
        if choice.lower() in 'yesitrue':
            showp = True
        elif choice.lower() in 'nofalse':
            showp = False
        else:
            print('warning in .ini file: UX: show_progress')
            showp = False

        string_overwrite = self['output']['clobber']
        if string_overwrite.lower() in 'yesitrue':
            overwrite = True
        elif string_overwrite.lower() in 'nofalse':
            overwrite = False
        else:
            print('warning in .ini file: output: clobber')
            overwrite = False

        names = ['ghz_inner',
                 'ghz_outer',
                 't_max',
                 'tau_a_min',
                 'tau_a_max',
                 'tau_a_nbins',
                 'tau_s_min',
                 'tau_s_max',
                 'tau_s_nbins',
                 'd_max_min',
                 'd_max_max',
                 'd_max_nbins',
                 'nran',
                 'run_parallel',
                 'njobs',
                 'exp_id',
                 'dir_plots',
                 'dir_output',
                 'pars_root',
                 'plot_fname',
                 'plot_ftype',
                 'fname',
                 'showp',
                 'overwrite',
                 'verbose']
        names = ' '.join(names)

        parset = namedtuple('pars', names)

        res = parset(ghz_inner,
                     ghz_outer,
                     t_max,
                     tau_a_min,
                     tau_a_max,
                     tau_a_nbins,
                     tau_s_min,
                     tau_s_max,
                     tau_s_nbins,
                     d_max_min,
                     d_max_max,
                     d_max_nbins,
                     nran,
                     run_parallel,
                     njobs,
                     exp_id,
                     dir_plots,
                     dir_output,
                     pars_root,
                     plot_fname,
                     plot_ftype,
                     fname,
                     showp,
                     overwrite,
                     verbose)

        self.p = res

    def check_settings(self):
        """Check if parameters make sense.

        Args:
            None

        Raises:
            None

        Returns:
            Exception if settings have inconsistencies.
        """
        from os import path, makedirs
        from sys import exit

        if self.p.verbose:
            print(self.message)
            print('Checking settings...')

        if not path.isdir(self.p.dir_output):
            print(f"Directory {self.p.dir_output} does not exist")

            try:
                makedirs(self.p.dir_output)
                if self.p.verbose:
                    print("Directory ", self.p.dir_output,  " Created ")
            except FileExistsError:
                # directory already exists
                pass

        if not path.isdir(self.p.dir_plots):
            print(f"Directory {self.p.dir_plots} does not exist")

            try:
                makedirs(self.p.dir_plots)
                if self.p.verbose:
                    print("Directory ", self.p.dir_plots,  " Created ")
            except FileExistsError:
                # directory already exists
                pass


class Node:
    """Node and linked list classes.

    This class contains tools to manipulate nodes.  A node is
    a point in the Galaxy that a acquires the ability to emit
    and receive messages at a given time.  A set of nodes make
    a linked list.
    """

    def __init__(self, data):
        """Initialize a node.

        Args:
            data: (single value)
            A number or value that can be compared and supports
            the <grater than> operator.
        Returns:
            None
        Raises:
            None
        """
        self.data = data
        self.next = None

    def getData(self):
        """Get data in a node.

        Args:
            None
        """
        return self.data

    def getNext(self):
        """Get the next node, if exists.

        Args:
            None
        """
        return self.next

    def setNext(self, newnext):
        """Set the next node.

        Args:
            Node
        """
        self.next = newnext


class OrderedList:
    """Ordered list class.

    Tools to make ordered lists. This structure is useful because it can be
    traversed and a new node can be added at any stage.
    # based on http://interactivepython.org/courselib/static/pythonds/
    #  BasicDS/ImplementinganOrderedList.html
    """

    def __init__(self):
        """Initialize ordered list.

        Args:
            None
        """
        self.head = None
        self.last = None

    def show(self):
        """Print an ordered list.

        Args:
            None
        """
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

            current = current.getNext()

    def add(self, data):
        """Add an element to an ordered list.

        Args:
            Data (number)
        """
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
        """Remove first element.

        Args:
            None
        """
        self.head = self.head.getNext()

    def isEmpty(self):
        """Ask if list is empty.

        Args:
            None
        """
        return self.head is None

    def size(self):
        """Retrieve the size of the list.

        Args:
            None
        """
        current = self.head
        count = 0
        while current is not None:
            count = count + 1
            current = current.getNext()
        return count


class ccn():
    """Class for causal contact nodes.

    methods:
        init:
            creates a node
        __len__:
            None
        __repr__:
            None
        __str__:
            None
    """

    def __init__(self):
        """Initialize.

        Args:
            None
        """
        self.state = 'pre-awakening'
        self.received = 0
        self.delivered = 0
        self.twoway = 0
        self.t_awakening = 0.
        self.t_doomsday = 0.
        self.n_listening = 0.
        self.n_listened = 0.

    def __len__(self):
        """Get the number of contacts for this node.

        Args:
            None
        """
        return self.received

    def __repr__(self):
        """Representation for print.

        Args:
            None
        """
        return 'Causal contact node in state {!s}, having\
                {!i} received signals and {!i} times listened'.format(
            self.state, self.received, self.delivered)

    def __str__(self):
        """Show the node as a string.

        Args:
            None
        """
        return 'Causal contact node in state {!s}, having\
                {!i} received signals and {!i} times listened'.format(
            self.state, self.received, self.delivered)


def unwrap_experiment_self(arg, **kwarg):
    """Wrap the serial function for parallel run.

    This function just call the serialized version, but allows to run
    it concurrently.
    """
    return GalacticNetwork.run_experiment_params(*arg, **kwarg)


class GalacticNetwork():
    """GalacticNetwork: network of contacts from CCNs.

    methods:
        init:
            creates a node
        __len__:
            None
        __repr__:
            None
        __str__:
            None
        run_experiment:
            None
        run_simulation:
            None
    """

    def __init__(self, conf=None):
        """Instantiate Galaxy object.

        Args:
            None
        """
        self.param_set = dict()
        self.conf = conf

    def __len__(self):
        """Return the number of contacts.

        Args:
            None
        """
        return self.ccns

    def __repr__(self):
        """Represent with a string.

        Args:
            None
        """
        print('message')

    def __str__(self):
        """Represent with a string.

        Args:
            None
        """
        print('message')

    def run_experiment(self, spars=None,
                       A=None, S=None, D=None,
                       parallel=False, njobs=None):
        """Run an experiment.

        An experiment requires a set of at least three parameters, which are
        taken from the configuration file.

        Args:
           A (number or list of numbers, optional)
           D (number or list of numbers, optional)
           S (number or list of numbers, optional)
           parallel (boolean, optional). Flag to indicate if run is made using
           the parallelized version.  Default: False.
        """
        import itertools
        p = self.conf.p
        if spars is None:
            tau_awakeningS = np.linspace(p.tau_a_min, p.tau_a_max,
                                         p.tau_a_nbins)
            tau_surviveS = np.linspace(p.tau_s_min, p.tau_s_max, p.tau_s_nbins)
            D_maxS = np.linspace(p.d_max_min, p.d_max_max, p.d_max_nbins)
        else:
            tau_awakeningS = [spars[0]]
            tau_surviveS = [spars[1]]
            D_maxS = [spars[2]]

        if A is not None:
            tau_awakeningS = A
        if S is not None:
            tau_surviveS = S
        if D is not None:
            D_maxS = D

        if njobs is not None:
            parallel = True

        ll = []
        for i in itertools.product(tau_awakeningS, tau_surviveS, D_maxS):
            ll.append(i)

        if parallel:
            if njobs is None:
                njobs = self.conf.p.njobs
            self.run_experiment_params_II(ll, njobs)
        else:
            self.run_experiment_param_set(ll)

    def run_experiment_params_II(self, params, njobs):
        """Run an experiment, parallel version.

        An experiment requires a set of at least three parameters, which are
        taken from the configuration file.

        Args:
        params: the parameters
        njobs: number of jobs
        """
        import pandas
        from joblib import Parallel, delayed

        # backend: threading / multiprocessing / loky
        Pll = Parallel(n_jobs=njobs, verbose=5, backend="loky")
        ids = np.array(range(len(params))) + 1
        z = zip([self]*len(params), params, ids)
        d_experiment = delayed(unwrap_experiment_self)
        results = Pll(d_experiment(i) for i in z)

        df = pandas.DataFrame(columns=['tau_awakening', 'tau_survive',
                                       'D_max', 'name'])

        p = self.conf.p
        k = 0
        j = 0
        for pp in params:
            (tau_awakening, tau_survive, D_max) = pp
            k += 1
            i = 0
            for experiment in range(p.nran):
                i += 1
                j += 1
                dirName = p.dir_output+p.exp_id + '/D' + str(int(D_max))+'/'
                filename = dirName + str(k).zfill(5) + '_'
                filename = filename + str(i).zfill(3) + '.pk'
                df.loc[j] = [tau_awakening, tau_survive, D_max, filename]

        # write files
        fn = self.conf.filenames
        fname = fn.dir_output + '/' + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df.to_csv(fname, index=False)

        return results

    def run_experiment_params(self, params, idx):
        """Make experiment.

        Requires a single value of parameters.
        Writes output on a file

        Args:
            p (tuple): A named tuple containing all parameters for the
            simulation: [tau_awakening, tau_survive, D_max, index]

        Raises:
            None

        Returns:
            None
        """
        from os import makedirs, path

        p = self.conf.p

        try:
            dirName = p.dir_output + p.exp_id+''
            makedirs(dirName)
            if p.verbose:
                print("Directory ", dirName,  " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass

        D_max = params[2]

        dirName = p.dir_output + p.exp_id + '/D' + str(int(D_max))
        try:
            makedirs(dirName)
            if p.verbose:
                print("Directory ", dirName, " Created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass

        i = 0
        for experiment in range(p.nran):
            i += 1
            dirName = p.dir_output+p.exp_id + '/D' + str(int(D_max))+'/'
            filename = dirName + str(idx).zfill(5) + '_'
            filename = filename + str(i).zfill(3) + '.pk'

            if(path.isfile(filename)):
                if(p.overwrite):
                    self.run_simulation(p, params)
                    pickle.dump(self.MPL, open(filename, "wb"))
                else:
                    continue
            else:
                self.run_simulation(p, params)
                pickle.dump(self.MPL, open(filename, "wb"))

    def run_experiment_param_set(self, params):
        """Make experiment.

        Requires a single value of parameters.
        Writes output on a file

        Args:
            params (list): A list containing all parameters for the
            simulation

        Raises:
            None

        Returns:
            None
        """
        from os import makedirs, path
        import pandas

        p = self.conf.p

        try:
            dirName = p.dir_output + p.exp_id+''
            makedirs(dirName)
            if p.verbose:
                print("Directory ", dirName, " Created ")
        except FileExistsError:
            print("Directory ", dirName, " already exists")

        Dl = list(map(list, zip(*params)))[2]
        D_max_names = [str(int(d)) for d in Dl]
        D_maxS = list(set(D_max_names))

        for d in D_maxS:
            dirName = p.dir_output + p.exp_id + '/D' + str(int(d))
            try:
                makedirs(dirName)
                if p.verbose:
                    print("Directory ", dirName, " Created ")
            except FileExistsError:
                print("Directory ", dirName, " already exists")

        df = pandas.DataFrame(columns=['tau_awakening', 'tau_survive',
                                       'D_max', 'name'])
        if p.showp:
            bf1 = "{desc}: {percentage:.4f}%|{bar}|"
            bf2 = "{n_fmt}/{total_fmt} ({elapsed}/{remaining})"
            bf = ''.join([bf1, bf2])
            iterator = tqdm(params, bar_format=bf)
        else:
            iterator = params

        k = 0
        j = 0
        for pp in iterator:
            (tau_awakening, tau_survive, D_max) = pp
            pars = list(pp)
            k += 1
            i = 0
            for experiment in range(p.nran):

                i += 1
                j += 1

                dirName = p.dir_output+p.exp_id + '/D' + str(int(D_max))+'/'
                filename = dirName + str(k).zfill(5) + '_'
                filename = filename + str(i).zfill(3) + '.pk'
                df.loc[j] = [tau_awakening, tau_survive, D_max, filename]

                if(path.isfile(filename)):
                    if(p.overwrite):
                        self.run_simulation(p, pars)
                        pickle.dump(self.MPL, open(filename, "wb"))
                    else:
                        continue
                else:
                    self.run_simulation(p, pars)
                    pickle.dump(self.MPL, open(filename, "wb"))
                    # print(f'warning: directory {dirName} does not exist!')

            fn = ''.join([p.dir_output, p.exp_id, '/',
                          self.conf.filenames.progress_root, '.csv'])
            with open(fn, 'a') as file:
                w = f"{tau_awakening}, {tau_survive}, {D_max}, {filename}\n"
                file.write(w)

        self.param_set = df

        # write files
        fn = self.conf.filenames
        fname = fn.dir_output + '/' + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df.to_csv(fname, index=False)

    def run_simulation(self, p=None, pars=None):
        """Make experiment.

        A single value of parameters

        Args:
            p ():
            pars ():

        Raises:
            None

        Returns:
            list of parameters as a named tuple
        """
        if p is None:
            p = self.conf.p
        if pars is None:
            ps = self.conf.p
            tau_awakening = ps.tau_a_min
            tau_survive = ps.tau_s_min
            D_max = ps.d_max_min
        else:
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

        # lista de tiempos de eventos futuros: ordered list
        # [time, ID_emit, ID_receive, case]
        t_forthcoming = OrderedList()

        # estructura de arbol para buscar vecinos
        if 'tree' in locals():
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
                    if 'tree' in locals():
                        try:
                            idx = tree.query_ball_point(query_point, r=D_max)
                        except NameError:
                            pass
                    else:
                        idx = []

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

    def show_single_ccns(self):
        """Show simulation results.

        Args:
            None
        """
        CETIs = self.MPL

        for i in range(len(CETIs)):
            print('%2d         (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr' %
                  (CETIs[i][0][1], CETIs[i][0][4],
                   CETIs[i][0][5], CETIs[i][0][2], CETIs[i][0][3]))

            k = len(CETIs[i]) - 1
            for j in range(k):
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) -
                    np.array(CETIs[i][j+1][2:4]))**2).sum())

                print('%2d sees %2d (%5.0f, %5.0f) yr      \
                      <%5.0f, %5.0f> lyr distance=%f' % (CETIs[i][j+1][0],
                                                         CETIs[i][j+1][1],
                                                         CETIs[i][j+1][4],
                                                         CETIs[i][j+1][5],
                                                         CETIs[i][j+1][2],
                                                         CETIs[i][j+1][3], Dx))


class results():
    """results: load and visualize results from simulations and experiments.

    description
    """

    def __init__(self, conf=None):
        """Instantiate a results object.

        Args:
            GalNet (GalacticNetwork class)
        """
        # super().__init__()
        self.params = dict()
        self.conf = conf

    def load(self):
        """Load parameter set and data.

        Load all data generated from an experiment.
        """
        fn = self.conf.filenames
        fname = fn.dir_output + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df = pd.read_csv(fname)
        self.params = df

    def redux_1d(self, subset=None):
        """Reddux experiment.

        Similar to previous
        """
        import pickle
        import numpy as np

        if subset is None:
            D = self.params
        else:
            D = self.params[subset]

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

                for j in range(1, k):  # traverse contacts

                    earlier = CETIs[i][j][4] - CETIs[i][0][4]
                    firstcontact = min(earlier, firstcontact)
                    Dx = np.sqrt(((
                        np.array(CETIs[i][0][2:4]) -
                        np.array(CETIs[i][j][2:4]))**2).sum())

                    waiting.append(earlier)
                    distancias.append(Dx)
                    hangon.append(CETIs[i][j][5] - CETIs[i][j][4])

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

    def plot_1d(self, key):
        """Plot the histogram of a quantity in the experiment.

        """
 

    def redux_2d(self, show_progress=False):
        """Reddux experiment.

        Similar to previous
        """
        import pickle
        import numpy as np

        # parameters
        p = self.conf.p
        tau_awakeningS = np.linspace(p.tau_a_min, p.tau_a_max, p.tau_a_nbins)
        tau_surviveS = np.linspace(p.tau_s_min, p.tau_s_max, p.tau_s_nbins)

        A = tau_awakeningS
        S = tau_surviveS

        N1 = len(tau_awakeningS)
        N2 = len(tau_surviveS)
        m1_d1 = np.zeros((N1, N2))
        m2_d1 = np.zeros((N1, N2))

        l0_d1 = self.params['D_max'] == self.params['D_max'][0]

        toolbar_width = 40

        print(self.params.keys())

        for i, a in enumerate(A):
            if p.verbose:
                print("%2.2d/%2.2d" % (i+1, N1))
            l1 = abs(self.params['tau_awakening']-a) < 1.e-5

            if show_progress:
                sys.stdout.write("[%s]" % (" " * toolbar_width))
                sys.stdout.flush()
                sys.stdout.write("\b" * (toolbar_width+1))
            for j, s in enumerate(S):
                if show_progress:
                    sys.stdout.write("-")
                    sys.stdout.flush()

                l2 = abs(self.params['tau_survive']-s) < 1.e-5

                cond = l0_d1 & l1 & l2

                if len(cond) > 0:

                    D = self.redux_1d(subset=cond)

                    # awaken = D['A']
                    inbox = D['inbox']
                    # distancias = D['dist']
                    # hangon = D['hangon']
                    # waiting = D['w']
                    # count = D['count']
                    # index = D['index']
                    firstc = D['c1']
                    # ncetis = D['n']
                    # x = D['x']
                    # y = D['y']

                    m1_d1[i][j] = inbox.count(0)/max(len(inbox), 1)
                    m2_d1[i][j] = firstc.count(0.)/max(len(firstc), 1)
                else:
                    m1_d1[i][j] = 0.
                    m2_d1[i][j] = 0.

            if show_progress:
                sys.stdout.write("]\n")  # this ends the progress bar

        m1_d1 = np.transpose(m1_d1)
        m2_d1 = np.transpose(m2_d1)

        fn = self.conf.filenames
        fname = fn.dir_output + fn.exp_id
        fname1 = fname + '/m1.pk'
        fname2 = fname + '/m2.pk'

        print(fname1)
        print(fname2)

        pickle.dump(m1_d1, open(fname1, 'wb'))
        pickle.dump(m2_d1, open(fname2, 'wb'))

        return((m1_d1, m2_d1))

    def show_ccns(self, i):
        """Show simulation results.

        Args:
            None
        """
        filename = self.params.loc[i][3]
        try:
            CETIs = pickle.load(open(filename, "rb"))
        except EOFError:
            CETIs = []

        for i in range(len(CETIs)):
            print('%2d         (%5.0f, %5.0f) yr      <%5.0f, %5.0f> lyr' %
                  (CETIs[i][0][1], CETIs[i][0][4],
                   CETIs[i][0][5], CETIs[i][0][2], CETIs[i][0][3]))

            k = len(CETIs[i]) - 1
            for j in range(k):
                Dx = np.sqrt(((
                    np.array(CETIs[i][0][2:4]) -
                    np.array(CETIs[i][j+1][2:4]))**2).sum())

                print('%2d sees %2d (%5.0f, %5.0f) yr      \
                      <%5.0f, %5.0f> lyr distance=%f' % (CETIs[i][j+1][0],
                                                         CETIs[i][j+1][1],
                                                         CETIs[i][j+1][4],
                                                         CETIs[i][j+1][5],
                                                         CETIs[i][j+1][2],
                                                         CETIs[i][j+1][3], Dx))
