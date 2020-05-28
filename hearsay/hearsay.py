"""HEARSAY.

This module contains tools to compute and analyze numerical simulations of
a Galaxy with constrained causally connected nodes. It simulates a 2D
simplified version of a disk galaxy and perform discrete event simulations
to explore three parameters:
1. the mean time for the appeareance of new nodes,
2. the mean lifetime of the nodes, and
3. the maximum reach of signals.

A simulation is a realization of the Constrained Causally Connected Network
(C3Net) model. The details of this model are explained in
Lares, Gramajo & Funes (under review).

Classes in this module:
- Parser
- C3Net
- Results

Additionally, it contains the function unwrap_run which is used for parallel
runs with the joblib library.
"""

import numpy as np
from configparser import ConfigParser
import itertools
import pandas as pd
import pickle
import sys
from tqdm import tqdm
from hearsay.olists import OrderedList


class Parser(ConfigParser):
    """parser class.

    Manipulation of configuration parameters. This method allows to read a
    configuration file or to set parameters for a Constrained Causally
    Conected Network (C3Net) model.
    """

    def __init__(self, argv=None, *args, **kwargs):
        """Initialize a parser.

        Parameters
        ----------
            None
        Returns
        -------
            None
        Raises
        ------
            None
        """
        super().__init__()
        self.message = None
        self.check_file(argv)
        self.read_config_file()

        self.load_filenames()
        self.load_config(*args, **kwargs)
        self.check_settings()

    def check_file(self, sys_args=""):
        """Parse paramenters for the simulation from a .ini file.

        Parameters
        ----------
            filename (str): the file name of the map to be read

        Raises
        ------
            None

        Returns
        -------
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

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            None
        """
        self.read(self.filename)

    def load_filenames(self):
        """Make filenames based on info in config file.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
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

    def load_config(self, keys=None, values=None, nran=None,
                    *args, **kwargs):
        """Load parameters from config file.

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            list of parameters as a named tuple
        """
        if isinstance(keys, list):
            # override configuration file with arguments
            if len(keys) != len(values):
                print('Error overriding parameters (using file values)')
            else:
                for k, v in zip(keys, values):
                    for sec in self.sections():
                        has = self.has_option(sec, k)
                        if has:
                            self[sec][k] = v

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
            run_parallel = False

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

        Parameters
        ----------
            None

        Raises
        ------
            None

        Returns
        -------
            Exception if settings have inconsistencies.
        """
        from os import path, makedirs

        if self.p.verbose:
            print(self.message)
            print('Checking settings...')

        # output directory
        if not path.isdir(self.p.dir_output):
            print(f"Directory {self.p.dir_output} does not exist")

            try:
                makedirs(self.p.dir_output)
                if self.p.verbose:
                    print("Directory ", self.p.dir_output,  " Created ")
            except FileExistsError:
                # directory already exists
                pass

        # experiment directory
        ID_dir = self.p.dir_output + self.p.exp_id
        if not path.isdir(ID_dir):
            print(f"Directory {ID_dir} does not exist")

            try:
                makedirs(ID_dir)
                if self.p.verbose:
                    print("Directory ", ID_dir,  " Created ")
            except FileExistsError:
                # directory already exists
                pass

        # plots directory
        if not path.isdir(self.p.dir_plots):
            print(f"Directory {self.p.dir_plots} does not exist")

            try:
                makedirs(self.p.dir_plots)
                if self.p.verbose:
                    print("Directory ", self.p.dir_plots,  " Created ")
            except FileExistsError:
                # directory already exists
                pass


def unwrap_run(arg, **kwarg):
    """Wrap the serial function for parallel run.

    This function just call the serialized version, but allows to run
    it concurrently.
    """
    return C3Net.run_suite(*arg, **kwarg)


class C3Net():
    """C3Net: Constrained Causally Connected Network model.

    methods:
        init:
            creates a node
        __len__:
            None
        __repr__:
            None
        __str__:
            None
        run:
            Run a suite of simulations for the full parametet set in
            the configuration file.
        run_suite:
            Run a suite of simulations for a given parameter set.
        run_suite_II:
            Run a suite of simulations for a given parameter set, to
            be run in parallel.
        run_simulation:
            Run a simulation for a point in parameter space.
        show_single_ccns:
            Show the contents of a simulation run.
    """

    def __init__(self, conf=None):
        """Instantiate Galaxy object.

        Parameters
        ----------
            None
        """
        self.params = None
        self.config = conf

    def __len__(self):
        """Return the number of contacts.

        Parameters
        ----------
            None
        """
        pass

    def __repr__(self):
        """Represent with a string.

        Parameters
        ----------
            None
        """
        print('message')

    def __str__(self):
        """Represent with a string.

        Parameters
        ----------
            None
        """
        print('message')

    def set_parameters(self, spars=None,
                       A=None, S=None, D=None,
                       write_file=False):
        """Set parameters for the experiment.

        If no arguments are given, the parameters are set from the ini file.
        Parameters
        ----------
        spars (dataframe, list or string, optional):
        Parameters to set the experiment.
        If spars is a pandas DataFrame, it must contain the keys:
        ['tau_awakening', 'tau_survive', 'd_max', 'filename'].
        If spars is a list, it must have length=4, comprisong the
        tau_awakening, tau_survive, d_max, and filename lists.
        If spars is a string, a file with that name will be read.
        The file must contain the same four columns, with the names.
        A (number or list, optional): Values of the tau_awakening parameter
        S (number or list, optional): Values of the tau_survive parameter
        D (number or list, optional): Values of the D_max parameter
        write_file (optional): filename to write the parameter set.
        """
        p = self.config.p

        if spars is None:
            tau_awakeningS = np.linspace(p.tau_a_min, p.tau_a_max,
                                         p.tau_a_nbins)
            tau_surviveS = np.linspace(p.tau_s_min, p.tau_s_max, p.tau_s_nbins)
            D_maxS = np.linspace(p.d_max_min, p.d_max_max, p.d_max_nbins)
        else:
            if isinstance(spars, pd.DataFrame):
                tau_awakeningS = spars['tau_awakening']
                tau_surviveS = spars['tau_survive']
                D_maxS = spars['D_max']
                # filenames = spars['filename']
            elif isinstance(spars, list):
                tau_awakeningS = spars[0]
                tau_surviveS = spars[1]
                D_maxS = spars[2]
            else:
                print('warning: a dataframe or list expected for spars')
                pass

        if A is not None:
            tau_awakeningS = A
        if S is not None:
            tau_surviveS = S
        if D is not None:
            D_maxS = D

        df = pd.DataFrame(columns=['tau_awakening', 'tau_survive',
                                   'D_max', 'filename'])
        if isinstance(spars, pd.DataFrame):
            if p.verbose:
                print('parameters dataframe detected')
            df['tau_awakening'] = spars['tau_awakening']
            df['tau_survive'] = spars['tau_survive']
            df['D_max'] = spars['D_max']
            df['filename'] = spars['filename']
        elif isinstance(spars, list):
            if p.verbose:
                print('parameters list deteted')
            params = []
            prd = itertools.product(tau_awakeningS, tau_surviveS, D_maxS)
            for i in prd:
                params.append(i)
            k = 0
            j = 0
            for pp in params:
                (tau_awakening, tau_survive, D_max) = pp
                k += 1
                i = 0
                for experiment in range(p.nran):
                    i += 1
                    j += 1
                    dirName = p.dir_output+p.exp_id + '/D' + str(int(D_max))
                    filename = dirName + '/' + str(k).zfill(5) + '_'
                    filename = filename + str(i).zfill(3) + '.pk'
                    df.loc[j] = [tau_awakening, tau_survive, D_max, filename]
        elif isinstance(spars, str):
            if p.verbose:
                print('parameters file detected')
            df = pd.read_csv(spars)
        elif spars is None:
            print('default action: load from config file')
            params = []
            prd = itertools.product(tau_awakeningS, tau_surviveS, D_maxS)
            for i in prd:
                params.append(i)
            k = 0
            j = 0

            A = []
            S = []
            D = []
            F = []

            outdir = p.dir_output + p.exp_id + '/D'
            for pp in params:
                (tau_awakening, tau_survive, D_max) = pp
                k += 1
                i = 0
                for experiment in range(p.nran):
                    A.append(tau_awakening)
                    S.append(tau_survive)
                    D.append(D_max)

                    i += 1
                    j += 1
                    parts = [outdir, str(int(D_max)), '/', str(k).zfill(5),
                             '_', str(i).zfill(3) + '.pk']
                    filename = ''.join(parts)

                    F.append(filename)

            df['tau_awakening'] = A
            df['tau_survive'] = S
            df['D_max'] = D
            df['filename'] = F
        else:
            if spars is not None:
                print('spars must be dataframe, list or string')
        # write files list
        fn = self.config.filenames
        fname = fn.dir_output + '/' + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df.to_csv(fname, index=False)

        self.params = df

    def run(self, parallel=False, njobs=None, interactive=False):
        """Run an experiment.

        An experiment requires a set of at least three parameters, which are
        taken from the configuration file.

        Parameters
        ----------
        parallel : Boolean
            Flag to indicate if run is made using the parallelized version.
            Default: False.
        njobs : int
            Number of concurrent jobs for the parallel version.
            If parallel is False njobs is ignored.
        interactive : boolean
            Flag to indicate if the result of the simulation suite is returned
            as a variable.

        Returns
        -------
        res: list
            Only returned if interactive=True.
            Contains the results from the simulations.  The size of the
            list is the number of simulations in the experiment, i.e., the
            number of lines in self.params.
            Each element of the list is a dictionary containing the complete
            list of CCNs and their contacts.

        See also
        --------
        hearsay.results.ccn_stats

        Example
        -------
        If the following experiment is set:
        >>> conf.load_config(['nran'], ['2']
        >>> A = [5000, 10000, 20000]
        >>> S = [20000]
        >>> D = [20000]
        >>> G.set_parameters(A=A, S=S, D=D)
        then a total of 6 experiments will be performed.  The result of this
        function is a list of length 6, each element containing an element that
        can be printed with the show_single_ccns method. See that method for
        more details.
        """
        if njobs is not None:
            parallel = True

        if parallel:
            if njobs is None:
                njobs = self.config.p.njobs
            if interactive:
                res = self.run_suite_II(njobs, interactive)
            else:
                self.run_suite_II(njobs)
        else:
            if interactive:
                res = self.run_suite(interactive)
            else:
                self.run_suite()

        if interactive:
            return res
        else:
            return None

    def run_suite_II(self, njobs, interactive=False):
        """Run an experiment, parallel version.

        An experiment requires a set of at least three parameters, which are
        taken from the configuration file.

        Parameters
        ----------
        params: the parameters
        njobs: number of jobs
        """
        from joblib import Parallel, delayed

        Pll = Parallel(n_jobs=njobs, verbose=5, prefer="processes")
        params = self.params.values.tolist()
        ids = np.array(range(len(params))) + 1
        ntr = [interactive]*len(params)
        z = zip([self]*len(params), params, ids, ntr)
        d_experiment = delayed(unwrap_run)
        results = Pll(d_experiment(i) for i in z)

        df = pd.DataFrame(columns=['tau_awakening', 'tau_survive',
                                   'D_max', 'filename'])

        p = self.config.p
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
        fn = self.config.filenames
        fname = fn.dir_output + '/' + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df.to_csv(fname, index=False)

        if interactive:
            return results
        else:
            return None

    def run_suite(self, interactive=False):
        """Make experiment.

        Requires a single value of parameters.
        Writes output on a file

        Parameters
        ----------
            params (list): A list containing all parameters for the
            simulation.  Format, e.g.: [(A1,S1,D1), (A2,S2,D2)]

        Raises
        ------
            None

        Returns
        -------
            None
        """
        from os import makedirs, path

        p = self.config.p
        params = self.params.values.tolist()

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

        if p.showp:
            bf1 = "{desc}: {percentage:.4f}% | "
            bf2 = "{n_fmt}/{total_fmt} ({elapsed}/{remaining})"
            bf = ''.join([bf1, bf2])
            iterator = tqdm(params, bar_format=bf)
        else:
            iterator = params

        results = []
        for pp in iterator:
            (tau_awakening, tau_survive, D_max, filename) = pp
            pars = list(pp)
            if path.isfile(filename):
                if p.overwrite:
                    self.run_simulation(p, pars)
                    pickle.dump(self.MPL, open(filename, "wb"))
                elif interactive:
                    self.run_simulation(p, pars)
                    MPL = self.MPL
                    results.append(MPL)
            else:
                self.run_simulation(p, pars)
                pickle.dump(self.MPL, open(filename, "wb"))
                if interactive:
                    MPL = self.MPL
                    results.append(MPL)

            fn = ''.join([p.dir_output, p.exp_id, '/',
                          self.config.filenames.progress_root, '.csv'])
            with open(fn, 'a') as file:
                w = f"{tau_awakening}, {tau_survive}, {D_max}, {filename}\n"
                file.write(w)

        if interactive:
            return results
        else:
            return None

    def run_simulation(self, p=None, pars=None):
        """Make experiment.

        A single value of parameters

        Parameters
        ----------
        p (configuration object) : configuration object
        pars (list) : list of (3) parameters:
            tau_A, tau_S and D_max

        Raises
        ------
            None

        Returns
        -------
        MPL : dict

           (ID of CCN,
            ID of CCN (repeated),
            x,
            y,
            time of A event,
            time of D event)
        Moreover, if there are contacts:
           (ID of receiving CCN,
            ID of emiting CCN,
            x of emiting CCN,
            y of emiting CCN,
            time of C event,
            time of B event)
        """
        if p is None:
            p = self.config.p
        if pars is None:
            ps = self.config.p
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

    def show_single_ccns(self, MPL=None, interactive=False):
        """Show simulation results.

        Parameters
        ----------
            None
        """
        if MPL is None:
            CETIs = self.MPL
        else:
            CETIs = MPL

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
        if interactive:
            return CETIs


class Results(C3Net):
    """results: load and visualize results from simulations and experiments.

    description
    """

    """
    To do:
    - number of contacts
    """

    def __init__(self, G=None):
        """Instantiate a results object.

        Parameters
        ----------
        G: C3Net object
            An object containing all the data about the simulation suite.
        """
        self.params = dict()
        self.config = tuple()
        if G is not None:
            self.params = G.params
            self.config = G.config

    def load(self):
        """Load parameter set and data.

        Load all data generated from an experiment.
        """
        fn = self.config.filenames
        fname = fn.dir_output + fn.exp_id
        fname = fname + '/' + fn.pars_root + '.csv'
        df = pd.read_csv(fname)
        self.params = df

    def ccn_stats(self, CCN):
        """Return statistics for a single causally connected network.

        This corresponds to a single simulation run, that gives a list of
        nodes, its properties and its contacts.  The properties of a node are
        the ID, the times of the A and D events and the postition in the
        (simulated) Galaxy.

        Parameters
        ----------
        CCN : dict
            An object (as read from pickle files) that represents a network of
            CCNs from a single simulation run

        Returns
        -------
        stats : tuple
            A tuple containing several statistics about the network.


        Notes
        -----
            The stats tuple includes parameters with counters (1, 2, 3),
            parameters with CCNs values (4-7) and
            parameters with contacts values (8-11)

            01. N : Total number of CCNs in the full period. Length=1

            02. M : Total number of contacts (i.e., CCNs that are on the
            space-time cone of another CCN.)

            03. K : Total number of CCNs that make at least one contact
            (i.e., CCNs that are on the space-time cone of at least
            another CCN.)

            04. lP : Time periods for each CCN.  Equivalent to the time span
            between the A and D events. Length=N

            05. lI : Number of contacts each CETI receives. Length=N

            06. lX : X position within the Galaxy disc. Length=N

            07. lY : Y position within the Galaxy disc. Length=N

            08. lL : Distances between contacted nodes. Length=K

            09. lH : Duration of each contact. Length=K

            10. lW : Time elapsed from awakening to contact. Length=K

            11. lF : Time elapsed from awakening to the first contact.
        """
        N = len(CCN)
        M = 0
        K = 0

        lP = []
        lI = []
        lX = []
        lY = []
        lL = []
        lH = []
        lW = []
        lF = []

        for i in range(N):

            k = len(CCN[i])
            lI.append(k-1)
            lP.append(CCN[i][0][5] - CCN[i][0][4])
            lX.append(CCN[i][0][2])
            lY.append(CCN[i][0][3])

            firstcontact = 1.e8

            for j in range(1, k):  # traverse contacts

                earlier = CCN[i][j][4] - CCN[i][0][4]
                firstcontact = min(earlier, firstcontact)
                Dx = np.sqrt(((
                    np.array(CCN[i][0][2:4]) -
                    np.array(CCN[i][j][2:4]))**2).sum())

                lW.append(earlier)
                lL.append(Dx)
                lH.append(CCN[i][j][5] - CCN[i][j][4])

            if k > 1:
                lF.append(firstcontact)

        return (N, M, K), (lP, lI, lX, lY, lL, lH, lW, lF)

    def redux(self, subset=None):
        """Redux experiment.

        Given a set of parameters, returns the global values
        """
        import pickle

        if subset is None:
            D = self.params
        else:
            D = self.params[subset]

        N = []
        M = []
        K = []

        lP = []
        lI = []
        lX = []
        lY = []
        lL = []
        lH = []
        lW = []
        lF = []

        for filename in D['filename']:

            try:
                CETIs = pickle.load(open(filename, "rb"))
            except EOFError:
                CETIs = []

            t1, t2 = self.ccn_stats(CETIs)

            N.append(t1[0])
            M.append(t1[1])
            K.append(t1[2])
            lP.append(t2[0])
            lI.append(t2[1])
            lX.append(t2[2])
            lY.append(t2[3])
            lL.append(t2[4])
            lH.append(t2[5])
            lW.append(t2[6])
            lF.append(t2[7])

        return({
            'N': N,
            'M': M,
            'K': K,
            'lP': lP,
            'lI': lI,
            'lX': lX,
            'lY': lY,
            'lL': lL,
            'lH': lH,
            'lW': lW,
            'lF': lF})

    def redux_1d(self, subset=None, applyto=None):
        """Reddux experiment.

        Compute statistics for the set of parameters limited to subset

        Parameters
        ----------
        subset: logical array
            Filter for the full parameter set.
        applyto: string
            Name of the variable to be used as X-axis
        Results
        -------

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

        for filename in D['filename']:

            try:
                CETIs = pickle.load(open(filename, "rb"))
            except EOFError:
                CETIs = []

            M = len(CETIs)
            ncetis.append(M)

            for i in range(M):

                k = len(CETIs[i])
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

    def redux_2d(self, show_progress=False):
        """Reddux experiment.

        Similar to previous
        """
        import pickle
        import numpy as np

        # parameters
        p = self.config.p
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

        fn = self.config.filenames
        fname = fn.dir_output + fn.exp_id
        fname1 = fname + '/m1.pk'
        fname2 = fname + '/m2.pk'

        print(fname1)
        print(fname2)

        pickle.dump(m1_d1, open(fname1, 'wb'))
        pickle.dump(m2_d1, open(fname2, 'wb'))

        return((m1_d1, m2_d1))

    def show_ccns(self, i, interactive=False):
        """Show simulation results.

        Parameters
        ----------
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
        if interactive:
            return CETIs
