*********
API Usage
*********

This tools can be used as an API, from a python prompt or from a command line.

Steps:

1. Complete the configuration of the experiment
2. All the settings of the experimets are parsed from the configuration files using configparser.
3. run an experiment
4. process the experiment
5. show results


Configuration files
===================

All parameters for the simulation are stored in a .ini file, for
example:

.. code-block::

   [experiment]

   # experiment ID
   exp_ID =  PHLX_02


   [simu]

   # internal GHZ radius, lyr
   GHZ_inner = 20000.
   # external GHZ radius, lyr
   GHZ_outer = 60000.  

   # time span to simulate
   t_max = 2.e6
   
   #tau_awakeningS
   tau_a_min = 2000
   tau_a_max = 80000
   tau_a_nbins = 10

   # tau_surviveS
   tau_s_min = 2000
   tau_s_max = 80000
   tau_s_nbins = 10

   # Separate data in directories according to D_max
   D_max_min = 20000
   D_max_max = 20000
   D_max_nbins = 1

   # Number of realizations for each simulation
   Nran = 10

   # Parallel run
   run_parallel = Y 
   Njobs = 10


   [output]

   dir_output = ../out/
   dir_plots = ../plt/
   pars_root = params
   progress_root = progress

   plot_fname = plot
   plot_ftype = PNG
   clobber = N

   [UX]

   show_progress = Y
   verbose = Y

It is not possible to add, remove or change fields from this file, if
so it will trigger failures in the testing process.  The fields are
organized in four categories:

:experiment:
   for the experiment ID.  Each time a new ID is used, it will generate
   a new directory in the ``dir_output`` directory.
:simu: for simulation parameters
:output: for the names of directories and files that store simulation
         results.
:UX: parameters related to the user experience

An example file is provided in the ``set`` directory.


Command line usage
==================

For a simple test, go to ``src`` and run:

.. code-block::

   $ python run_experiment.py ../set/experiment.ini
   $ python process_experiment.py ../set/experiment.ini
   $ python plot_experiment.py ../set/experiment.ini

It is important to use the same configuration file on these three steps, 
since parameters are used to build filenames.

If no name is given for a configuration file, it will use a dafault file
in ``../set/experiment.ini``.

API usage
==================

To use functionalities, import the :class:`hearsay` module:

.. code-block:: python

   from hearsay import hearsay


First, we must parse the configuration parameters from the .ini file.

All parameters with an assigned value must be read with the 
`configparser <https://docs.python.org/3/library/configparser.html>`_
module.   The ConfigParser class is inherited in :class:`hearsay.parser`.

Variables can be accessed using the names of the sections and the
names of the fields.  For example, conf['simu']['t_max'].

There are several posibilities for loading the configuration
parameters.

From the command line it is possible to give the name of the file
containing the parameter settings::

   python run_experiment.py < ../set/experiment.ini

In this case, the file must contain the following::

   from sys import argv
   conf = hearsay.parser(argv)

From the python interface, it is possible to give the filename as a
string:

.. code-block:: python

   from hearsay import hearsay
   conf = hearsay.parser('../set/experiment.ini')

Also, in the default case, the function ``hearsay.parser`` can be
called without arguments, and the default configuration file will be
loaded:

.. code-block:: python

   from hearsay import hearsay
   conf = hearsay.parser()

After the instantiation of a parser object without arguments, the
default parameters can be overwritten with the specific methods:

.. code-block:: python

   from hearsay import hearsay

   conf = hearsay.parser()
   conf.check_file('../set/experiment.ini')
   conf.read_config_file()
   conf.load_filenames()
   conf.load_parameters()
    

Finally, the simulation is made with the
:class:`hearsay.GalacticNetwork`
class, where the function :meth:`hearsay.GalacticNetwork.run_experiment` makes
the computations.


.. code-block:: python

   G = hearsay.GalacticNetwork(conf)
   G.run_experiment()

This function accepts the ``parallel`` flag which indicates 
a parallel version of the code will run::

   G.run_experiment(parallel=True)


The analysis and visualization of the results can be done as follows:

.. code-block:: python

   R = hearsay.results(conf)
   R.load()
   res = R.redux_2d()

The method ``hearsay.results.redux_2d`` computes the matrices.


A complete example of visualization is provided in the ``src`` directory.




