*********
API Usage
*********

This project is organized as an API to be used from a python prompt.

Steps:

- Complete the configuration of the experiment
- All the settings of the experimets are parsed from the configuration
  files using configparser.


Prerequisites
=============

* A virtual environment is recomended
* Install required packages with pip install -r requirements.txt
* Complete the configuration file



Configuration files
===================

All parameters for the simulation are stored in a .ini file, for
example:

.. code-block::

   [experiment]

   # experiment ID
   exp_ID = SKRU_11


   [simu]

   # radio interno de la zona galactica habitable, años luz
   GHZ_inner = 20000.
   # radio interno de la zona galactica habitable, años luz
   GHZ_outer = 60000.  

   # maximo tiempo para simular
   t_max = 1.e6
    
   #tau_awakeningS = np.linspace(0, 200000, 51)[1:]
   tau_a_min = 1000
   tau_a_max = 200000
   tau_a_nbins = 10

   # tau_surviveS = np.linspace(0, 500000, 51)[1:]
   tau_s_min = 1000
   tau_s_max = 200000
   tau_s_nbins = 10


   # Separate data in directories according to D_max
   #D_maxS = np.linspace(0, 40000, 11)[1::2]
   #D_maxS = [500, 1000., 10000., 20000., 40000, 80000]
   D_max_min = 600
   D_max_max = 3000
   D_max_nbins = 1

   Nran = 10


   [output]

   dir_output = ../out/
   dir_plots = ../plt/
   pars_root = params

   plot_fname = plot
   plot_ftype = PNG


It is not possible to add, remove or change fields from this file, if
so it will trigger fails in the testing process.  The fields are
organized in three categories:

- experiment
  for the experiment ID.  Each time a new ID is used, it will generate
  a new directory in the ``dir_output`` directory.
- simu
- output





Command line usage
==================

For a simple test, go to ``hearsay`` and run:

.. code-block::

   $ python run_experiment.py ../set/config.ini


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



.. code-block:: python

   from hearsay import hearsay
   from sys import argv

   conf = hearsay.parser()
   conf.check_file(argv)
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

