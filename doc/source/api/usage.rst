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
* Install required packages with ::
  
   pip install -r requirements.txt

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
    
   #tau_awakeningS
   tau_a_min = 1000
   tau_a_max = 200000
   tau_a_nbins = 10

   # tau_surviveS
   tau_s_min = 1000
   tau_s_max = 200000
   tau_s_nbins = 10


   # Separate data in directories according to D_max
   #D_maxS
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

:experiment:
   for the experiment ID.  Each time a new ID is used, it will generate
   a new directory in the ``dir_output`` directory.
:simu: for simulation parameters
:output: for the names of directories and files that store simulation
         results.



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

There are several posibilities for loading the configuration
parameters.

From the command line it is possible to give the name of the file
containing the parameter settings::

   python run_experiment.py < ../set/experiment.ini

In this case, the file must contain the following::

   from hearsay import hearsay
   from sys import argv
   conf = hearsay.parser(argv)

From the python interface, it is possible to give the filename as a
string:

.. code-block:: python

   from hearsay import hearsay
   conf = hearsay.parser('../MySettings/MyFile.ini')

Also, in the default case, the function ``hearsay.parser`` can be
called without arguments, and the default configuration file will be
loaded:

.. code-block:: python

   from hearsay import hearsay
   conf = hearsay.parser()


After the instantiation of a parser object without arguments, the
default file can be overwritten with the specific methods:

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

