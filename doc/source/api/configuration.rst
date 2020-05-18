**********************
Configuration
**********************

Experiment section
========================

In this section we must give a unique ID for an experiment.  It can include
numbers and characters, for example "01" or "first_experiment"

.. code-block::

   [experiment]

   # experiment ID
   exp_ID =  ID_001

The files resulting from the simulations will be stored in a directory with this name
under the directory indicated in the ``dir_output`` varible (output section).
In this case, it will create the directory ``../out/ID_001`` if it does not exist.


Simulation section
========================

.. code-block::

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

These variables are used to set:

* the inner and outer radii of the Galactic Habitable Zone (GHZ_inner and GHZ_outer)
* the maximum time span for a simulation (t_max), in years
* the minimum and maximum values for the tau_awakening parameter, in kpc, 
  and the number of values in that range.  For examples, the values: 
  tau_a_min = 3, tau_a_max = 8 and tau_a_nbins = 10 will generate ten values 
  between 3 and 8 using the same criteria as numpy.linspace, i.e., 
  tau_survive_values = numpy.linspace(tau_a_min, tau_a_max, tau_a_nbins)
* the minimum and maximum values for the tau_survive parameter, in years, 
  and the number of values in that range.
* the minimum and maximum values for the D_max paramter, in kpc, and the number 
  of values in that range.
* A flag indicating whether the run will be made in parallel (run_parallel).
  The values 'y', 'Y', 'yes', 'YES', 'true', 'True', or 'TRUE' can be used to
  set a parallel run, and the values 'n', 'N', 'no', 'NO', 'No', False', 'false',
  or 'FALSE' can be used to ser a serial run.
* If a parallel run has been set, the parameter Njobs can be used to set the
  number of threads. If run_parallel has been set to False, Njobs will be ignored.


Output section
========================

.. code-block::

   [output]

   dir_output = ../out/
   dir_plots = ../plt/
   pars_root = params
   progress_root = progress

   plot_fname = plot
   plot_ftype = PNG
   clobber = N

These are the names of the directories for output and plots (fir_output and dir_plots, resp.).
pars_root is used as the root name for the file that stores the parameters.  For example, this 
configuration will generate a file named params.csv in the directory ../out/ID_001/.

If the sme experiment ID is used twice, it will ignore the files with the same names.
The clobber variable allows to chose if these files will be overwritten (True, Yes) 
or not (False, No).

Verbose section
========================

.. code-block::

   [UX]

   show_progress = Y
   verbose = Y

- show_progress can be set to True/False or Y/N (similarly to the run_parallel variable),
  used to show a progress bar for the experiment.
- verbose will print on STDOUT several messages indicating the steps of the simulation.



