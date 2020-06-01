*********
Tutorials
*********

=======================
Contents of objects
=======================


C3Net.params
===========================

A pandas dataframe with the columns:

- tau_awakening
- tau_survive
- d_max
- filename



C3Net.config
===========================

This is a parser object, which is part of the :class:`hearsay.hearsay.Parser`
method.

Contains all configuration parameters in
:attr:`hearsay.C3Net.config.p`::

   G.config.p
      Out: pars(ghz_inner=20000.0, ghz_outer=60000.0, t_max=2000000.0, 
      tau_a_min=2000.0, tau_a_max=80000.0, tau_a_nbins=10, 
      tau_s_min=2000.0, tau_s_max=80000.0, tau_s_nbins=10, 
      d_max_min=20000.0, d_max_max=20000.0, d_max_nbins=1, 
      nran=3, run_parallel=True, njobs=10, exp_id='PHLX_02', 
      dir_plots='../plt/', dir_output='../out/', pars_root='params', 
      plot_fname='plot', plot_ftype='PNG', fname='../plt/plot_PHLX_02PNG', 
      showp=True, overwrite=False, verbose=True)

:attr:`hearsay.C3Net.config.filenames`::

   G.config.filenames
   pars(exp_id='PHLX_02', dir_plots='../plt/', dir_output='../out/', 
   pars_root='params', progress_root='progress', plot_fname='plot', 
   plot_ftype='PNG', fname='../plt/plot_PHLX_02PNG')


Output of a simulation
===========================

The output of a single simulation is a dictionary.  The length of this
object is the number of nodes in the simulation run.  Each entry has a list
which contains the node itself and the nodes that reach contact to that node.

The first entry of this list contains:

- index of the node
- index of the node (repeated)
- position X
- position Y
- time of the A event
- time of the D event

The next entries of the list, if any, contain the contacts.

- index of the receiver node
- index of the emiter node 
- position X of the emiter node 
- position Y of the emiter node 
- time of the C event for the receiver node
- time of the B event for the receiver node


=================================
Runnig and analyzing experiments
=================================

In this section we show how to use hearsay to run experiments and analyze the 
results.

First, we import the required modules:

.. code-block:: python

   import hearsay
   import pandas as pd
   from matplotlib import pyplot as plt
   import numpy as np

# TUTORIAL 1: experiment from ini file

Now, we use the configuration file to load an experiment setup:

.. code-block:: python

   conf = hearsay.parser('experiment.ini')
   G = hearsay.C3Net(conf)
   G.set_parameters()
   net = G.run(interactive=True)
   R = hearsay.results(conf)
   R.load()
   res = R.redux_1d()
   plt.hist(res['A'])
   plt.show()

# TUTORIAL 2: CORRER UNA SIMULACION

It is possible to run a limited number of parameters of the experiment, 
or even an entirely new parameter set.  For example, if we want the parameters:

tau_awakening = 20000
tau_survive = 20000
D_max = 20000
Nran = 7

we can just update the parameters:

.. code-block:: python

   conf.load_config(['nran'], ['7'])
   tau_awakening = 20000
   tau_survive = 20000
   D_max = 20000
   directory = ''.join([G.config.filenames.dir_output, G.config.filenames.exp_id])
   filename = ''.join([directory, 'test.pk'])
   pars = [[tau_awakening, tau_survive, D_max, filename]]
   df = pd.DataFrame(pars, columns=['tau_awakening', 'tau_survive',
                                    'D_max', 'filename'])
   G.set_parameters(df)


And then we can analyze them using:

.. code-block:: python

   res = G.run(interactive=True)
   G.show_single_ccns(res[0])
