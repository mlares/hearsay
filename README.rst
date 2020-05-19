HEARSAY

Simulations for the probability of alien contact

.. image:: https://readthedocs.org/projects/hearsay/badge/?version=latest
   :target: https://hearsay.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. image:: https://travis-ci.org/mlares/hearsay.svg?branch=master
    :target: https://travis-ci.org/mlares/hearsay


The purpose of this project is to compute simulations of the causal
contacts between emiters in the Galaxy.

Minimal example
***************

The following lines show how to install and run an example simulation
suite.  We assume that the package is downloaded in ``$hearsay_dir``
and the 
working directory is ``$working_dir``


1. Clone hearsay from
   `GitHub <https://github.com/mlares/hearsay.git>`_

   .. code-block::

      cd $hearsay_dir
      git clone https://github.com/mlares/hearsay.git

2. Create a virtual environment for python

   .. code-block::

      virtualenv -p $(which python3) MyVE
      source MyVE/bin/activate

3. Install the hearsay package

   .. code-block::

      cd $hearsay_dir
      pip install .

4. Create a configuration file.  A template can be found in
   ``$hearsay_dir/set/experiment.ini``

   .. code-block::

      cd $working_dir
      cp $hearsay_dir/set/experiment.ini $working_dir

5. Edit the configuration file.  Set the following values:

   .. code-block::

      experiment_ID = run_001
      dir_output = out
      dir_plots = plt

6. Create directories for output and plots, using the same values than
   the variables ``dir_output0``  and ``dir_plots`` in the
   configuration file, for example:

   .. code-block::

      cd $working_dir
      mkdir out
      mkdir plt

7. create a file ``experiment.py`` that contains the following:

   .. code-block::

      from hearsay import hearsay
      from sys import argv
      conf = hearsay.parser(argv)
      G = hearsay.GalacticNetwork(conf)
      G.run_experiment()
      R = hearsay.results(conf)
      R.load()
      res = R.redux_1d()
      R.plot_1d()
                                                      
   A file with the name entered in the variable ``plot_fname`` of the
   configuration file will be saved in the directory ``plt``.

                                              
Results from the simulations are stored in ``out/run_001/`` directory, and can be read as follows:

.. code-block:: python

      from hearsay import hearsay
      conf = hearsay.parser()
      R = hearsay.results(conf)
      R.show_ccns(ind)

where ``ind`` is the simulation index, given by the line number in the file ``out/run_001/params.csv``
   


