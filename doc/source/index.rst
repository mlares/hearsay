.. hearsay documentation master file, created by
   sphinx-quickstart on Tue Mar  3 16:42:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

################################################
hearsay
################################################

The purpose of this project is to compute simulations of the causal
contacts between emiters in the Galaxy.

Project by Marcelo Lares (CONICET, UNC, Argentina)

A python virtual environment is suggested to work with this project.
Requirements are listed in the project home directory file:
``requirements.txt``.


Science case
***************

.. toctree::
   :maxdepth: 2

   sci/proposal
   sci/references

API
***************

.. toctree::
   :maxdepth: 2

   api/gettingstarted
   api/configuration
   api/usage
   api/tutorial.rst
   api/hearsay


Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

 
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

   conf = hearsay.parser('hearsay_dir/set/experiment.ini')
   G = hearsay.GalacticNetwork(conf)
   G.set_parameters()
   net = G.run(interactive=True)
   R = hearsay.results(conf)
   R.load()
   res = R.redux_1d()
   plt.hist(res['A'])
   plt.show()
    
A file with the name entered in the variable ``plot_fname`` of the
configuration file will be saved in the directory ``plt``.

