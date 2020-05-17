***************
Getting started
***************

Hearsay for python
===============================

Hearsay has been tested for python 3.7

More testing is currently under development.


Preparing a virtual environment
===============================

It is recommended to install a virtual environment for a clean python ecosystem.

.. code-block::

    virtualenv MyVE
    source MyVE/bin/activate

or 

mkvirtualenv -p $(which python3)

Downloading hearsay
===============================

Hearsay is publically available from a GitHub repository.  It can be downloaded with::

    git clone https://github.com/mlares/hearsay.git

The code can be explored using GitHub, including development activity and documentation.

Installing hearsay
===============================

Once the virtualenvironment has been set (recommended), then install the hearsay package::

    pip install -r requirements.txt

It is convenient to save the root directory of the hearsay installation.  
In bash, for example,

export hearsat_rootdir="$(pwd)"


Hearsay module can be used anywhere provided the following command 
is executed within the environment in the directory $hearsay_rootdir::

    pip install .

Testing hearsay
===============================

We first need to create an output directory, as set in the .ini file::

    dir_output = ../out/

So, from a bash prompt:

mkdir $hearsay_root/out


In order to run a test experiment, go to the ``src`` directory and run::

    python run_experiment.py ../set/experiment.ini


