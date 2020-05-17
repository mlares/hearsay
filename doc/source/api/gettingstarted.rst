***************
Getting started
***************

Preparing a virtual environment
===============================

``virtualenv MyVE``
``source MyVE/bin/activate``

or 

mkvirtualenv -p $(which python3)

Installing hearsay
===============================

Once the virtualenvironment has been set (recommended), then install the hearsay package::

    pip install -r requirements.txt

It is convenient to save the root directory of the hearsay installation.  
In bash, for example,

export hearsat_rootdir="$(pwd)"

Then we need to create an output directory, as set in the .ini file::

    dir_output = ../out/

So, from a bash prompt:

mkdir $hearsay_root/out

Hearsay module can be used anywhere provided the following command 
is executed within the environment in the directory $hearsay_rootdir::

    pip install .

Testing hearsay
===============================

In order to run a test experiment, go to the ``src`` directory and run::

python run_experiment.py ../set/experiment.ini


