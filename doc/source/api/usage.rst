*********
API Usage
*********

- instalation through pypy not yet implemented
- make setup.py installer
- from a python script, call import ccn


This project is organized as an API to be used from a python prompt.

Steps:

- Complete the configuration of the experiment
- All the settings of the experimets are parsed from the configuration
  files using configparser.


Prerequisites
=============

* Put data files on the ``dat`` directory.
* Complete the names of the data files in the configuration file



Configuration files
===================


.. code-block::

   [simu]

   # radio interno de la zona galactica habitable, años luz
   GHZ_inner = 20000.
   # radio interno de la zona galactica habitable, años luz
   GHZ_outer = 60000.  

   # maximo tiempo para simular
   t_max = 1.e6

   # experiment ID
   exp_ID = SKRU_07
    
   tau_awakeningS = np.linspace(0, 200000, 51)[1:]
   tau_surviveS = np.linspace(0, 500000, 51)[1:]

   # Separate data in directories according to D_max
   #D_maxS = np.linspace(0, 40000, 11)[1::2]
   D_maxS = [500, 1000., 10000., 20000., 40000, 80000]

   Nran = 50                                            


   [output]

   output_dir = ../out/






Run experiments at IATE
=======================

In order to use the `HPC services at IATE <https://wiki.oac.uncor.edu/doku.php>`_ the following steps shoul be followed:


1. log in into a cluster (e.g., ``ssh clemente``)
2. git clone or pull the `CBR_correlation <https://github.com/mlares/CBR_CrossCorr>`_ project.
3. prepare a SLURM script (src/submit_python_jobs.sh)
4. launch the script: ``sbatch submit_python_jobs.sh``


SLURM script example for *clemente* running python in parallel:

.. code-block::
   #!/bin/bash

   # SLURM script for: CLEMENTE
    
   ## Las líneas #SBATCH configuran los recursos de la tarea
   ## (aunque parezcan estar comentadas)

   # More info:
   # http://homeowmorphism.com/articles/17/Python-Slurm-Cluster-Five-Minutes


   ## Nombre de la tarea
   #SBATCH --job-name=CMB_corr

   ## Cola de trabajos a la cual enviar.
   #SBATCH --partition=small

   ## tasks requested
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=20

   ## STDOUT
   #SBATCH -o submit_python_jobs.out

   ## STDOUT
   #SBATCH -e submit_python_jobs.err

   ## Tiempo de ejecucion. Formato dias-horas:minutos.
   #SBATCH --time 0-1:00

   ## Script que se ejecuta al arrancar el trabajo

   ## Cargar el entorno del usuario incluyendo la funcionalidad de modules
   ## No tocar
   . /etc/profile

   # conda init bash
   # source /home/${USER}/.bashrc

   module load gcc/8.2.0
   conda activate
   # por las dudas activar conda antes de correr el sbatch

   ## Launch program

   srun python /home/mlares/CBR_CrossCorr/src/run_correlation.py ../set/config_big.ini

   ## launch script
   ## $>sbatch submit_python_jobs.sh







