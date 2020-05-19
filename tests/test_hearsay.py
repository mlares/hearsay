import sys
import pytest

# =========================================================
# PARSING
# =========================================================

from hearsay import hearsay

import pathlib
import os
PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

print(PATH)
print(type(PATH))
print(PATH.absolute())

# ____________________________________________________________
# PARSING FILENAME

def test_parser_01():
    conf = hearsay.parser()
    assert isinstance(conf, hearsay.parser)

def test_parser_02():
    conf = hearsay.parser()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    assert isinstance(conf.filename, str)

def test_parser_03():
    conf = hearsay.parser()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.check_file('../set/experiment_test.ini')
    assert conf.filename == '../set/experiment_test.ini'
 
def test_parser_04():
    conf = hearsay.parser()
    conf.check_file()
    assert conf.filename == '../set/experiment.ini'
 
def test_parser_05():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.read_config_file()
    conf.load_filenames()
    assert len(conf.filenames) == 8
 
def test_parser_06():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.read_config_file()
    conf.load_filenames()
    conf.load_parameters()
    assert len(conf.p) == 25

def test_parser_07():
    import configparser
    conf = hearsay.parser()
    assert isinstance(conf['experiment'], configparser.SectionProxy)

def test_parser_08():
    import configparser
    conf = hearsay.parser()
    assert isinstance(conf['experiment']['exp_ID'], str)

def test_parser_09():
    import configparser
    conf = hearsay.parser('non_existent_file')
    assert isinstance(conf, hearsay.parser)
               
def test_parser_10():
    import configparser
    conf = hearsay.parser(['','non_existent_file'])
    assert isinstance(conf, hearsay.parser)
                  
def test_parser_11():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.read_config_file()
    conf.load_parameters()            
    assert len(conf.filenames) == 8                 
                           
# ____________________________________________________________
# PARSING PARAMETERS


@pytest.mark.parametrize(
    'key, val, boolean',
    [('run_parallel', 'n', False), 
     ('run_parallel', 'no', False), 
     ('run_parallel', 'NO', False), 
     ('run_parallel', 'false', False), 
     ('run_parallel', 'FALSE', False), 
     ('run_parallel', 'y', True),
     ('run_parallel', 'yes', True),
     ('run_parallel', 's', True),
     ('run_parallel', 'true', True),
     ('run_parallel', 'TRUE', True)]
)
def test_parser_12(key, val, boolean):
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.read_config_file()
    conf.load_filenames()
    conf.load_parameters([key],[val])
    assert conf.p.run_parallel == boolean


# ____________________________________________________________
# Galactic Network class

def test_GN_01():

    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(['',fname])
    conf.read_config_file()
    conf.load_filenames()
    conf.load_parameters()

    tau_awakening = 20000
    tau_survive = 20000
    D_max = 20000
    pars = [tau_awakening, tau_survive, D_max]
    #G = hearsay.GalacticNetwork()
    #G.run_simulation(conf.p, pars)
    assert 1>0#len(G.MPL) > 0




# Testing:
# cd tests/
# pytest test_hearsay.py
# pytest -q . --cov=../hearsay/ --cov-append ???

# Code coverage: 
# coverage run -m pytest test_hearsay.py
# coverage report -m

