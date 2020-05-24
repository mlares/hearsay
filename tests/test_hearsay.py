import sys
import pytest
from hearsay import hearsay
import pandas as pd
import pathlib
import os
import configparser
PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
 

# =========================================================
# PARSING
# =========================================================

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
    assert len(conf.p) == 25

def test_parser_07():
    conf = hearsay.parser()
    assert isinstance(conf['experiment'], configparser.SectionProxy)

def test_parser_08():
    conf = hearsay.parser()
    assert isinstance(conf['experiment']['exp_ID'], str)

def test_parser_09():
    conf = hearsay.parser('non_existent_file')
    assert isinstance(conf, hearsay.parser)
               
def test_parser_10():
    conf = hearsay.parser(['','non_existent_file'])
    assert isinstance(conf, hearsay.parser)
                  
def test_parser_11():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    assert len(conf.filenames) == 8                 
 
def test_parser_12():
    conf = hearsay.parser(['','../set/experiment_test.ini'])
    conf.check_file(fname)
    assert len(conf.filenames) == 8                 
 
def test_parser_13():
    conf = hearsay.parser(['non_existent_file'])
    assert len(conf.filenames) == 8                 
 
def test_parser_14():
    conf = hearsay.parser()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.check_settings()
    assert len(conf.filenames) == 8                 
 
def test_parser_14():
    conf = hearsay.parser()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.load_config(['dir_output'], ['non_existent_dir'])
    conf.check_settings()
    assert len(conf.filenames) == 8                 
                           

# ____________________________________________________________
# PARSING PARAMETERS

def complete_boolean_options(par, key):
    yes = ['y', 'Y','yes','YES','Yes','True','true','TRUE']
    nop = ['n', 'N', 'no', 'NO', 'No', 'False', 'false', 'FALSE', 'maybe']
    li = []
    for i in yes:
        li.append((par, key, i, True))
    for i in nop:
        li.append((par, key, i, False))
    return li

l1 = complete_boolean_options('run_parallel', 'run_parallel')
l2 = complete_boolean_options('verbose', 'verbose')
l3 = complete_boolean_options('show_progress', 'showp')
lista = l1 + l2 + l3

@pytest.mark.parametrize('par, key, val, expected', lista)
def test_parser_12(par, key, val, expected):
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(fname)
    conf.read_config_file()
    conf.load_config([par], [val])
    assert getattr(conf.p, key) == expected
 


# ____________________________________________________________
# Galactic Network class

def test_GN_01():
    conf = hearsay.parser()
    G = hearsay.GalacticNetwork(conf)
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
    res = G.run(interactive=True)
    assert len(res)==1
     
def test_GN_02():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['3'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000, 40000]
    S = [10000]
    D = [10000]
    G.set_parameters(A=A, S=S, D=D)
    res = G.run(interactive=True)
    assert len(res)==9
 
def test_GN_03():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['2'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000, 40000]
    S = [10000, 12000, 14000]
    D = [10000, 12000, 14000]
    G.set_parameters(A=A, S=S, D=D)
    res = G.run(interactive=True)
    assert len(res)==54
 
def test_GN_04():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['1'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000]
    S = [10000, 12000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)
    assert len(res)==4
     
def test_GN_05():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['1'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000]
    S = [10000, 12000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)
    R = hearsay.results(G)
    R.load()
    res = R.redux_1d()
    assert len(res)==11
 
def test_GN_06():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['1'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000]
    S = [10000, 12000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)
    R = hearsay.results(G)
    R.load()
    res = R.redux_2d()
    assert len(res)==2

def test_GN_07():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['1'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000]
    S = [10000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)
    res = G.show_single_ccns(interactive=True)
    assert isinstance(res, dict)
 
def test_GN_08():
    conf = hearsay.parser()
    conf.load_config(['nran', 'njobs', 'run_parallel'], ['10', '2', 'Y'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000]
    S = [10000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)
    res = G.show_single_ccns(interactive=True)
    assert isinstance(res, dict)
 
     
# ____________________________________________________________
# Results class

def test_results_01():
    conf = hearsay.parser()
    conf.load_config(['nran'], ['1'])
    G = hearsay.GalacticNetwork(conf)
    G.set_parameters()
    A = [20000, 30000]
    S = [10000, 12000]
    D = [10000]
    lista = [A, S, D]
    G.set_parameters(spars=lista)
    res = G.run(interactive=True)

    R = hearsay.results(G)
    R.load()
    res = R.show_ccns(0, True)
    assert isinstance(res, dict)
     

# Testing:
# cd tests/
# pytest test_hearsay.py
# pytest -q . --cov=../hearsay/ --cov-append

# Code coverage: 
# coverage run -m pytest test_hearsay.py
# coverage report -m
# coverage report html

""" 75-76, 167, 237, 241-242, 320-328, 331-339, 415-431, 470, 492,
534, 542, 550, 590, 632, 652-653, 674, 677-682, 687, 692, 704-741,
769-770, 782-783, 793, 801-802, 808-812, 823, 842, 844-847, 876-879,
933-936, 1017-1018, 1047, 1135-1136, 1227-1229, 1232-1233, 1258-1259,
1262, 1289-1290, 1299-1303 """
