import sys
import pytest
import hearsay as ccn

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
    conf.check_file(['',fname])
    assert conf.filename == '../set/experiment_test.ini'
 
def test_parser_04():
    conf = hearsay.parser()
    conf.check_file()
    assert conf.filename == '../set/experiment.ini'
 

# ____________________________________________________________
# PARSING PARAMETERS

def test_parser_05():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(['',fname])
    conf.read_config_file()
    conf.load_filenames()
    assert len(conf.filenames) == 6

 
def test_parser_06():
    conf = hearsay.parser()
    conf.check_file()
    fname = PATH / '../set/experiment_test.ini'
    conf.check_file(['',fname])
    conf.read_config_file()
    conf.load_filenames()
    conf.load_parameters()
    assert len(conf.p) == 19
                              

# ____________________________________________________________
# Galactiv Network class

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
    G = hearsay.GalacticNetwork()
    G.run_simulation(conf.p, pars)
    assert len(G.MPL) > 0


# ____________________________________________________________
# Node and ordered list


# ____________________________________________________________


class Test_CCN:
    """
    This module is intended to store test functions
    related to CCNs and communication networks.
    """
 
    def test_node_single_contact(self):
        #{{{
        """
        test_node_single_contact(self):
            test the assignment of a songle contact.

        Tasks:

        Args:

        Raises:
            errors?

        Returns:
        """
