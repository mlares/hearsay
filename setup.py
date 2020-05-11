# ===========================================================
# DOCS
# ===========================================================

"""Utilities to make discrete event simulations

"""


# ===========================================================
# IMPORTS
# ===========================================================

import pathlib
import os

# from ez_setup import use_setuptools
# use_setuptools()

from setuptools import setup, find_packages

# ===========================================================
# CONSTANTS
# ===========================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

REQUIREMENTS = [
    "numpy", "attrs",
    "matplotlib", "seaborn"]

with open(PATH / "README.rst") as fp:
    LONG_DESCRIPTION = fp.read()

DESCRIPTION = (
    "Utilities to access different Argentina-Related databases of "
    "COVID-19 data from the IATE task force.")

with open(PATH / "hearsay" / "__init__.py") as fp:
    VERSION = [
        l for l in fp.readlines() if l.startswith("__version__")
    ][0].split("=", 1)[-1].strip().replace('"', "")

# ===========================================================
# FUNCTIONS
# ===========================================================

setup(name="hearsay", packages=find_packages())
