***********
To do list
***********


Documentation
=============

- Complete documentation: it is generated automatically from
  metacomments, using Sphinx.


Testing
=======

- Asserts:

  * What to do is data files are missing
  * Prevent variable overflows and division by zero

- pytest

  * directories?
  * classes?

- TOX: `tox <https://tox.readthedocs.io/en/latest/>`_ is a generic virtualenv management and test command line tool you can use for:

   * checking your package installs correctly with different Python versions and interpreters
   * running your tests in each of the environments, configuring your test tool of choice
   * acting as a frontend to Continuous Integration servers, greatly reducing boilerplate and merging CI and shell-based testing.


`About good integration practices <https://docs.pytest.org/en/latest/goodpractices.html>`_



Building and Deployment
=======================

- Continuous integration: `Travis? <https://travis-ci.com/>`_

  Continuous Integration is the practice of merging in small code
  changes frequently - rather than merging in a large change at the
  end of a development cycle. The goal is to build healthier software
  by developing and testing in smaller increments.


- setup.py: it does the similar job of pip, easy_install etc.

`Ver como configurar el setup.py<https://packaging.python.org/tutorials/packaging-projects/>`_

