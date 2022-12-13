..  Copyright 2022 Johannes Reiff
    SPDX-License-Identifier: Apache-2.0

***************
Getting Started
***************

.. highlight:: console

.. note::
    The following commands are assumed to be run
    from the project's root directory,
    i.e., the directory containing :git-file:`pyproject.toml`.



Installing from source
======================

RODEO can be installed directly from source using standard Python utilities.
Simply run ::

    $ pip install .

for a default installation, or ::

    $ pip install -e .[dev]

for an editable installation that includes
development dependencies for documentation and testing.



Building wheels
===============

Run ::

    $ pip wheel --no-deps --wheel-dir _dist .

to package RODEO into an installable wheel (located in ``_dist/``).
Wheels can easily be distributed and installed via ::

    $ pip install path/to/package.whl

or similar.



Building the documentation
==========================

.. note::
    RODEO needs to be importable to build the documentation.
    This can be achieved most reliably by installing the package beforehand.

Run ::

    $ sphinx-build -M html doc _doc

to build the HTML documentation.
This will place the index file at ``_doc/html/index.html``.
To build a self-contained PDF version, run ::

    $ sphinx-build -M latexpdf doc _doc

instead.
This results in the file ``_doc/latex/rodeo.pdf``.
See ::

    $ sphinx-build -M help doc _doc

for a full list of available formats.



Running tests
=============

.. note::
    RODEO needs to be importable to run the tests.
    This can be achieved most reliably by installing the package beforehand.

Use ::

    $ pytest -W error

to run the tests.
Slow tests can be skipped by adding ``--skip-slow`` or ``-m 'not slow'``
to the command above.
