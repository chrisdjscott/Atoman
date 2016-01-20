Installation
============

Requirements
------------

Python 2.7 or 3.2 or later is required (if using Python 3.x then you must use
VTK 7.0 or later).

C and C++ compilers are also required (gcc and g++ have been tested). Note that
OpenMP is required by default and if you want to use a compiler that does not
support OpenMP, e.g. the default gcc (clang) on Mac OS X, then you will need to
comment the lines in `setup.py` where it adds the OpenMP specific commands for
the compilers.

The following python modules are required:

    * setuptools
    * NumPy
    * SciPy
    * Matplotlib
    * PySide (Qt4)
    * Pillow
    * Sphinx
    * nose
    * VTK (tested with 5.10, 6.3 and 7.0)
    * Paramiko (optional)
    * PyHull (optional)
    * PyInstaller (optional)

For full functionality the following programs should also be installed:

    * POV-Ray (optional)
    * FFmpeg (optional)

If these programs are not installed to a default location then you may beed to
specify their location in the preferences dialog.

Installing
----------

Use one of the following two methods for installing this software.

Method 1 - in place build
~~~~~~~~~~~~~~~~~~~~~~~~~

Build the documentation (includes an in place build of extensions):

.. code:: sh
   
    python setup.py build_sphinx


and verify the tests run with no errors:

.. code:: sh

    python setup.py test

Then you can add the `/path/to/Atoman` directory to your PYTHONPATH and run
`python -m atoman` from anywhere to run the software.

Method 2 - site packages
~~~~~~~~~~~~~~~~~~~~~~~~

Build the documentation (includes inplace build of extensions):

.. code:: sh

    python setup.py build_sphinx

and verify the tests run with no errors:

.. code:: sh

    python setup.py test

Then run:

.. code:: sh

    python setup.py build
    python setup.py install

If you are not using a virtual environment you may need to `sudo` the last
(install) command.

Following this you should be able to run `python -m atoman` from anywhere to
run the software. Alternatively you could use the `atoman` command to run the
software.

Packaging
---------

On Mac OS X you can build a .app bundle using PyInstaller. Simply change to the
`pyinstaller/` directory and run `./build.sh` (you may need to edit the path to
pyinstaller.py).
