# CDJSVis README file

This file is aimed at people wanting to run the code from source. If you are running a prebuilt .app application then the only section that concerns you is the [Other software](http://magrid-server-5.lut.ac.uk/macdjs/CDJSVis/tree/master#other-software) section below.

## Requirements

This code requires [Python 2.7](http://www.python.org/download/releases/2.7/) to run.  It depends on a number of Python modules/libraries and also some other programmes.

### Python modules

This is a list of third-party python modules that this code requires to run.  They can be installed in a variety of ways, such as with [MacPorts](http://www.macports.org/), [pip](https://pypi.python.org/pypi/pip), [apt-get](http://en.wikipedia.org/wiki/Advanced_Packaging_Tool) or similar.

Required:

* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [PySide](https://pypi.python.org/pypi/PySide) (including 'pyside-rcc' (commonly included in 'pyside-tools' or similar))
* [Pillow](https://pillow.readthedocs.org/)
* [VTK](http://www.vtk.org/) (with the Python bindings enabled)
* [Sphinx](http://sphinx-doc.org/)
* [nose](https://nose.readthedocs.org/en/latest/) (testing)
* [Paramiko](http://www.paramiko.org/)

Optional:

* [pyhull](http://pythonhosted.org/pyhull/) (Optional, for computing facet areas of convex hulls)

### Other libraries

This code also requires the following libraries to be installed:

* [GSL](http://www.gnu.org/software/gsl/) - GNU Scientific Library

### Other software

In order to be able to use all features the following programmes should be installed:

*   [POV-Ray](http://www.povray.org/) (for offline rendering of images; looks better)
*   [Ffmpeg](https://www.ffmpeg.org/) (for creating movies; sequencer and rotator)

If these are not installed in default system locations you can set the paths to where they are installed in the preferences dialog.

## Installation

You can either build the code in-place or install it into the Python site-packages directory (both using the setup.py script).

### Installing to Python site-packages

Build the documentation (includes inplace build of extensions):

```sh
python setup.py build_sphinx
```

and verify the tests run with no errors:

```sh
python setup.py test
```

Then run:

```sh
python setup.py build
python setup.py install
```

In you are not using a virtual environment you may need to `sudo` the last (install) command.

### In-place build

Build the documentation (includes inplace build of extensions):

```sh
python setup.py build_sphinx
```

and verify the tests run with no errors:

```sh
python setup.py test
```

Then you can add the `/path/to/CDJSVis` directory to you PYTHONPATH and PATH and run cdjsvis.py from anywhere.

## Building application (Mac OS X)

On Mac OS X you can build a .app application using [PyInstaller](http://www.pyinstaller.org/). Simply change to the pyinstaller/ directory and run `./build.sh` (you may need to edit the path to pyinstaller.py).

