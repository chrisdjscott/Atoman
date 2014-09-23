# CDJSVis README file

## Requirements

This code requires [Python 2.7](http://www.python.org/download/releases/2.7/) to run.  It depends on a number of Python modules/libraries and also some other programmes.

### Python modules

This is a list of third-party python modules that this code requires to run.  They can be installed in a variety of ways, such as with [MacPorts](http://www.macports.org/), [pip](https://pypi.python.org/pypi/pip), [apt-get](http://en.wikipedia.org/wiki/Advanced_Packaging_Tool) or similar.

* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [PySide](https://pypi.python.org/pypi/PySide) (Python bindings to the Qt GUI framework, including pyside-rcc)
* Python Imaging Library ([PIL](http://www.pythonware.com/products/pil/))
* [pyhull](http://pythonhosted.org/pyhull/)
* [pyvoro](https://pypi.python.org/pypi/pyvoro)
* [VTK](http://www.vtk.org/) Python bindings
* [Sphinx](http://sphinx-doc.org/)
* [nose](https://nose.readthedocs.org/en/latest/) (testing)
* [Paramiko](http://www.paramiko.org/)

### Other libraries

This code also requires the following libraries to be installed:

* [GSL](http://www.gnu.org/software/gsl/) - GNU Scientific Library

### Other software

In order to be able to use all features the following programmes should be installed:

*   [POV-Ray](http://www.povray.org/) (for offline rendering of images; looks better)
*   [Ffmpeg](https://www.ffmpeg.org/) (for creating movies; sequencer and rotator)

If these are not installed in default system locations you can set the paths to where they are installed in the preferences dialog.

## Installation

The recommended way to install the code is to change into the root directory and run

```sh
python setup.py build_sphinx
```

which will build all the C extensions in place and also build the documentation.

Then you can add the `/path/to/CDJSVis` directory to you PYTHONPATH and PATH and run CDJSVis.py.

## Running the tests

Currently there are not many tests, but you should run the ones that are there. To run the tests you should run

```sh
python setup.py test
```

## Building application (Mac OS X)

On Mac OS X you can build a .app application using [PyInstaller](http://www.pyinstaller.org/). Simply change to the pyinstaller/ directory and run `./build.sh` (you may need to edit the path to pyinstaller.py).

