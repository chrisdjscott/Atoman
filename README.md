# CDJSVis README file

This file is aimed at people wanting to run the code from source. If you are running a prebuilt '.app' application then the only section that concerns you is the [Other software](http://magrid-server-5.lut.ac.uk/macdjs/CDJSVis/tree/master#other-software) section below.

## Requirements

This code requires [Python 2.7](http://www.python.org/download/releases/2.7/) to run.  It depends on a number of Python modules/libraries and also some other programmes.

### Python modules

This is a list of third-party python modules that this code requires to run.  They can be installed in a variety of ways, such as with [MacPorts](http://www.macports.org/), [pip](https://pypi.python.org/pypi/pip), [apt-get](http://en.wikipedia.org/wiki/Advanced_Packaging_Tool) or similar.

Required:

* [NumPy](http://www.numpy.org/)
* [SciPy](http://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [PySide](https://pypi.python.org/pypi/PySide)
* [Pillow](https://pillow.readthedocs.org/)
* [VTK](http://www.vtk.org/) (with the Python bindings enabled)
* [Sphinx](http://sphinx-doc.org/)
* [nose](https://nose.readthedocs.org/en/latest/) (testing)

Optional:

* [Paramiko](http://www.paramiko.org/) - for loading files via the SFTP file browser
* [pyhull](http://pythonhosted.org/pyhull/) - for computing facet areas of convex hulls

### Other libraries

This code also requires the following libraries to be installed:

* [GSL](http://www.gnu.org/software/gsl/) - GNU Scientific Library

### Other software

In order to be able to use all features the following programmes should be installed:

*   [POV-Ray](http://www.povray.org/) (for offline rendering of images; looks better)
*   [Ffmpeg](https://www.ffmpeg.org/) (for creating movies; sequencer and rotator)

If these are not installed in default system locations you can set the paths to where they are installed in the preferences dialog.

## Installation

You can either build the code in-place or install it into the Python site-packages directory, for example within a [virtualenv](http://virtualenv.readthedocs.org/en/latest/).

In both cases you should first copy the config file: `cp setup.cfg.example setup.cfg` and add/edit any relevant sections. There are some comments in the file although
normally it can be left as is.

### In-place build

Build the documentation (includes inplace build of extensions):

```
python setup.py build_sphinx
```

and verify the tests run with no errors:

```
python setup.py test
```

Then you can add the `/path/to/CDJSVis` directory to you PYTHONPATH and PATH and run cdjsvis.py from anywhere.

### Installing to Python site-packages

Build the documentation (includes inplace build of extensions):

```
python setup.py build_sphinx
```

and verify the tests run with no errors:

```
python setup.py test
```

Then run:

```
python setup.py build
python setup.py install
```

If you are not using a virtual environment you may need to `sudo` the last (install) command.

## Building application (Mac OS X)

On Mac OS X you can build a .app application using [PyInstaller](http://www.pyinstaller.org/). Simply change to the pyinstaller/ directory and run `./build.sh` (you may need to edit the path to pyinstaller.py).

