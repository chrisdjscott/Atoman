# Atoman - analysis and visualisation of atomistic simulations

## License and copying

Developed by Chris Scott  
Copyright 2015 Loughborough University  
Released under the MIT license, see "LICENSE.md" for details.  

## Dependencies

This code requires [Python](http://www.python.org) 2.7 or 3.5+. Atoman depends on a number of
Python libraries, a C and C++ compiler that supports OpenMP (tested with gcc and g++) and also
some other optional software.

### Python modules

This list of required modules can be found in "requirements.txt" and installed, for example,
using pip:

```
pip install -r requirements.txt --trusted-host download.qt.io
```

### Other software

In order to be able to use all features the following programmes should be installed:

*   [POV-Ray](http://www.povray.org/) (for offline rendering of images; looks better)
*   [Ffmpeg](https://www.ffmpeg.org/) (for creating movies; sequencer and rotator)

If these are not installed in default system locations you can set the paths to where they are
installed in the preferences dialog in the GUI.

## Installation

You can either build the code in-place or install the Python package, for example within a
[virtualenv](http://virtualenv.readthedocs.org/en/latest/).

In both cases you should first copy the config file: `cp setup.cfg.example setup.cfg` and add/edit
any relevant sections. There are some comments in the file to help with selecting the options.

You need to choose one of the following two methods for installing the software.

### In-place build

Build the C extensions in-place:

```
python setup.py build_ext -i
```

Verify the tests run with no errors:

```
python setup.py test
```

Then you can add the `/path/to/Atoman` directory to you PYTHONPATH and run `python -m atoman`
from anywhere to run the software.

### Install the Python package

Build the C extensions in-place:

```
python setup.py build_ext -i
```

Verify the tests run with no errors:

```
python setup.py test
```

Install the package:

```
python setup.py build
python setup.py install
```

If you are not using a virtual environment you may need to `sudo` the last (install) command.

Following this you should be able to run `python -m atoman` from anywhere to run the software.
Alternatively you could use the `Atoman` command to run the software.
