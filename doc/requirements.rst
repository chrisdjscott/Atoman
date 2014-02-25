Requirements
============

.. _precompiled_version_requirements:

Precompiled version
-------------------

In order to be able to use all features when using the precompiled version the following programmes should be installed:

*   POV-Ray (for offline rendering of images; looks better)
*   Ffmpeg (for creating movies; sequencer and rotator)

If these are not installed in default system locations you can set the paths to where they are installed in the preference dialog.

Building from source
--------------------

The following python modules must be installed to build the code from source:

    * NumPy
    * SciPy
    * Matplotlib
    * PySide
    * PIL
    * Pyhull
    * pyvoro

In addition, the following programmes must also be installed:

    * Qt4
    * VTK

The programmes mentioned for the :ref:`precompiled_version_requirements` should also be installed to get full functionality.

Packaging
---------

Packaging into an application is achieved using PyInstaller.
