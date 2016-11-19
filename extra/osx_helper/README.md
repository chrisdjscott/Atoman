# OSX Helper

On Mac OS X there is a problem with Qt5/VTK that causes the VTK window to show
up in the bottom quarter only, when using a retina display:

http://vtk.1045678.n5.nabble.com/QVTK-Widget-and-Retina-MacBook-High-DPI-screen-td5739142.html
http://public.kitware.com/pipermail/vtkusers/2015-February/090117.html

## Installing

Run these commands to build (CMake is required):

```sh
mkdir build && cd build
cmake ..
make
```

Note it is important that CMake use the Apple Clang compiler, which should be automatically
detected.

The build directory must be added to DYLD_LIBRARY_PATH so that the library can
be located. On bash this could be done with (assuming you are in the build
directory when running the command):

```sh
export DYLD_LIBRARY_PATH=$(pwd):$DYLD_LIBRARY_PATH
```
