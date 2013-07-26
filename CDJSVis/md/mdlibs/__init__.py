
"""
Inititialise the mdlibs package.

We sym link to the correct libs depending on the architecture.

Author: Chris Scott

"""
import os


# change to mdlibs dir
cwd = os.getcwd()
os.chdir(os.path.dirname(os.path.realpath(__file__)))

libName = "LBOMDInterface.so"
libDir = "lib"

if os.path.islink(libName):
    head, currentLib = os.path.split(os.readlink(libName))
#    print "CURRENT LINKED", currentLib
else:
    currentLib = None


# if we are on a mac
lib = None
if os.uname()[0] == "Darwin":
    lib = "LBOMDInterface.maci7.so"

# if we are on a linux box
elif os.uname()[0] == "Linux":
    
    # if on hydra
    if os.uname()[1][:5] == "hydra":
        lib = "LBOMDInterface.hydra.so"
    
    # hera
    elif os.uname()[1][:4] == "hera":
        lib = "LBOMDInterface.hera.so"
    
    # otherwise use my linux (Ubuntu) compiled library
    else:
        lib = "LBOMDInterface.linux.so"

# now do the linking (if required)
if lib is None or lib == currentLib:
    pass
#    print "PASSING"
else:
    if currentLib is not None:
        os.unlink(libName)
    os.symlink(os.path.join(libDir, lib), libName)
#    print "LINKING", lib

os.chdir(cwd)
