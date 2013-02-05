SHELL = /bin/sh

################################################################################
# ARCHITECTURE DEPENDENT SECTION
# Generally the only problem with the settings below is the INCLUDE path
# which needs to include the python and numpy directories
################################################################################

# attempt to autodetect options

# mac, assumes latest macports python being used and setup correctly
ifeq (Darwin, $(shell uname -s))
    # compiler flags
    CCLINKFLAGS =   -bundle -flat_namespace -undefined suppress
	CCFLAGS =  -c -O3 -fPIC
	CC = gcc
    
    # which python version are we using (includes are different)
    INCLUDES = 
#    ifeq (2.7.3, $(shell python -V 2>&1 | awk '{ gsub(/Python /,""); print }'))
#        INCLUDES = -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include
#    endif
#    ifeq (2.6.6, $(shell python -V 2>&1 | awk '{ gsub(/Python /,""); print }'))
#        INCLUDES = -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include
#    endif
endif
    
# linux - ubuntu or hydra
ifeq (Linux, $(shell uname -s))
    # compiler flags
    CCLINKFLAGS =   -shared -flat_namespace
	CCFLAGS = -c -O3 -fPIC
	CC = gcc
	
	# different includes depending on OS
	# default, eg Ubuntu
	INCLUDES = -I/usr/include/python2.7/
	
	# modify for hydra
    ifeq (hydra3, $(shell uname -n))
        CCFLAGS += -std=c99
        INCLUDES = #-I/home/ma/macdjs/Applications/local/include/python2.7/ -I/home/ma/macdjs/Applications/local/lib/python2.7/site-packages/numpy/core/include
    endif
endif

################################################################################
# SHOULD NOT NEED TO CHANGE BELOW THIS LINE
################################################################################

SHOBJECTS = *.o=.so
OBJECTS = ${COBJECTS:.c=.o}
COBJECTS = *.c

ifeq (Darwin, $(shell uname -s))
	LIBSUFFIX = dylib
endif
ifeq (Linux, $(shell uname -s))
	LIBSUFFIX = so
endif

LIBFILES = _bonds.$(LIBSUFFIX)

all : $(LIBFILES)

# create object file
_bonds.$(LIBSUFFIX) : bonds.o
	$(CC) $(CCLINKFLAGS) bonds.o ../visclibs/boxeslib.o ../visclibs/utilities.o -o _bonds.$(LIBSUFFIX)

# compile module
bonds.o : bonds.c 
	$(CC) $(CCFLAGS) $(INCLUDES) bonds.c 

# clean
clean:
	rm  -f *.o *.so *_wrap.c *.pyc *.dylib
