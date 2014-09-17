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
    CCLINKFLAGS = -bundle
	CCFLAGS = -c -O3 -fPIC -Wall
	CC = gcc
    INCLUDES = -I/opt/local/include
    LIBS = -L/opt/local/lib -lgsl -lgslcblas
endif
    
# linux - ubuntu or hydra
ifeq (Linux, $(shell uname -s))
    # compiler flags
    CCLINKFLAGS = -shared
	CCFLAGS = -c -O3 -fPIC -Wall
	CC = gcc
	LIBS = -lgsl -lgslcblas
	
	# different includes depending on OS
	# default, eg Ubuntu
	INCLUDES = -I/usr/include/python2.7/
	
	# modify for hydra
    ifeq (hydra3, $(shell uname -n))
        CCFLAGS += -std=c99
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

LIBFILES = _filtering.$(LIBSUFFIX) _output.$(LIBSUFFIX) \
		   _vectors.$(LIBSUFFIX) _rendering.$(LIBSUFFIX) \

# target
all : $(LIBFILES)

# create object file
_filtering.$(LIBSUFFIX) : filtering.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) filtering.o utilities.o boxeslib.o -o _filtering.$(LIBSUFFIX)

_output.$(LIBSUFFIX) : output.o
	$(CC) $(CCLINKFLAGS) output.o -o _output.$(LIBSUFFIX)

_vectors.$(LIBSUFFIX) : vectors.o utilities.o
	$(CC) $(CCLINKFLAGS) vectors.o utilities.o -o _vectors.$(LIBSUFFIX)

_rendering.$(LIBSUFFIX) : rendering.o
	$(CC) $(CCLINKFLAGS) rendering.o -o _rendering.$(LIBSUFFIX)

# compile module
boxeslib.o : boxeslib.c
	$(CC) $(CCFLAGS) $(INCLUDES) boxeslib.c 

utilities.o : utilities.c
	$(CC) $(CCFLAGS) $(INCLUDES) utilities.c

filtering.o : filtering.c 
	$(CC) $(CCFLAGS) $(INCLUDES) filtering.c 

output.o : output.c 
	$(CC) $(CCFLAGS) $(INCLUDES) output.c

vectors.o : vectors.c 
	$(CC) $(CCFLAGS) $(INCLUDES) vectors.c

rendering.o : rendering.c 
	$(CC) $(CCFLAGS) $(INCLUDES) rendering.c

neb_list.o : neb_list.c 
	$(CC) $(CCFLAGS) $(INCLUDES) neb_list.c

# clean
clean:
	rm  -f *.o *.so *_wrap.c *.pyc *.dylib
