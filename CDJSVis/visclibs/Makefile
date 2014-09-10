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

LIBFILES = _bonds.$(LIBSUFFIX) _filtering.$(LIBSUFFIX) _rdf.$(LIBSUFFIX) _input.$(LIBSUFFIX) _output.$(LIBSUFFIX) \
		   _picker.$(LIBSUFFIX) _clusters.$(LIBSUFFIX) _defects.$(LIBSUFFIX) _vectors.$(LIBSUFFIX) _rendering.$(LIBSUFFIX) \
		   _bond_order.$(LIBSUFFIX) _acna.$(LIBSUFFIX)

# target
all : $(LIBFILES)

# create object file
_bonds.$(LIBSUFFIX) : bonds.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) bonds.o boxeslib.o utilities.o -o _bonds.$(LIBSUFFIX)

_rdf.$(LIBSUFFIX) : rdf.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) rdf.o boxeslib.o utilities.o -o _rdf.$(LIBSUFFIX)

_filtering.$(LIBSUFFIX) : filtering.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) filtering.o utilities.o boxeslib.o -o _filtering.$(LIBSUFFIX)

_input.$(LIBSUFFIX) : input.o
	$(CC) $(CCLINKFLAGS) input.o -o _input.$(LIBSUFFIX)

_output.$(LIBSUFFIX) : output.o
	$(CC) $(CCLINKFLAGS) output.o -o _output.$(LIBSUFFIX)

_picker.$(LIBSUFFIX) : picker.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) picker.o utilities.o boxeslib.o -o _picker.$(LIBSUFFIX)

_clusters.$(LIBSUFFIX) : clusters.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) clusters.o utilities.o boxeslib.o -o _clusters.$(LIBSUFFIX)

_defects.$(LIBSUFFIX) : defects.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) defects.o utilities.o boxeslib.o -o _defects.$(LIBSUFFIX)

_vectors.$(LIBSUFFIX) : vectors.o utilities.o
	$(CC) $(CCLINKFLAGS) vectors.o utilities.o -o _vectors.$(LIBSUFFIX)

_rendering.$(LIBSUFFIX) : rendering.o
	$(CC) $(CCLINKFLAGS) rendering.o -o _rendering.$(LIBSUFFIX)

_bond_order.$(LIBSUFFIX) : bond_order.o neb_list.o utilities.o boxeslib.o
	$(CC) $(CCLINKFLAGS) bond_order.o neb_list.o utilities.o boxeslib.o $(LIBS) -o _bond_order.$(LIBSUFFIX)

_acna.$(LIBSUFFIX) : acna.o neb_list.o utilities.o boxeslib.o
	$(CC) $(CCLINKFLAGS) acna.o neb_list.o utilities.o boxeslib.o $(LIBS) -o _acna.$(LIBSUFFIX)

# compile module
boxeslib.o : boxeslib.c
	$(CC) $(CCFLAGS) $(INCLUDES) boxeslib.c 

utilities.o : utilities.c
	$(CC) $(CCFLAGS) $(INCLUDES) utilities.c

bonds.o : bonds.c 
	$(CC) $(CCFLAGS) $(INCLUDES) bonds.c

rdf.o : rdf.c 
	$(CC) $(CCFLAGS) $(INCLUDES) rdf.c

filtering.o : filtering.c 
	$(CC) $(CCFLAGS) $(INCLUDES) filtering.c 

input.o : input.c 
	$(CC) $(CCFLAGS) $(INCLUDES) input.c

output.o : output.c 
	$(CC) $(CCFLAGS) $(INCLUDES) output.c

picker.o : picker.c 
	$(CC) $(CCFLAGS) $(INCLUDES) picker.c

clusters.o : clusters.c 
	$(CC) $(CCFLAGS) $(INCLUDES) clusters.c

defects.o : defects.c 
	$(CC) $(CCFLAGS) $(INCLUDES) defects.c

vectors.o : vectors.c 
	$(CC) $(CCFLAGS) $(INCLUDES) vectors.c

rendering.o : rendering.c 
	$(CC) $(CCFLAGS) $(INCLUDES) rendering.c

neb_list.o : neb_list.c 
	$(CC) $(CCFLAGS) $(INCLUDES) neb_list.c

bond_order.o : bond_order.c 
	$(CC) $(CCFLAGS) $(INCLUDES) bond_order.c

acna.o : acna.c 
	$(CC) $(CCFLAGS) $(INCLUDES) acna.c

# clean
clean:
	rm  -f *.o *.so *_wrap.c *.pyc *.dylib
