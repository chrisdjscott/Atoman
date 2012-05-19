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
    ifeq (2.7.3, $(shell python -V 2>&1 | awk '{ gsub(/Python /,""); print }'))
        INCLUDES = -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include
    endif
    ifeq (2.6.6, $(shell python -V 2>&1 | awk '{ gsub(/Python /,""); print }'))
        INCLUDES = -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6 -I/opt/local/Library/Frameworks/Python.framework/Versions/2.6/lib/python2.6/site-packages/numpy/core/include
    endif
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
        INCLUDES = -I/home/ma/macdjs/Applications/local/include/python2.7/ -I/home/ma/macdjs/Applications/local/lib/python2.7/site-packages/numpy/core/include
    endif
endif

SWIG := swig -python

################################################################################
# SHOULD NOT NEED TO CHANGE BELOW THIS LINE
################################################################################

SHOBJECTS = *.o=.so
OBJECTS = ${COBJECTS:.c=.o}
COBJECTS = *.c
SWIGSOURCE = input_c.i filtering_c.i defects_c.i clusters_c.i output_c.i

all : _input_c.so _filtering_c.so _defects_c.so _clusters_c.so _output_c.so

# create object file
_input_c.so : input_c_wrap.o input_c.o
	$(CC) $(CCLINKFLAGS) input_c_wrap.o input_c.o -o _input_c.so

_output_c.so : output_c_wrap.o output_c.o
	$(CC) $(CCLINKFLAGS) output_c_wrap.o output_c.o -o _output_c.so

_filtering_c.so : filtering_c_wrap.o filtering_c.o utilities.o
	$(CC) $(CCLINKFLAGS) filtering_c_wrap.o filtering_c.o utilities.o -o _filtering_c.so

_defects_c.so : defects_c_wrap.o defects_c.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) defects_c.o defects_c_wrap.o boxeslib.o utilities.o -o _defects_c.so

_clusters_c.so : clusters_c_wrap.o clusters_c.o boxeslib.o utilities.o
	$(CC) $(CCLINKFLAGS) clusters_c.o clusters_c_wrap.o boxeslib.o utilities.o -o _clusters_c.so

# compile module
input_c.o : input_c.c 
	$(CC) $(CCFLAGS) $(INCLUDES) input_c.c 

output_c.o : output_c.c 
	$(CC) $(CCFLAGS) $(INCLUDES) output_c.c 

filtering_c.o : filtering_c.c 
	$(CC) $(CCFLAGS) $(INCLUDES) filtering_c.c 

defects_c.o : defects_c.c
	$(CC) $(CCFLAGS) $(INCLUDES) defects_c.c 

clusters_c.o : clusters_c.c
	$(CC) $(CCFLAGS) $(INCLUDES) clusters_c.c 

boxeslib.o : boxeslib.c
	$(CC) $(CCFLAGS) $(INCLUDES) boxeslib.c 

utilities.o : utilities.c
	$(CC) $(CCFLAGS) $(INCLUDES) utilities.c 

# generate and compile swig wrapper
input_c_wrap.o : input_c.i 
	$(SWIG) input_c.i
	$(CC) $(CCFLAGS) $(INCLUDES) input_c_wrap.c 

output_c_wrap.o : output_c.i 
	$(SWIG) output_c.i
	$(CC) $(CCFLAGS) $(INCLUDES) output_c_wrap.c 

filtering_c_wrap.o : filtering_c.i 
	$(SWIG) filtering_c.i
	$(CC) $(CCFLAGS) $(INCLUDES) filtering_c_wrap.c 

defects_c_wrap.o : defects_c.i
	$(SWIG) defects_c.i
	$(CC) $(CCFLAGS) $(INCLUDES) defects_c_wrap.c 

clusters_c_wrap.o : clusters_c.i
	$(SWIG) clusters_c.i
	$(CC) $(CCFLAGS) $(INCLUDES) clusters_c_wrap.c

# clean
clean:
	rm  -f *.o *.so *_wrap.c *.pyc $(SWIGSOURCE:.i=.py)
	

