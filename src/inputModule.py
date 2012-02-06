
"""
File input methods.

author: Chris Scott
last edited: February 2012
"""

import os
import sys
import shutil

try:
    import utilities
except:
    sys.exit(__name__, "ERROR: utilities not found")





################################################################################
def readFile(filename, tmpLocation, lattice, fileType, state):
    
    # read file
    loc = checkForZipped(filename, tmpLocation)
    filename = os.path.join(loc, filename)
    
    print "READING", filename, fileType, state
    
    # first read the header
    if fileType == "LBOMD":
        
        if state == "ref":
            readLBOMDRef(filename, tmpLocation, lattice, fileType, state) 
            
            
            
        
        
    
    cleanUnzipped(loc)
    
    
################################################################################
def readLBOMDRef(filename, tmpLocation, lattice, fileType, state):
    
    f = open(filename)
            
    line = f.readline().strip()
    NAtoms = int(line)
    
    line = f.readline().strip()
    array = line.split()
    lattice.setDims(array)
    
    lattice.reset(NAtoms)
    






################################################################################
def checkForZipped(filename, tmpLocation):
    
    if os.path.exists(filename):
        fileLocation = '.'
    else:
        if os.path.exists(filename + '.bz2'):
            print 'INFO: bzip exists'
            command = "bzcat -k %s.bz2 > " % (filename)
            zippedFile = file+'.bz2'
        elif os.path.exists(filename + '.gz'):
            print 'INFO: gzip exists'
            command = "zcat %s.gz > " % (filename)
            zippedFile = filename+'.gz'
        else:
            print "WARNING: cannot find file:", filename
            return -1
            
        fileLocation = tmpLocation
        command = command + os.path.join(fileLocation, filename)
        os.system(command)
                
    return fileLocation



################################################################################
def cleanUnzipped(fileLocation):
    
    if fileLocation == '.':
        pass
    else:
        shutil.rmtree(fileLocation)


