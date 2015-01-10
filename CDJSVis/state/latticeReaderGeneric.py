
"""
Read user-specified format Lattice FILES_PER_THREAD

"""
import os
import sys
import copy
import re
import logging

import numpy as np

from . import latticeReaders
from . import _latticeReaderGeneric
from .atoms import elements
from ..visutils import utilities
from .lattice import Lattice



class FileFormat(object):
    """
    File format
    
    """
    def __init__(self, name):
        self.name = name
        self.header = []
        self.body = []
        self.delimiter = ' '
        self.atomIndexOffset = 1
    
    def newHeaderLine(self):
        self.header.append([])
    
    def addHeaderValue(self, key, typecode):
        if not len(self.header):
            raise RuntimeError("You must add a header line before adding a header value")
        
        if typecode != 'i' and typecode != 's' and typecode !='d':
            raise ValueError("Invalid value for header typecode ('i', 'd', 's'): '%s'", typecode)
        
        self.header[-1].append((key, typecode, 1))
    
    def newBodyLine(self):
        self.body.append([])
    
    def addBodyValue(self, key, typecode, dim):
        if not len(self.body):
            raise RuntimeError("You must add a body line before adding a body value")
        
        if typecode != 'i' and typecode !='d':
            raise ValueError("Invalid value for body typecode ('i', 'd'): '%s'", typecode)
        
        if dim != 1 and dim != 3:
            raise ValueError("Invalid value for dim (1 or 3): %d", dim)
        
        self.body[-1].append((key, typecode, dim))
    
    def setDelimiter(self, delim):
        self.delimiter = delim
    
    def setAtomIndexOffset(self, offset):
        self.atomIndexOffset = offset


class LatticeReaderGeneric(latticeReaders.GenericLatticeReader):
    """
    Generic format Lattice reader
    
    """
    def __init__(self, tmpLocation, availableFormats):
        super(LatticeReaderGeneric, self).__init__(tmpLocation, None, None, None)
        
        self.availableFormats = availableFormats
    
    def readFile(self, filename, fileFormat=None, rouletteIndex=None):
        """
        Read file.
        
        """
        self.logger.info("Reading file: '%s'", filename)
        
        # strip gz/bz2 extension
        if filename.endswith(".bz2"):
            filename = filename[:-4]
        elif filename.endswith(".gz"):
            filename = filename[:-3]
        
        filepath, zipFlag = self.checkForZipped(filename)
        if zipFlag == -1:
            self.displayWarning("Could not find file: "+filename)
            self.logger.warning("Could not find file: %s", filename)
            return -1, None
        
        try:
            status, state = self.readFileMain(filepath, fileFormat, rouletteIndex)
        
        except:
            print sys.exc_info()
            status = 255
        
        finally:
            self.cleanUnzipped(filepath, zipFlag)
        
        if status:
            self.logger.error("Generic Lattice reader failed with error code: %d", status)
        
        elif state is not None:
            self.currentFile = os.path.abspath(filename)
        
        return status, state
    
    def readFileMain(self, filename, fileFormat, rouletteIndex):
        """
        Main read
        
        """
        
        
        resultDict = _latticeReaderGeneric.readGenericLatticeFile(filename, fileFormat.header, fileFormat.body,
                                                                  fileFormat.delimiter, fileFormat.atomIndexOffset)
        
        # create Lattice object
        lattice = Lattice()
        lattice.NAtoms = resultDict.pop("NAtoms")
        lattice.pos = np.ravel(resultDict.pop("Position"))
        
        specieList = resultDict.pop("specieList")
        specieCount = resultDict.pop("specieCount")
        
        





