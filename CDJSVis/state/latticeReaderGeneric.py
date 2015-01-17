
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


################################################################################

class FileFormats(object):
    """
    Object containing file formats
    
    """
    def __init__(self):
        self._fileFormats = {}
        self._maxIdentifierLength = 0
        self.logger = logging.getLogger(__name__+".FileFormats")
    
    def getFormat(self, name):
        """
        Get the format with the given name
        
        """
        return self._fileFormats[name]
    
    def getMaxIdentifierLength(self):
        """
        Return length of longest identifier
        
        """
        return self._maxIdentifierLength
    
    def addFileFormat(self, fileFormat):
        """
        Add file format
        
        """
        if not isinstance(fileFormat, FileFormat):
            raise TypeError("Input file format must be of type FileFormat")
        
        if fileFormat.name in self._fileFormats:
            raise ValueError("File format name is not unique ('%s')" % fileFormat.name)
        
        self.logger.debug("Adding file format: '%s'", fileFormat.name)
        
        # add to dict
        self._fileFormats[fileFormat.name] = fileFormat
        
        # max identifier length
        idlen = len(fileFormat.getIdentifier())
        if idlen > self._maxIdentifierLength:
            self._maxIdentifierLength = idlen
            self.logger.debug("Max identifier length is: %d", idlen)
    
    def read(self, filename="file_formats.IN"):
        """
        Read from file
        
        """
        with open(filename) as f:
            numFormats = int(f.readline())
            
            for _ in xrange(numFormats):
                fmt = FileFormat()
                fmt.read(f)
                self.addFileFormat(fmt)
                
                print "="*80
                fmt.print_()
        print "="*80
        
        self.checkLinkedNames()
    
    def checkLinkedNames(self):
        """
        Check that linked names exist
        
        """
        print "*********Check Linked Names not implemented yet*********"
        
    
    def save(self, filename="file_formats.IN"):
        """
        Save to file
        
        """
        with open(filename, "w") as f:
            f.write("%d\n" % len(self._fileFormats))
            for key in self._fileFormats:
                fmt = self._fileFormats[key]
                fmt.write(f)
    
    def __len__(self):
        return len(self._fileFormats)
    
    def __iter__(self):
        for v in self._fileFormats.values():
            yield v

################################################################################

class FileFormat(object):
    """
    File format
    
    """
    def __init__(self, name=None):
        self.name = name
        self.header = []
        self.body = []
        self.delimiter = ' '
        self.atomIndexOffset = 1
        self.linkedName = None
    
    def setName(self, name):
        """
        Set name
        
        """
        self.name = name
    
    def newHeaderLine(self):
        """
        Add a new header line
        
        """
        self.header.append([])
    
    def addHeaderValue(self, key, typecode):
        """
        Add header value to last header line
        
        """
        if not len(self.header):
            raise RuntimeError("You must add a header line before adding a header value")
        
        if typecode != 'i' and typecode != 's' and typecode !='d':
            raise ValueError("Invalid value for header typecode ('i', 'd', 's'): '%s'", typecode)
        
        self.header[-1].append((key, typecode, 1))
    
    def newBodyLine(self):
        """
        Add a new body line
        
        """
        self.body.append([])
    
    def addBodyValue(self, key, typecode, dim):
        """
        Add body value to last body line
        
        """
        if not len(self.body):
            raise RuntimeError("You must add a body line before adding a body value")
        
        if typecode != 'i' and typecode !='d':
            raise ValueError("Invalid value for body typecode ('i', 'd'): '%s'", typecode)
        
        if dim != 1 and dim != 3:
            raise ValueError("Invalid value for dim (1 or 3): %d", dim)
        
        self.body[-1].append((key, typecode, dim))
    
    def setDelimiter(self, delim):
        """
        Set the delimiter
        
        """
        self.delimiter = delim
        if " " not in self.delimiter:
            self.delimiter += " "
    
    def setAtomIndexOffset(self, offset):
        """
        Set atomIndexOffset. atomID - atomIndexOffset = storage index
        
        """
        self.atomIndexOffset = offset
    
    def setLinkedName(self, name):
        """
        Set name of format this format is linked to
        
        """
        self.linkedName = name
    
    def verify(self):
        """
        Verify the format is acceptable
        
        """
        errors = []
        
        haveNAtoms = False
        for headerItem in self.header:
            key, typecode = headerItem
            
            # currently NAtoms must be in the header (this will change eventually)
            if key == "NAtoms":
                haveNAtoms = True
        
        if not haveNAtoms:
            errors.append("Format header must contain 'NAtoms'")
        
        havePos = False
        haveSymbol = False
        for bodyLine in self.body:
            for bodyItem in bodyLine:
                key, typecode, dim = bodyItem
                
                # must have pos
                if key == "Position":
                    havePos = True
                elif key == "Symbol":
                    haveSymbol = True
        
        if not havePos:
            errors.append("Format body must contain 'Position'")
        
        if (not haveSymbol) and self.linkedName is None:
            errors.append("Fornat body must contain 'Symbol' or the format must be linked to another format type")
        
        return errors
    
    def write(self, fh):
        """
        Write the format to file
        
        """
        # write name
        fh.write("%s\n" % self.name)
        
        # delimiter
        fh.write("%s\n" % self.delimiter)
        
        # atom index offset
        fh.write("%d\n" % self.atomIndexOffset)
        
        # linked file
        if self.linkedName is None:
            fh.write("\n")
        else:
            fh.write("%s\n" % self.linkedName)
        
        # number of header lines
        fh.write("%d\n" % len(self.header))
        
        # header lines
        for line in self.header:
            # number of values in the line
            fh.write("%d\n" % len(line))
            
            # values
            for item in line:
                fh.write("%s\n" % item[0])
                fh.write("%s\n" % item[1])
        
        # number of body lines per atom
        fh.write("%d\n" % len(self.body))
        
        # body lines
        for line in self.body:
            # number of values in the line
            fh.write("%d\n" % len(line))
            
            # values
            for item in line:
                fh.write("%s\n" % item[0])
                fh.write("%s\n" % item[1])
                fh.write("%d\n" % item[2])
    
    def read(self, fh):
        """
        Read the format from a file
        
        """
        # name
        #TODO: should check the name is unique!! or do this on the FileFormats object after finished read
        self.name = fh.readline().rstrip("\n")
        
        # delimiter
        self.setDelimiter(fh.readline().rstrip("\n"))
        
        # atom index offset
        self.setAtomIndexOffset(int(fh.readline()))
        
        # linked file
        line = fh.readline().rstrip("\n")
        if not len(line):
            self.linkedName = None
        else:
            self.setLinkedName(line)
        
        # number of header lines
        numHeaderLines = int(fh.readline())
        
        # header lines
        for _ in xrange(numHeaderLines):
            self.newHeaderLine()
            
            # number of values in the line
            numValues = int(fh.readline())
            
            # values
            for _ in xrange(numValues):
                key = fh.readline().rstrip("\n")
                typecode = fh.readline().rstrip("\n")
                self.addHeaderValue(key, typecode)
        
        # number of body lines
        numBodyLines = int(fh.readline())
        
        # body lines
        for _ in xrange(numBodyLines):
            self.newBodyLine()
            
            # number of values in the line
            numValues = int(fh.readline())
            
            # values
            for _ in xrange(numValues):
                key = fh.readline().rstrip("\n")
                typecode = fh.readline().rstrip("\n")
                dim = int(fh.readline())
                self.addBodyValue(key, typecode, dim)
    
    def print_(self):
        """
        Print debug info about the Format
        
        """
        print "FileFormat: '%s'" % self.name
        print "  Delimiter: '%s'" % self.delimiter
        print "  Atom index offset: %d" % self.atomIndexOffset
        print "  Linked name: %r" % self.linkedName
        
        print "  Header (%d lines):" % len(self.header)
        for i, line in enumerate(self.header):
            print "    %d: " % (i,),
            for item in line:
                print "%r%s" % (item, self.delimiter),
            print
        
        print "  Body (%d lines per atom):" % len(self.body)
        for i, line in enumerate(self.body):
            print "    %d: " % (i,),
            for item in line:
                print "%r%s" % (item, self.delimiter),
            print
        
        print "  Identifier: %r" % self.getIdentifier()
    
    def getIdentifier(self):
        """
        Make identifier for this file
        
        """
        identifier = []
        for line in self.header:
            identifier.append(len(line))
        for _ in xrange(5):
            for line in self.body:
                count = 0
                for value in line:
                    count += value[2]
                identifier.append(count)
        
        return identifier
    
################################################################################

class LatticeReaderGeneric(object):
    """
    Generic format Lattice reader
    
    """
    def __init__(self, tmpLocation):
        self.tmpLocation = tmpLocation
        self.logger = logging.getLogger(__name__)
    
    def unzipFile(self, filename):
        """
        Unzip command
        
        """
        bn = os.path.basename(filename)
        root, ext = os.path.splitext(bn)
        filepath = os.path.join(self.tmpLocation, root)
        if ext == ".bz2":
            command = 'bzcat -k "%s" > "%s"' % (filename, filepath)
        
        elif ext == ".gz":
            command = 'gzip -dc "%s" > "%s"' % (filename, filepath)
        
        else:
            raise RuntimeError("File '%s' is not a zip file", filename)
        
        status = os.system(command)
        if status or not os.path.exists(filepath):
            raise RuntimeError("Unzip command failed: '%s'" % command)
        
        return filepath
    
    def checkForZipped(self, filename):
        """
        Check if file exists (unzip if required)
        
        """
        zip_exts = ('.bz2', '.gz')
        ext = os.path.splitext(filename)[1]
        filepath = None
        zipFlag = False
        if os.path.exists(filename) and ext in zip_exts:
            # unzip
            zipFlag = True
            filepath = self.unzipFile(filename)
        
        elif os.path.exists(filename):
            filepath = filename
        
        else:
            if os.path.exists(filename + '.bz2'):
                filename = filename + '.bz2'
                zipFlag = True
                filepath = self.unzipFile(filename)
            
            elif os.path.exists(filename + '.gz'):
                filename = filename + '.gz'
                zipFlag = True
                filepath = self.unzipFile(filename)
        
        if filepath is None:
            raise IOError("Could not locate file: '%s'" % filename)
        
        return filepath, zipFlag
    
    def cleanUnzipped(self, filepath, zipFlag):
        """
        Clean up unzipped file.
        
        """
        if zipFlag:
            os.unlink(filepath)
    
    def readFile(self, filename, fileFormat, rouletteIndex=None, linkedLattice=None):
        """
        Read file.
        
        """
        self.logger.info("Reading file: '%s'", filename)
        
        # check if zipped
        filepath, zipFlag = self.checkForZipped(filename)
        
        try:
            status, state = self.readFileMain(filepath, fileFormat, rouletteIndex, linkedLattice)
        
        finally:
            self.cleanUnzipped(filepath, zipFlag)
        
        if status:
            self.logger.error("Generic Lattice reader failed with error code: %d", status)
        
        return status, state
    
    def readFileMain(self, filename, fileFormat, rouletteIndex, linkedLattice):
        """
        Main read
        
        """
        # if linked then must have same NAtoms!
        linkedNAtoms = -1
        if linkedLattice is not None:
            linkedNAtoms = linkedLattice.NAtoms
        
        # call C lib
        resultDict = _latticeReaderGeneric.readGenericLatticeFile(filename, fileFormat.header, fileFormat.body,
                                                                  fileFormat.delimiter, fileFormat.atomIndexOffset,
                                                                  linkedNAtoms)
        
        print "KEYS", resultDict.keys()
        
        # create Lattice object
        lattice = Lattice()
        
        # number of atoms
        lattice.NAtoms = resultDict.pop("NAtoms")
        
        # atom ID
        lattice.atomID = resultDict.pop("atomID")
        
        # position
        lattice.pos = np.ravel(resultDict.pop("Position"))
        
        # charge
        needCharge = True
        if "Charge" in resultDict:
            lattice.charge = resultDict.pop("Charge")
            needCharge = False
        elif linkedLattice is None:
            lattice.charge = np.zeros(lattice.NAtoms, np.float64)
        
        # loop back over pos to get min/max pos
        minPos, maxPos = _latticeReaderGeneric.getMinMaxPos(lattice.pos)
        lattice.minPos = minPos
        lattice.maxPos = maxPos
        self.logger.debug("Min pos: %r", minPos)
        self.logger.debug("Max pos: %r", maxPos)
        
        # cell dimensions
        if "xdim" in resultDict:
            lattice.cellDims[0] = resultDict.pop("xdim")
        else:
            lattice.cellDims[0] = maxPos[0]
        if "ydim" in resultDict:
            lattice.cellDims[1] = resultDict.pop("ydim")
        else:
            lattice.cellDims[1] = maxPos[1]
        if "zdim" in resultDict:
            lattice.cellDims[2] = resultDict.pop("zdim")
        else:
            lattice.cellDims[2] = maxPos[2]
        self.logger.debug("Cell dimensions: %r", lattice.cellDims)
        
        # specie list
        specieList = resultDict.pop("specieList")
        specieCount = resultDict.pop("specieCount")
        needSpecie = True
        if len(specieList) and "Symbol" in resultDict:
            lattice.specieList = specieList
            lattice.specieCount = specieCount
            lattice.specie = resultDict.pop("Symbol")
            needSpecie = False
        
        # get data from linked lattice
        if linkedLattice is not None:
            if needSpecie:
                self.logger.debug("Copying specie from linked Lattice")
            if needCharge:
                self.logger.debug("Copying charge from linked Lattice")
            
            # call C lib
            numSpecies = len(linkedLattice.specieList)
            result = _latticeReaderGeneric.getDataFromLinkedLattice(int(needSpecie), numSpecies, linkedLattice.specie, 
                                                                    int(needCharge), linkedLattice.charge)
            
            if needSpecie:
                lattice.specie = result[0]
                lattice.specieCount = result[1]
            
            if needCharge:
                lattice.charge = result[2]
            
            # copy specie list
            lattice.specieList = copy.deepcopy(linkedLattice.specieList)
        
        # specie mass, etc...
        self.logger.info("Adding species to Lattice")
        numSpecies = len(lattice.specieList)
        lattice.specieMass = np.empty(numSpecies, np.float64)
        lattice.specieCovalentRadius = np.empty(numSpecies, np.float64)
        lattice.specieAtomicNumber = np.empty(numSpecies, np.int32)
        lattice.specieRGB = np.empty((numSpecies, 3), np.float64)
        for i in xrange(numSpecies):
            lattice.specieMass[i] = elements.atomicMass(lattice.specieList[i])
            lattice.specieCovalentRadius[i] = elements.covalentRadius(lattice.specieList[i])
            lattice.specieAtomicNumber[i] = elements.atomicNumber(lattice.specieList[i])
            rgbtemp = elements.RGB(lattice.specieList[i])
            lattice.specieRGB[i][0] = rgbtemp[0]
            lattice.specieRGB[i][1] = rgbtemp[1]
            lattice.specieRGB[i][2] = rgbtemp[2]
            self.logger.info("%d %s (%s) atoms", lattice.specieCount[i], lattice.specieList[i], elements.atomName(lattice.specieList[i]))
        
        # read what's left in resultDict: scalars and vectors
        for key, data in resultDict.iteritems():
            if len(data.shape) == 1 and data.shape[0] == lattice.NAtoms:
                self.logger.debug("Saving '%s' scalar data to Lattice", key)
                lattice.scalarsDict[key] = data
            
            elif len(data.shape) == 2 and data.shape[0] == lattice.NAtoms and data.shape[1] == 3:
                self.logger.debug("Saving '%s' vector data to Lattice", key)
                lattice.vectorsDict[key] = data
            
            else:
                raise RuntimeError("Unrecognised shape data extracted from lattice: %s (%r)" % (key, data.shape))
        
        return 0, lattice
