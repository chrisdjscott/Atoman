
"""
Read user-specified format Lattice FILES_PER_THREAD

"""
import os
import copy
import re
import logging
import tempfile
import shutil

import numpy as np

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
    
    def read(self, filename=None):
        """
        Read from file
        
        """
        # if no file specified, assume in the data directory
        if filename is None:
            # the file name
            filename = utilities.dataPath("file_formats.IN")
            
            # if it doesn't exist we create the default one first
            if not os.path.exists(filename):
                self.logger.debug("File formats file does not exist in data dir; creating: '{0}'".format(filename))
                utilities.createDataFile("file_formats.IN", _defaultFileFormatsFile)
        
        # read file
        self.logger.debug("Reading file formats file: '{0}'".format(filename))
        with open(filename) as f:
            numFormats = int(f.readline())
            
            for _ in xrange(numFormats):
                fmt = FileFormat()
                fmt.read(f)
                
                errors = fmt.verify()
                if errors:
                    raise RuntimeError("Could not verify FileFormat '%s'!\n%s" % (fmt.name, "\n".join(errors)))
                
                self.addFileFormat(fmt)
                fmt.print_()
        
        self.checkLinkedNames()
    
    def checkLinkedNames(self):
        """
        Check that linked names exist
        
        """
        poplist = []
        names = self._fileFormats.keys()
        for fmt in self._fileFormats.values():
            if fmt.linkedName is not None:
                if fmt.name not in names:
                    self.logger.error("Linked name '%s' for format type '%s' does not exist! Removing format '%s'", fmt.linkedName, fmt.name, fmt.name)
                    poplist.append(fmt.name)
        
        for key in poplist:
            self._fileFormats.pop(key)
    
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
    
    def __contains__(self, item):
        return item in self._fileFormats

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
        
        self.logger = logging.getLogger(__name__+".FileFormat")
    
    def inHeader(self, key):
        """
        Test if the key was defined in the header
        
        """
        for line in self.header:
            for item in line:
                if item[0] == key:
                    return True
        
        return False
    
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
    
    def getDelimiter(self):
        """
        Return the delimiter
        
        """
        delim = self.delimiter
        
        if " " not in delim:
            delim += " "
        if "\n" not in delim:
            delim += "\n"
        if "\r" not in delim:
            delim += "\r"
        if "\t" not in delim:
            delim += "\t"
        
        return delim
    
    def setDelimiter(self, delim):
        """
        Set the delimiter
        
        """
        self.delimiter = delim
    
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
        for headerLine in self.header:
            for headerItem in headerLine:
                key, typecode, dim = headerItem
                
                # currently NAtoms must be in the header (this will change eventually)
                if key == "NAtoms":
                    haveNAtoms = True
        
        if not haveNAtoms:
            errors.append("Format header must contain 'NAtoms'")
        
        havePos = False
        haveAtomId = False
        haveSymbol = False
        for bodyLine in self.body:
            for bodyItem in bodyLine:
                key, typecode, dim = bodyItem
                
                # must have pos
                if key == "Position":
                    havePos = True
                elif key == "Symbol":
                    haveSymbol = True
                elif key == "atomID":
                    haveAtomId = True
        
        if not havePos:
            errors.append("Format body must contain 'Position'")
        
        if (not haveSymbol) and (self.linkedName is None or not haveAtomId):
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
        lines = []
        lines.append("File format:")
        lines.append("  Name: '%s'" % self.name)
        lines.append("  Delimiter: '%s'" % self.delimiter)
        lines.append("  Atom index offset: %d" % self.atomIndexOffset)
        lines.append("  Linked name: %r" % self.linkedName)
        lines.append("  Header (%d lines):" % len(self.header))
        for i, line in enumerate(self.header):
            nl = "    %d: " % i
            for item in line:
                nl += "%r%s" % (item, self.delimiter)
            lines.append(nl)
        
        lines.append("  Body (%d lines per atom):" % len(self.body))
        for i, line in enumerate(self.body):
            nl = "    %d: " % i
            for item in line:
                nl += "%r%s" % (item, self.delimiter)
            lines.append(nl)
        
        lines.append("  Identifier: %r" % self.getIdentifier())
        self.logger.debug("\n".join(lines))
    
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
    def __init__(self, tmpLocation=None, updateProgress=None, hideProgress=None):
        self.logger = logging.getLogger(__name__+".LatticeReaderGeneric")
        
        # create tmp dir if one isn't passed
        if tmpLocation is None:
            self.rmTmpDir = True
            self.tmpLocation = tempfile.mkdtemp()
            self.logger.debug("Created tmp directory: '%s'", self.tmpLocation)
        else:
            self.rmTmpDir = False
            self.tmpLocation = tmpLocation
        
        
        self.updateProgress = updateProgress
        self.hideProgress = hideProgress
        self.intRegex = re.compile(r'[0-9]+')
    
    def __del__(self):
        # remove the temporary directory if we created it
        if self.rmTmpDir and os.path.exists(self.tmpLocation):
            try:
                shutil.rmtree(self.tmpLocation)
            except:
                pass
    
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
        
        self.logger.debug("Running: '%s'", command)
        
        # progress bar
        if self.updateProgress is not None:
            self.updateProgress(0, 0, "Unzipping: '%s'" % bn)
        
        # run command
        status = os.system(command)
        
        # hide progress bar
        if self.hideProgress is not None:
            self.hideProgress()
        
        # handle error
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
        
        # delimiter
        delim = fileFormat.getDelimiter()
        
        # call C lib
        if self.updateProgress is None:
            resultDict = _latticeReaderGeneric.readGenericLatticeFile(filename, fileFormat.header, fileFormat.body,
                                                                      delim, fileFormat.atomIndexOffset, linkedNAtoms)
        
        else:
            try:
                bn = os.path.basename(filename)
                resultDict = _latticeReaderGeneric.readGenericLatticeFile(filename, fileFormat.header, fileFormat.body,
                                                                          delim, fileFormat.atomIndexOffset,
                                                                          linkedNAtoms, self.updateProgress, bn)
            
            finally:
                self.hideProgress()
        
        self.logger.debug("Keys: %r", resultDict.keys())
        
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
        needCellDims = False
        if "xdim" in resultDict:
            lattice.cellDims[0] = resultDict.pop("xdim")
        elif linkedLattice is None:
            lattice.cellDims[0] = maxPos[0]
        else:
            needCellDims = True
        
        if "ydim" in resultDict:
            lattice.cellDims[1] = resultDict.pop("ydim")
        elif linkedLattice is None:
            lattice.cellDims[1] = maxPos[1]
        else:
            needCellDims = True
        
        if "zdim" in resultDict:
            lattice.cellDims[2] = resultDict.pop("zdim")
        elif linkedLattice is None:
            lattice.cellDims[2] = maxPos[2]
        else:
            needCellDims = True
        
        if not needCellDims:
            self.logger.debug("Cell dimensions: %r", lattice.cellDims)
        
        # specie list
        specieList = resultDict.pop("specieList")
        specieCount = resultDict.pop("specieCount")
        needSpecie = True
        if len(specieList) and "Symbol" in resultDict:
            lattice.specieList = np.array(specieList)
            lattice.specieCount = np.array(specieCount, dtype=np.int32)
            lattice.specie = resultDict.pop("Symbol")
            needSpecie = False
        
        # get data from linked lattice
        if linkedLattice is not None:
            if needSpecie:
                self.logger.debug("Copying specie from linked Lattice")
                lattice.specie = copy.deepcopy(linkedLattice.specie)
                lattice.specieCount = copy.deepcopy(linkedLattice.specieCount)
                lattice.specieList = copy.deepcopy(linkedLattice.specieList)
            
            if needCharge:
                self.logger.debug("Copying charge from linked Lattice")
                lattice.charge = copy.deepcopy(linkedLattice.charge)
            
            if needCellDims:
                self.logger.debug("Copying cellDims from linked lattice")
                lattice.cellDims = copy.deepcopy(linkedLattice.cellDims)
        
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
        
        # read what's left in resultDict: scalars and vectors and lattice attributes
        for key, data in resultDict.iteritems():
            # take lattice attributes (Defined in header) first
            if fileFormat.inHeader(key):
                self.logger.debug("Saving '%s' attribute to Lattice (%r)", key, data)
                lattice.attributes[key] = data
            
            # now take scalars
            elif len(data.shape) == 1 and data.shape[0] == lattice.NAtoms:
                self.logger.debug("Saving '%s' scalar data to Lattice", key)
                # for now we require all scalar data to be stored as float (will change this if I have time)
                if data.dtype != np.float64:
                    lattice.scalarsDict[key] = data.astype(np.float64)
                else:
                    lattice.scalarsDict[key] = data
            
            # now take vectors
            elif len(data.shape) == 2 and data.shape[0] == lattice.NAtoms and data.shape[1] == 3:
                self.logger.debug("Saving '%s' vector data to Lattice", key)
                # for now we require all vector data to be stored as float (will change this if I have time)
                if data.dtype != np.float64:
                    lattice.vectorsDict[key] = data.astype(np.float64)
                else:
                    lattice.vectorsDict[key] = data
            
            else:
                raise RuntimeError("Unrecognised shape data extracted from lattice: %s (%r)" % (key, data.shape))
        
        # This section is specific to LKMC...
        
        # guess roulette
        stepNumber = None
        if rouletteIndex is None:
            # file name
            basename = os.path.basename(filename)
            
            # look for integers in the name
            result = self.intRegex.findall(basename)
            
            if len(result):
                try:
                    index = int(result[0])
                except ValueError:
                    rouletteIndex = None
                else:
                    stepNumber = index
                    if index > 0:
                        rouletteIndex = index - 1
        
        # attempt to read roulette file
        if rouletteIndex is not None:
            # different path?
            head = os.path.dirname(filename)
            if len(head):
                testpath = head
            else:
                testpath = None
            
            # read simulation time
            simTime = utilities.getTimeFromRoulette(rouletteIndex, testpath=testpath)
            
            if simTime is not None:
                lattice.attributes["Time"] = simTime
                self.logger.info("Detected simulation time as: %f", simTime)
            
            # get barrier
            barrier = utilities.getBarrierFromRoulette(rouletteIndex, testpath=testpath)
            if barrier is not None:
                lattice.attributes["Barrier"] = barrier
                self.logger.info("Detected barrier as: %f", barrier)
            
            # only store step if simTime or barrier were found
            if simTime is not None or barrier is not None:
                # step number
                if stepNumber is None:
                    stepNumber = rouletteIndex + 1
                lattice.attributes["KMC step"] = stepNumber
                self.logger.info("Detected KMC step as: %d", stepNumber)
        
        self.logger.debug("Lattice attribs: %r", lattice.attributes.keys())
        self.logger.debug("Lattice scalars: %r", lattice.scalarsDict.keys())
        self.logger.debug("Lattice vectors: %r", lattice.vectorsDict.keys())
        
        return 0, lattice

################################################################################
# the default file formats file
_defaultFileFormatsFile = """8
LBOMD Lattice
 
0

2
1
NAtoms
i
3
xdim
d
ydim
d
zdim
d
1
3
Symbol
i
1
Position
d
3
Charge
d
1
LBOMD REF
 
1

2
1
NAtoms
i
3
xdim
d
ydim
d
zdim
d
1
7
atomID
i
1
Symbol
i
1
Position
d
3
Kinetic energy
d
1
Potential energy
d
1
Force
d
3
Charge
d
1
LBOMD XYZ
 
1
LBOMD REF
2
1
NAtoms
i
1
Time
d
1
4
atomID
i
1
Position
d
3
Kinetic energy
d
1
Potential energy
d
1
LBOMD XYZ (Velocity)
 
1
LBOMD REF
2
1
NAtoms
i
1
Time
d
1
5
atomID
i
1
Position
d
3
Kinetic energy
d
1
Potential energy
d
1
Velocity
d
3
LBOMD XYZ (Charge)
 
1
LBOMD REF
2
1
NAtoms
i
1
Time
d
1
5
atomID
i
1
Position
d
3
Kinetic energy
d
1
Potential energy
d
1
Charge
d
1
Indenter
 
0

2
1
NAtoms
i
0
1
3
Symbol
i
1
Position
d
3
SKIP
i
1
FAILSAFE
 
1

5
1
Time
d
1
SKIP
s
3
Thermostat
s
SKIP
s
Target temperature
d
1
NAtoms
i
6
SKIP
s
SKIP
s
SKIP
s
SKIP
s
SKIP
s
SKIP
s
4
3
atomID
i
1
Atom type
i
1
Symbol
i
1
1
Position
d
3
1
Charge
d
1
1
Velocity
d
3
CASTEP XYZ
 
0

2
1
NAtoms
i
1
SKIP
s
1
2
Symbol
i
1
Position
d
3
"""
