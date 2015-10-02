
"""
Base module for filters.

"""
import logging

import numpy as np


################################################################################

class FilterResult(object):
    """
    Result object returned by a filter.
    
    """
    def __init__(self):
        self._clusterList = []
        self._bubbleList = []
        self._structureCounterDict = {}
        self._structureCounterDictName = None
        self._scalars = {}
        self._vectors = {}
        self._text = {}
    
    def addScalars(self, name, scalars):
        """Add the given scalars."""
        self._scalars[name] = scalars
    
    def getScalars(self):
        """Return scalars."""
        return self._scalars
    
    def addVectors(self, name, vectors):
        """Add the given vectors."""
        self._vectors[name] = vectors
    
    def hasStructureCounterDict(self):
        """Returns True if there is a structure counter dict."""
        return True if self._structureCounterDictName is not None else False
    
    def setStructureCounterName(self, name):
        """Set the name for the structure counter."""
        self._structureCounterDictName = name
    
    def addStructureCount(self, structure, count):
        """Add structure counter dict."""
        self._structureCounterDict[structure] = count
    
    def getStructureCounterName(self):
        """Return the structure counter name."""
        return self._structureCounterDictName
    
    def getStructureCounterDict(self):
        """Return the structure counter dict."""
        return self._structureCounterDict
    
    def setClusterList(self, clusterList):
        """Set the cluster list."""
        self._clusterList = clusterList
    
    def getClusterList(self):
        """Return the cluster list."""
        return self._clusterList
    
    def hasClusterList(self):
        """Returns True if the cluster list is not empty."""
        return True if len(self._clusterList) else False
    
    def setBubbleList(self, bubbleList):
        """Set the bubble list."""
        self._bubbleList = bubbleList
    
    def getBubbleList(self):
        """Return the bubble list."""
        return self._bubbleList
    
    def hasBubbleList(self):
        """Returns True if the bubble list is not empty."""
        return True if len(self._bubbleList) else False

################################################################################

class FilterInput(object):
    """
    Input object for filters.
    
    """
    def __init__(self):
        self.visibleAtoms = None
        self.inputState = None
        self.refState = None
        self.fullScalars = np.empty(0, np.float64)
        self.NScalars = 0
        self.fullVectors = np.empty(0, np.float64)
        self.NVectors = 0
        self.ompNumThreads = 1
        self.voronoiOptions = None
        self.bondDict = None
        self.voronoi = None
        self.driftCompensation = False
        self.driftVector = np.zeros(3, np.float64)
        self.vacancies = np.empty(0, np.float64)
        self.interstitials = np.empty(0, np.float64)
        self.splitInterstitials = np.empty(0, np.float64)
        self.antisites = np.empty(0, np.float64)
        self.onAntisites = np.empty(0, np.float64)
        self.defectFilterSelected = False

################################################################################

class BaseSettings(object):
    """Filter settings object should inherit from this class."""
    def __init__(self):
        self._settings = {}
    
    def printSettings(self, func=None):
        """Print the settings."""
        if not callable(func):
            def func(text):
                print text
        for key in sorted(self._settings.keys()):
            value = self._settings[key]
            func("%s => %r" % (key, value))
    
    def registerSetting(self, name, default=None):
        """Register the given setting."""
        if name in self._settings:
            raise ValueError("Specified setting '{0}' already exists!".format(name))
        
        self._settings[name] = default
    
    def updateSetting(self, name, value):
        """Update the given setting with the specified value."""
        if name not in self._settings:
            raise ValueError("Specified setting '{0}' does not exist!".format(name))
        
        self._settings[name] = value
    
    def updateSettingArray(self, name, index, value):
        """Update the given setting with the specified value."""
        if name not in self._settings:
            raise ValueError("Specified setting '{0}' does not exist!".format(name))
        
        if index >= len(self._settings[name]):
            raise IndexError("Specified index is out of range: {0} >= {1}".format(index, len(self._settings[name])))
        
        self._settings[name][index] = value
    
    def getSetting(self, name):
        """Return the current value of the given setting."""
        if name not in self._settings:
            raise ValueError("Specified setting '{0}' does not exist!".format(name))
        
        value = self._settings[name]
        
        return value

################################################################################

class BaseFilter(object):
    """Filters should inherit from this object."""
    def __init__(self, filterName):
        # filter name
        self.filterName = filterName
        
        # logger
        loggerName = __name__
        words = str(filterName).title().split()
        dialogName = "%sFilter" % "".join(words)
        moduleName = dialogName[:1].lower() + dialogName[1:]
        array = loggerName.split(".")
        array[-1] = moduleName
        loggerName = ".".join(array)
        self.logger = logging.getLogger(loggerName)
        
        # attributes
        self.requiresVoronoi = False
    
    def apply(self, *args, **kwargs):
        raise NotImplementedError("apply method not implemented")
