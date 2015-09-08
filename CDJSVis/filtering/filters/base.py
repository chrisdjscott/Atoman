
"""
Base module for filters.

"""
import logging

################################################################################

class BaseSettings(object):
    """Filter settings object should inherit from this class."""
    def __init__(self):
        self._settings = {}
    
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
        
        # attributes
        self._scalars = {}
        self._vectors = {}
        self._text = {}
        
        # logger
        loggerName = __name__
        words = str(filterName).title().split()
        dialogName = "%sFilter" % "".join(words)
        moduleName = dialogName[:1].lower() + dialogName[1:]
        array = loggerName.split(".")
        array[-1] = moduleName
        loggerName = ".".join(array)
        self.logger = logging.getLogger(loggerName)
    
    def getText(self):
        """Return text from this filter."""
        return self._text
    
    def getScalars(self):
        """Return the current scalars."""
        return self._scalars
    
    def getVectors(self):
        """Return the current vectors."""
        return self._vectors
    
    def apply(self, *args, **kwargs):
        raise NotImplementedError("apply method not implemented")
