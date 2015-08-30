
"""
Base module for filters.

"""

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
    
