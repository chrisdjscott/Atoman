
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
    
    def getSetting(self, name):
        if name not in self._settings:
            raise ValueError("Specified setting '{0}' does not exist!".format(name))
        
        value = self._settings[name]
        
        return value

################################################################################

class BaseFilter(object):
    """Filters should inherit from this object."""
    
