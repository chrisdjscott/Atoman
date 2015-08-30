
"""
Coordination number filter

"""
from . import base


class CoordinationNumberFilterSettings(base.BaseSettings):
    """
    Settings for the coordination number filter
    
    """
    def __init__(self):
        super(CoordinationNumberFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minCoordNum", default=0)
        self.registerSetting("maxCoordNum", default=100)
