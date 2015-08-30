
"""
Species filter

"""
from . import base


class SpeciesFilterSettings(base.BaseSettings):
    """
    Settings for the species filter
    
    """
    def __init__(self):
        super(SpeciesFilterSettings, self).__init__()
        
        self.registerSetting("visibleSpeciesList", default=[])
