
"""
Voronoi volume filter

"""
from . import base


class VoronoiVolumeFilterSettings(base.BaseSettings):
    """
    Settings for the Voronoi volume filter
    
    """
    def __init__(self):
        super(VoronoiVolumeFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minVoroVol", default=0.0)
        self.registerSetting("maxVoroVol", default=9999.99)
