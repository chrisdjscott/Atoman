
"""
Voronoi neighbours filter

"""
from . import base


class VoronoiNeighboursFilterSettings(base.BaseSettings):
    """
    Settings for the Voronoi neighbours filter
    
    """
    def __init__(self):
        super(VoronoiNeighboursFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minVoroNebs", default=0)
        self.registerSetting("maxVoroNebs", default=999)