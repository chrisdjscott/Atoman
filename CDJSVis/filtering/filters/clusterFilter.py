
"""
Cluster filter

"""
from . import base


class ClusterFilterSettings(base.BaseSettings):
    """
    Settings for the cluster filter
    
    """
    def __init__(self):
        super(ClusterFilterSettings, self).__init__()
        
        self.registerSetting("calculateVolumes", default=False)
        self.registerSetting("calculateVolumesVoro", default=True)
        self.registerSetting("calculateVolumesHull", default=False)
        self.registerSetting("hideAtoms", default=False)
        self.registerSetting("neighbourRadius", default=5.0)
        self.registerSetting("hullCol", default=[0,0,1])
        self.registerSetting("hullOpacity", default=0.5)
        self.registerSetting("minClusterSize", default=8)
        self.registerSetting("maxClusterSize", default=-1)
