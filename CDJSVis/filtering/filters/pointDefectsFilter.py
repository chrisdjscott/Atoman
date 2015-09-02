
"""
Point defects filter

"""
from . import base


class PointDefectsFilterSettings(base.BaseSettings):
    """
    Settings for the point defects filter
    
    """
    def __init__(self):
        super(PointDefectsFilterSettings, self).__init__()
        
        self.registerSetting("vacancyRadius", default=1.3)
        self.registerSetting("showInterstitials", default=True)
        self.registerSetting("showAntisites", default=True)
        self.registerSetting("showVacancies", default=True)
        self.registerSetting("findClusters", default=False)
        self.registerSetting("neighbourRadius", default=3.5)
        self.registerSetting("minClusterSize", default=3)
        self.registerSetting("maxClusterSize", default=-1)
        self.registerSetting("hullCol", default=[0,0,1])
        self.registerSetting("hullOpacity", default=0.5)
        self.registerSetting("calculateVolumes", default=False)
        self.registerSetting("calculateVolumesVoro", default=True)
        self.registerSetting("calculateVolumesHull", default=False)
        self.registerSetting("drawConvexHulls", default=False)
        self.registerSetting("hideDefects", default=False)
        self.registerSetting("identifySplitInts", default=True)
        self.registerSetting("vacScaleSize", default=0.75)
        self.registerSetting("vacOpacity", default=0.8)
        self.registerSetting("vacSpecular", default=0.4)
        self.registerSetting("vacSpecularPower", default=10)
        self.registerSetting("useAcna", default=False)
        self.registerSetting("acnaMaxBondDistance", default=5.0)
        self.registerSetting("acnaStructureType", default=1)
        self.registerSetting("filterSpecies", default=False)
        self.registerSetting("visibleSpeciesList", default=[])
        self.registerSetting("drawDisplacementVectors", default=False)
        self.registerSetting("bondThicknessVTK", default=0.4)
        self.registerSetting("bondThicknessPOV", default=0.4)
        self.registerSetting("bondNumSides", default=5)
