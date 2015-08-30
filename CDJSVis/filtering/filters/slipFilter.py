
"""
Slip filter

"""
from . import base


class SlipFilterSettings(base.BaseSettings):
    """
    Settings for the slip filter
    
    """
    def __init__(self):
        super(SlipFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minSlip", default=0.0)
        self.registerSetting("maxSlip", default=9999.0)
