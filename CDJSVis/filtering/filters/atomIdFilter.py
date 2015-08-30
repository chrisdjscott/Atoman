
"""
Atom ID filter

"""
from . import base


class AtomIdFilterSettings(base.BaseSettings):
    """
    Settings for the atom ID filter
    
    """
    def __init__(self):
        super(AtomIdFilterSettings, self).__init__()
        
        self.registerSetting("filterString", default="")
