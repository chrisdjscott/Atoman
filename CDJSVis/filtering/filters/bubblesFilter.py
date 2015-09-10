
"""
Bubbles
=======

Locate bubbles in the system.


The following parameters apply to this filter:

.. glossary::
    
    Bubble species
        The species of the bubbles.
    
    


"""
from . import base


class BubblesFilterSettings(base.BaseSettings):
    """
    Setting for the bubbles filter.
    
    """
    def __init__(self):
        super(BubblesFilterSettings, self).__init__()
        
        self.registerSetting("bubbleSpecies", [])
    




class BubbleFilter(base.BaseFilter):
    """
    The bubbles filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the bubbles filter."""
        if not isinstance(settings, BubblesFilterSettings):
            raise TypeError("BubbleFilter requires a settings object of type BubbleFilterSettings.")
        
        if not isinstance(filterInput, base.FilterInput):
            raise TypeError("BubbleFilter requires an input object of type FilterInput")
        
        # unpack input object
        visibleAtoms = filterInput.visibleAtoms
        inputState = filterInput.visibleAtoms
        refState = filterInput.refState
        
        
        # settings
        bubblesSpecies = settings.getSetting("bubbleSpecies")
        if not len(bubblesSpecies):
            self.logger.warning("No bubble species have been specified therefore no bubbles can be detected!")
            visibleAtoms.resize(0, refcheck=False)
            result = base.FilterResult()
            return result
        
        
        
    
    
