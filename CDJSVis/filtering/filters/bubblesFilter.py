
"""
Bubbles
=======

Locate bubbles in the system.


The following parameters apply to this filter:

.. glossary::
    
    Bubble species
        The species of the bubbles.
    
    


"""
import copy

import numpy as np

from . import base
from . import pointDefectsFilter
from . import speciesFilter
from . import clusterFilter


class BubblesFilterSettings(base.BaseSettings):
    """
    Setting for the bubbles filter.
    
    """
    def __init__(self):
        super(BubblesFilterSettings, self).__init__()
        
        self.registerSetting("bubbleSpecies", [])
        self.registerSetting("vacancyRadius", 1.3)
        self.registerSetting("vacNebRad", 4.0)
    




class BubblesFilter(base.BaseFilter):
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
        inputState = filterInput.inputState
        refState = filterInput.refState
        
        
        # settings
        bubbleSpecies = settings.getSetting("bubbleSpecies")
        if not len(bubbleSpecies):
            self.logger.warning("No bubble species have been specified therefore no bubbles can be detected!")
            visibleAtoms.resize(0, refcheck=False)
            result = base.FilterResult()
            return result
        self.logger.debug("Bubble species: %r", bubbleSpecies)
        
        
        
        # clusters of bubble species and coms?
        
        
        
        # create new lattice with bubble species removed
        self.logger.debug("%d atoms in input before removing bubble species", inputState.NAtoms)
        lattice = copy.deepcopy(inputState)
        i = 0
        while i < lattice.NAtoms:
            if lattice.atomSym(i) in bubbleSpecies:
                lattice.removeAtom(i)
            else:
                i += 1
        self.logger.debug("%d atoms in input after removing bubble species", lattice.NAtoms)
        with open("noBubbleSpecies.dat", "w") as f:
            f.write("%d\n" % lattice.NAtoms)
            f.write("%f %f %f\n" % tuple(lattice.cellDims))
            for i in xrange(lattice.NAtoms):
                f.write("%s %f %f %f %f\n" % (lattice.atomSym(i), lattice.pos[3*i], lattice.pos[3*i+1], lattice.pos[3*i+2],
                                              lattice.charge[i]))
        
        # locate clusters of vacancies
        defectsFilter = pointDefectsFilter.PointDefectsFilter("Point defects")
        defectsSettings = pointDefectsFilter.PointDefectsFilterSettings()
        defectsSettings.updateSetting("vacancyRadius", settings.getSetting("vacancyRadius"))
        defectsSettings.updateSetting("showInterstitials", False)
        defectsSettings.updateSetting("showAntisites", False)
        defectsSettings.updateSetting("findClusters", True)
        defectsSettings.updateSetting("neighbourRadius", settings.getSetting("vacNebRad"))
        defectsSettings.updateSetting("minClusterSize", 1)
        defectsSettings.updateSetting("visibleSpeciesList", lattice.specieList)
        
        defectsInput = base.FilterInput()
        defectsInput.inputState = lattice
        defectsInput.refState = refState
        defectsInput.interstitials = np.empty(lattice.NAtoms, np.int32)
        defectsInput.vacancies = np.empty(refState.NAtoms, np.int32)
        defectsInput.antisites = np.empty(refState.NAtoms, np.int32)
        defectsInput.onAntisites = np.empty(refState.NAtoms, np.int32)
        defectsInput.splitInterstitials = np.empty(3 * refState.NAtoms, np.int32)
        
        defectsResult = defectsFilter.apply(defectsInput, defectsSettings)
        vacClusters = defectsResult.getClusterList()
        self.logger.debug("%d vacancy clusters", len(vacClusters))
    
