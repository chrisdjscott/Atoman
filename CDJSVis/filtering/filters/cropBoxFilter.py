
"""
Crop box
========

Crop atoms within a box or, alternatively, the inverse.

"""
from . import base
from . import _filtering


class CropBoxFilterSettings(base.BaseSettings):
    """
    Settings for the crop box filter
    
    """
    def __init__(self):
        super(CropBoxFilterSettings, self).__init__()
        
        self.registerSetting("invertSelection", default=False)
        self.registerSetting("xEnabled", default=False)
        self.registerSetting("yEnabled", default=False)
        self.registerSetting("zEnabled", default=False)
        self.registerSetting("xmin", default=0.0)
        self.registerSetting("xmax", default=10.0)
        self.registerSetting("ymin", default=0.0)
        self.registerSetting("ymax", default=10.0)
        self.registerSetting("zmin", default=0.0)
        self.registerSetting("zmax", default=10.0)


class CropBoxFilter(base.BaseFilter):
    """
    Crop box filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        defectFilterSelected = filterInput.defectFilterSelected
        
        # settings
        xmin = settings.getSetting("xmin")
        xmax = settings.getSetting("xmax")
        ymin = settings.getSetting("ymin")
        ymax = settings.getSetting("ymax")
        zmin = settings.getSetting("zmin")
        zmax = settings.getSetting("zmax")
        xEnabled = int(settings.getSetting("xEnabled"))
        yEnabled = int(settings.getSetting("yEnabled"))
        zEnabled = int(settings.getSetting("zEnabled"))
        invertSelection = int(settings.getSetting("invertSelection"))
        
        # are we cropping defects or atoms...
        if defectFilterSelected:
            self.logger.debug("Cropping defects")
            
            # unpack inputs
            inp = filterInput.inputState
            ref = filterInput.refState
            interstitials = filterInput.interstitials
            vacancies = filterInput.vacancies
            antisites = filterInput.antisites
            onAntisites = filterInput.onAntisites
            splitInterstitials = filterInput.splitInterstitials
            
            # call C library
            result = _filtering.cropDefectsFilter(interstitials, vacancies, antisites, onAntisites, splitInterstitials, inp.pos,
                                                  ref.pos, xmin, xmax, ymin, ymax, zmin, zmax, xEnabled, yEnabled, zEnabled,
                                                  invertSelection)
            
            # unpack
            NInt, NVac, NAnt, NSplit = result
            vacancies.resize(NVac, refcheck=False)
            interstitials.resize(NInt, refcheck=False)
            antisites.resize(NAnt, refcheck=False)
            onAntisites.resize(NAnt, refcheck=False)
            splitInterstitials.resize(NSplit * 3, refcheck=False)
        
        else:
            self.logger.debug("Cropping atoms")
            
            # unpack inputs
            lattice = filterInput.inputState
            visibleAtoms = filterInput.visibleAtoms
            NScalars = filterInput.NScalars
            fullScalars = filterInput.fullScalars
            NVectors = filterInput.NVectors
            fullVectors = filterInput.fullVectors
            
            # call C library
            NVisible = _filtering.cropFilter(visibleAtoms, lattice.pos, xmin, xmax, ymin, ymax, zmin, zmax, xEnabled,
                                             yEnabled, zEnabled, invertSelection, NScalars, fullScalars, NVectors, fullVectors)
    
            # resize visible atoms
            visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
