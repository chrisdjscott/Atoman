
"""
Slice
=====

This filter allows you to crop atoms by slicing through the lattice with a given plane.
Atoms on one side of the plane will be cropped. For convenience there is an option to
show the current position of the slice plane.

Parameters for this filter:

.. glossary::

    x0
        The *x* value of the centre of the plane
    
    y0
        The *y* value of the centre of the plane
    
    z0
        The *z* value of the centre of the plane
    
    xn
        The *x* value of the normal of the plane
    
    yn
        The *y* value of the normal of the plane
    
    zn
        The *z* value of the normal of the plane
    
    Invert
        Invert the selection of atoms.

"""
from . import base
from . import _filtering


class SliceFilterSettings(base.BaseSettings):
    """
    Settings for the slice filter
    
    """
    def __init__(self):
        super(SliceFilterSettings, self).__init__()
        
        self.registerSetting("x0", default=0.0)
        self.registerSetting("y0", default=0.0)
        self.registerSetting("z0", default=0.0)
        self.registerSetting("xn", default=1.0)
        self.registerSetting("yn", default=0.0)
        self.registerSetting("zn", default=0.0)
        self.registerSetting("invert", default=False)


class SliceFilter(base.BaseFilter):
    """
    Slice filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        defectFilterSelected = filterInput.defectFilterSelected
        
        # settings
        x0 = settings.getSetting("x0")
        y0 = settings.getSetting("y0")
        z0 = settings.getSetting("z0")
        xn = settings.getSetting("xn")
        yn = settings.getSetting("yn")
        zn = settings.getSetting("zn")
        invert = settings.getSetting("invert")
        
        # filter
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
            
            # call C lib
            self.logger.debug("Calling sliceDefectsFilter C function")
            result = _filtering.sliceDefectsFilter(interstitials, vacancies, antisites, onAntisites, splitInterstitials,
                                                   inp.pos, ref.pos, x0, y0, z0, xn, yn, zn, invert)
            
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
            
            # call C lib
            self.logger.debug("Calling sliceFilter C function")
            NVisible = _filtering.sliceFilter(visibleAtoms, lattice.pos, x0, y0, z0, xn, yn, zn, invert, NScalars, fullScalars,
                                              NVectors, fullVectors)
    
            # resize visible atoms
            visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
