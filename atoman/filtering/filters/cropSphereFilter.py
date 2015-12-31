
"""
Crop sphere
===========

This filter allows you to crop all atoms within a sphere from the lattice or,
alternatively, the inverse.

Parameters are:

.. glossary::
    
    x centre
        The x value for the centre of the sphere.

    y centre
        The y value for the centre of the sphere.
    
    z centre
        The z value for the centre of the sphere.
    
    Radius
        The radius of the sphere.
    
    Invert selection
        Instead of cropping atoms within the sphere, crop those outside it.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
from . import base
from . import _filtering


class CropSphereFilterSettings(base.BaseSettings):
    """
    Settings for the crop sphere filter
    
    """
    def __init__(self):
        super(CropSphereFilterSettings, self).__init__()
        
        self.registerSetting("invertSelection", default=False)
        self.registerSetting("xCentre", default=0.0)
        self.registerSetting("yCentre", default=0.0)
        self.registerSetting("zCentre", default=0.0)
        self.registerSetting("radius", default=1.0)


class CropSphereFilter(base.BaseFilter):
    """
    Crop sphere filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        lattice = filterInput.inputState
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        
        # settings
        xCentre = settings.getSetting("xCentre")
        yCentre = settings.getSetting("yCentre")
        zCentre = settings.getSetting("zCentre")
        radius = settings.getSetting("radius")
        invertSelection = int(settings.getSetting("invertSelection"))
        
        # call C library
        NVisible = _filtering.cropSphereFilter(visibleAtoms, lattice.pos, xCentre, yCentre, zCentre, 
                                               radius, lattice.cellDims, lattice.PBC, invertSelection, 
                                               NScalars, fullScalars, NVectors, fullVectors)

        # resize visible atoms
        visibleAtoms.resize(NVisible, refcheck=False)
        
        # result
        result = base.FilterResult()
        
        return result
