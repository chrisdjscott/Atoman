
"""
Voronoi volume
==============

This filter allows you to filter atoms by their Voronoi volumes.

"""
import numpy as np

from . import base
from . import _filtering


class VoronoiVolumeFilterSettings(base.BaseSettings):
    """
    Settings for the Voronoi volume filter
    
    """
    def __init__(self):
        super(VoronoiVolumeFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minVoroVol", default=0.0)
        self.registerSetting("maxVoroVol", default=9999.99)


class VoronoiVolumeFilter(base.BaseFilter):
    """
    Voronoi volume filter.
    
    """
    def __init__(self, *args, **kwargs):
        super(VoronoiVolumeFilter, self).__init__(*args, **kwargs)
        self.requiresVoronoi = True
    
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        vor = filterInput.voronoiAtoms.getVoronoi(filterInput.inputState)
        
        # settings
        minVoroVol = settings.getSetting("minVoroVol")
        maxVoroVol = settings.getSetting("maxVoroVol")
        filteringEnabled = settings.getSetting("filteringEnabled")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # make array containing volumes of atoms
        atom_volumes = vor.atomVolumesArray()
        
        # call C lib
        NVisible = _filtering.voronoiVolumeFilter(visibleAtoms, atom_volumes, minVoroVol, maxVoroVol, scalars,
                                                  NScalars, fullScalars, filteringEnabled, NVectors, fullVectors)

        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # make result and add scalars
        result = base.FilterResult()
        result.addScalars("Voronoi volume", scalars)
        
        return result
