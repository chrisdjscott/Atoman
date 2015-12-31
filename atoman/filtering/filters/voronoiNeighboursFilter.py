
"""
Voronoi neighbours
==================

This filter allows you to filter atoms by the number of Voronoi
neighbours.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np

from . import base
from . import _filtering


class VoronoiNeighboursFilterSettings(base.BaseSettings):
    """
    Settings for the Voronoi neighbours filter
    
    """
    def __init__(self):
        super(VoronoiNeighboursFilterSettings, self).__init__()
        
        self.registerSetting("filteringEnabled", default=False)
        self.registerSetting("minVoroNebs", default=0)
        self.registerSetting("maxVoroNebs", default=999)


class VoronoiNeighboursFilter(base.BaseFilter):
    """
    Voronoi neighbours filter.
    
    """
    def __init__(self, *args, **kwargs):
        super(VoronoiNeighboursFilter, self).__init__(*args, **kwargs)
    
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        NScalars = filterInput.NScalars
        fullScalars = filterInput.fullScalars
        NVectors = filterInput.NVectors
        fullVectors = filterInput.fullVectors
        visibleAtoms = filterInput.visibleAtoms
        vor = filterInput.voronoiAtoms.getVoronoi()
        
        # settings
        minVoroNebs = settings.getSetting("minVoroNebs")
        maxVoroNebs = settings.getSetting("maxVoroNebs")
        filteringEnabled = settings.getSetting("filteringEnabled")
        
        # new scalars array
        scalars = np.zeros(len(visibleAtoms), dtype=np.float64)
        
        # make array containing numbers of neighbours
        num_nebs_array = vor.atomNumNebsArray()
        
        # call C lib
        NVisible = _filtering.voronoiNeighboursFilter(visibleAtoms, num_nebs_array, minVoroNebs, maxVoroNebs, scalars,
                                                      NScalars, fullScalars, filteringEnabled, NVectors, fullVectors)

        # resize visible atoms and scalars
        visibleAtoms.resize(NVisible, refcheck=False)
        scalars.resize(NVisible, refcheck=False)
        
        # make result and add scalars
        result = base.FilterResult()
        result.addScalars("Voronoi neighbours", scalars)
        
        return result
