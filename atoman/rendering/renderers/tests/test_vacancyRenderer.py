
"""
Tests for the vacancy renderer

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np
import vtk
from vtk.util import numpy_support

from .. import vacancyRenderer
from ... import utils
from ....filtering.filters import pointDefectsFilter
from six.moves import range


################################################################################

# required unless ColouringOptions is rewritten to have a non GUI dependent settings object
class DummyColouringOpts(object):
    def __init__(self):
        self.colourBy = "Species"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColourRGB = (1.0, 0.0, 0.0)
        self.scalarBarText = "Height in Y (A)"

################################################################################

class TestVacancyRenderer(unittest.TestCase):
    """
    Test the vacancy renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        # arrays
        points = np.asarray([[1.2,1.2,1.6],  [0,0,0], [8,8,8], [5.4,8,1], [4,1,0]], dtype=np.float64)
        scalars = np.asarray([0,0,1,0,1], dtype=np.float64)
        radii = np.asarray([1.2, 1.2, 0.8, 1.5, 1.1], dtype=np.float64)
        
        # convert to vtk arrays
        self.atomPoints = utils.NumpyVTKData(points)
        self.radiusArray = utils.NumpyVTKData(radii, name="radius")
        self.scalarsArray = utils.NumpyVTKData(scalars, name="colours")
        
        # lut
        self.nspecies = 2
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfColors(self.nspecies)
        self.lut.SetNumberOfTableValues(self.nspecies)
        self.lut.SetTableRange(0, self.nspecies - 1)
        self.lut.SetRange(0, self.nspecies - 1)
        for i in range(self.nspecies):
            self.lut.SetTableValue(i, 1, 0, 0, 1.0)

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.atomPoints = None
        self.radiusArray = None
        self.scalarsArray = None
        self.lut = None

    def test_vacancyRenderer(self):
        """
        Vacancy renderer
        
        """
        # the renderer
        renderer = vacancyRenderer.VacancyRenderer()
        
        # some settings
        colouringOptions = DummyColouringOpts()
        atomScaleFactor = 1
        settings = pointDefectsFilter.PointDefectsFilterSettings()
        
        # render atoms
        renderer.render(self.atomPoints, self.scalarsArray, self.radiusArray, self.nspecies, colouringOptions,
                        atomScaleFactor, self.lut, settings)
        
        # check result is correct type
        self.assertIsInstance(renderer.getActor(), utils.ActorObject)
