
"""
Unit tests for the vector renderer

"""
import unittest

import numpy as np
import vtk
from vtk.util import numpy_support

from .. import vectorRenderer
from ... import utils


################################################################################

# required unless ColouringOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyColouringOpts(object):
    def __init__(self):
        self.colourBy = "Species"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColourRGB = (1.0, 0.0, 0.0)
        self.scalarBarText = "Height in Y (A)"

################################################################################

# required unless VectorsOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyVectorsOptions(object):
    def __init__(self):
        self.selectedVectorsName = None
        self.vectorRadiusPOV = 0.03
        self.vectorRadiusVTK = 0.03
        self.vectorResolution = 6
        self.vectorScaleFactor = 1.0
        self.vectorNormalise = False

################################################################################

class TestVectorRenderer(unittest.TestCase):
    """
    Test the vector renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        # arrays
        points = np.asarray([[1.2,1.2,1.6],  [0,0,0], [8,8,8], [5.4,8,1], [4,1,0]], dtype=np.float64)
        scalars = np.asarray([0,0,1,0,1], dtype=np.float64)
        vectors = np.asarray([[5,1.2,4.6],  [4,0,0], [2,2,-1], [-5.4,1,2], [0,11,0]], dtype=np.float64)
        
        # convert to vtk arrays
        self.atomPoints = vtk.vtkPoints()
        self.atomPoints.SetData(numpy_support.numpy_to_vtk(points, deep=1))
        self.vectorsArray = numpy_support.numpy_to_vtk(vectors, deep=1)
        self.scalarsArray = numpy_support.numpy_to_vtk(scalars, deep=1)
        
        # lut
        self.nspecies = 2
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfColors(self.nspecies)
        self.lut.SetNumberOfTableValues(self.nspecies)
        self.lut.SetTableRange(0, self.nspecies - 1)
        self.lut.SetRange(0, self.nspecies - 1)
        for i in xrange(self.nspecies):
            self.lut.SetTableValue(i, 1, 0, 0, 1.0)

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.atomPoints = None
        self.vectorsArray = None
        self.scalarsArray = None
        self.lut = None

    def test_vectorRenderer(self):
        """
        Vector renderer
        
        """
        # the renderer
        renderer = vectorRenderer.VectorRenderer()
        
        # some settings
        colouringOptions = DummyColouringOpts()
        vectorsOptions = DummyVectorsOptions()
        
        # render atoms
        actorObj = renderer.render(self.atomPoints, self.scalarsArray, self.vectorsArray, self.nspecies, colouringOptions,
                                   vectorsOptions, self.lut)
        
        # check result is correct type
        self.assertIsInstance(actorObj, utils.ActorObject)
