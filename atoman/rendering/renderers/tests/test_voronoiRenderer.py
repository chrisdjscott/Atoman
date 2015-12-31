
"""
Unit tests for the Voronoi renderer

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np
import vtk

from .. import voronoiRenderer
from ... import utils
from ....filtering import voronoi
from ....system import lattice
from six.moves import range


# required unless ColouringOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyColouringOpts(object):
    def __init__(self):
        self.colourBy = "Species"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColourRGB = (1.0, 0.0, 0.0)
        self.scalarBarText = "Height in Y (A)"


# required unless VoronoiOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyVoronoiOpts(object):
    def __init__(self):
        self.dispersion = 10.0
        self.displayVoronoi = False
        self.useRadii = False
        self.opacity = 0.8
        self.outputToFile = False
        self.outputFilename = "voronoi.csv"
        self.faceAreaThreshold = 0.1


class TestVoronoiRenderer(unittest.TestCase):
    """
    Test the Voronoi renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("He", [0, 0, 0], 0)
        self.lattice.addAtom("Fe", [2, 0, 0], 0)
        self.lattice.addAtom("He", [0, 2, 0], 0)
        self.lattice.addAtom("Fe", [0, 0, 2], 0)
        self.lattice.addAtom("He", [5, 5, 5], 0)
        self.lattice.addAtom("He", [2, 2, 0], 0)
        self.lattice.addAtom("He", [2, 0, 2], 0)
        self.lattice.addAtom("He", [0, 2, 2], 0)
        self.lattice.addAtom("Fe", [2, 2, 2], 0)
        
        # visible atoms and scalars
        self.visAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        self.scalars = np.asarray([0, 1, 0, 1, 0, 0, 0, 0, 1], dtype=np.float64)
        
        # lut
        self.nspecies = 2
        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfColors(self.nspecies)
        self.lut.SetNumberOfTableValues(self.nspecies)
        self.lut.SetTableRange(0, self.nspecies - 1)
        self.lut.SetRange(0, self.nspecies - 1)
        for i in range(self.nspecies):
            self.lut.SetTableValue(i, 1, 0, 0, 1.0)
        
        # voronoi options
        self.voroOpts = DummyVoronoiOpts()
        
        # colouring options
        self.colOpts = DummyColouringOpts()
        
        # calc voronoi
        calc = voronoi.VoronoiAtomsCalculator(self.voroOpts)
        self.voro = calc.getVoronoi(self.lattice)

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.lattice = None
        self.nspecies = None
        self.visAtoms = None
        self.scalars = None
        self.lut = None
        self.voroOpts = None
        self.colOpts = None
        self.voro = None

    def test_voronoiRenderer(self):
        """
        Voronoi renderer
        
        """
        # run the renderer
        renderer = voronoiRenderer.VoronoiRenderer()
        renderer.render(self.lattice, self.visAtoms, self.scalars, self.lut, self.voro, self.voroOpts, self.colOpts)
        
        # check result is correct type
        self.assertIsInstance(renderer.getActor(), utils.ActorObject)
