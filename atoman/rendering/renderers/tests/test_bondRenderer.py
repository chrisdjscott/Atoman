
"""
Unit tests for the bond renderer

"""
import os
import unittest
import shutil
import tempfile

import numpy as np
import vtk
from vtk.util import numpy_support

from .. import bondRenderer
from ... import utils
from ....system.latticeReaders import LbomdDatReader, basic_displayError, basic_displayWarning, basic_log


################################################################################

def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "testing", path)

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

# required unless BondsOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyBondsOpts(object):
    def __init__(self):
        self.drawBonds = True
        self.bondThicknessPOV = 0.2
        self.bondThicknessVTK = 0.2
        self.bondNumSides = 5

################################################################################

class TestBondCalculator(unittest.TestCase):
    """
    Test the bond calculator

    """
    def setUp(self):
        """
        Called before each test

        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("kenny_lattice.dat"))
        if status:
            self.fail("Error reading in Lattice")
        
        # load result
        datadir = os.path.join(os.path.dirname(__file__), "data")
        self.nbonds = np.loadtxt(os.path.join(datadir, "nbonds.csv"), dtype=np.int32)
        self.bonds = np.loadtxt(os.path.join(datadir, "bonds.csv"), dtype=np.int32)
        self.bondvectors = np.loadtxt(os.path.join(datadir, "bondvectors.csv"), dtype=np.float64)

    def tearDown(self):
        """
        Called after each test

        """
        shutil.rmtree(self.tmpLocation)
        self.tmpLocation = None
        self.lattice = None
        self.nbonds = None
        self.bonds = None
        self.bondVectors = None
    
    def test_bondCalculator(self):
        """
        Bond calculator
        
        """
        # visible atoms array
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        
        # bond arrays (species list is [Si, B_, O_])
        bondMinArray = np.zeros((3, 3), dtype=np.float64)
        bondMaxArray = np.zeros((3, 3), dtype=np.float64)
        bondMaxArray[0][2] = 2.1
        bondMaxArray[2][0] = 2.1
        bondMaxArray[1][2] = 2.1
        bondMaxArray[2][1] = 2.1
        drawList = ["Si-O_", "B_-O_"]
        
        # bond calculator
        bondCalc = bondRenderer.BondCalculator()
        result = bondCalc.calculateBonds(self.lattice, visibleAtoms, bondMinArray, bondMaxArray, drawList)
        NBondsTotal, bondArray, NBondsArray, bondVectorArray, bondSpecieCounter = result
        
        # check result
        self.assertEqual(NBondsTotal, 1482)
        self.assertEqual(bondSpecieCounter[0][0], 0)
        self.assertEqual(bondSpecieCounter[1][1], 0)
        self.assertEqual(bondSpecieCounter[2][2], 0)
        self.assertEqual(bondSpecieCounter[0][1], 0)
        self.assertEqual(bondSpecieCounter[1][0], 0)
        self.assertEqual(bondSpecieCounter[0][2], 1118)
        self.assertEqual(bondSpecieCounter[2][0], 1118)
        self.assertEqual(bondSpecieCounter[1][2], 364)
        self.assertEqual(bondSpecieCounter[2][1], 364)
        self.assertTrue(np.array_equal(NBondsArray, self.nbonds), msg="NBonds arrays differ")
        self.assertTrue(np.array_equal(bondArray, self.bonds), msg="Bonds arrays differ")
        self.assertTrue(np.allclose(bondVectorArray, self.bondvectors), msg="Bond vector arrays differ")

################################################################################

class TestBondRenderer(unittest.TestCase):
    """
    Test the bond renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("kenny_lattice.dat"))
        if status:
            self.fail("Error reading in Lattice")
        
        # load result
        datadir = os.path.join(os.path.dirname(__file__), "data")
        self.nbonds = np.loadtxt(os.path.join(datadir, "nbonds.csv"), dtype=np.int32)
        self.bonds = np.loadtxt(os.path.join(datadir, "bonds.csv"), dtype=np.int32)
        self.bondvectors = np.loadtxt(os.path.join(datadir, "bondvectors.csv"), dtype=np.float64)
        
        # lut
        self.nspecies = 3
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
        shutil.rmtree(self.tmpLocation)
        self.tmpLocation = None
        self.lattice = None
        self.nbonds = None
        self.bonds = None
        self.bondVectors = None
    
    def test_bondRenderer(self):
        """
        Bond renderer
        
        """
        # visible atoms array
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        
        # options forms
        colouringOptions = DummyColouringOpts()
        bondsOptions = DummyBondsOpts()
        
        # scalars
        scalarsData = utils.NumpyVTKData(self.lattice.specie, name="colours")
        
        # render
        bondRend = bondRenderer.BondRenderer()
        bondRend.render(self.lattice, visibleAtoms, self.nbonds, self.bonds, self.bondvectors,
                        scalarsData, colouringOptions, bondsOptions, self.lut)
        
        # check result is correct type
        self.assertIsInstance(bondRend.getActor(), utils.ActorObject)
