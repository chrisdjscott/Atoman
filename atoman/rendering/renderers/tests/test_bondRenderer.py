
"""
Unit tests for the bond renderer

"""
import os
import unittest
import shutil
import tempfile

import numpy as np
import vtk

from .. import bondRenderer
from ... import utils
from ... import _rendering
from ....system import lattice
from ....system.latticeReaders import LbomdDatReader, basic_displayError, basic_displayWarning, basic_log


def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "testing", path)


# required unless ColouringOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyColouringOpts(object):
    def __init__(self):
        self.colourBy = "Species"
        self.heightAxis = 1
        self.minVal = 0.0
        self.maxVal = 1.0
        self.solidColourRGB = (1.0, 0.0, 0.0)
        self.scalarBarText = "Height in Y (A)"


# required unless BondsOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyBondsOpts(object):
    def __init__(self):
        self.drawBonds = True
        self.bondThicknessPOV = 0.2
        self.bondThicknessVTK = 0.2
        self.bondNumSides = 5


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
        scalars = utils.NumpyVTKData(self.lattice.specie.astype(np.float64), name="colours")
        
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
        result = bondCalc.calculateBonds(self.lattice, visibleAtoms, scalars, bondMinArray, bondMaxArray, drawList)
        bondCoords, bondVectors, bondScalars, bondSpecieCounter = result
        
        # check result
        self.assertIsInstance(bondCoords, utils.NumpyVTKData)
        self.assertIsInstance(bondVectors, utils.NumpyVTKData)
        self.assertIsInstance(bondScalars, utils.NumpyVTKData)
        coords = bondCoords.getNumpy()
        vectors = bondCoords.getNumpy()
        scalars = bondCoords.getNumpy()
        self.assertEqual(coords.shape[0], 2964)
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(vectors.shape[0], 2964)
        self.assertEqual(vectors.shape[1], 3)
        self.assertEqual(scalars.shape[0], 2964)
        self.assertEqual(bondSpecieCounter[0][0], 0)
        self.assertEqual(bondSpecieCounter[1][1], 0)
        self.assertEqual(bondSpecieCounter[2][2], 0)
        self.assertEqual(bondSpecieCounter[0][1], 0)
        self.assertEqual(bondSpecieCounter[1][0], 0)
        self.assertEqual(bondSpecieCounter[0][2], 1118)
        self.assertEqual(bondSpecieCounter[2][0], 1118)
        self.assertEqual(bondSpecieCounter[1][2], 364)
        self.assertEqual(bondSpecieCounter[2][1], 364)
        # self.assertTrue(np.array_equal(NBondsArray, self.nbonds), msg="NBonds arrays differ")
        # self.assertTrue(np.array_equal(bondArray, self.bonds), msg="Bonds arrays differ")
        # self.assertTrue(np.allclose(bondVectorArray, self.bondvectors), msg="Bond vector arrays differ")


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
        nbonds = np.loadtxt(os.path.join(datadir, "nbonds.csv"), dtype=np.int32)
        bonds = np.loadtxt(os.path.join(datadir, "bonds.csv"), dtype=np.int32)
        bondvectors = np.loadtxt(os.path.join(datadir, "bondvectors.csv"), dtype=np.float64)
        
        scalarsData = utils.NumpyVTKData(self.lattice.specie.astype(np.float64), name="colours")
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        res = _rendering.makeBondsArrays(visibleAtoms, scalarsData.getNumpy(), self.lattice.pos, nbonds, bonds,
                                         bondvectors)
        bondCoords, bondVectors, bondScalars = res
        self.bondCoords = utils.NumpyVTKData(bondCoords)
        self.bondVectors = utils.NumpyVTKData(bondVectors, name="vectors")
        self.bondScalars = utils.NumpyVTKData(bondScalars, name="colours")
        
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
        self.bondCoords = None
        self.bondVectors = None
        self.bondScalars = None
    
    def test_bondRenderer(self):
        """
        Bond renderer
        
        """
        # options forms
        colouringOptions = DummyColouringOpts()
        bondsOptions = DummyBondsOpts()
        
        # render
        bondRend = bondRenderer.BondRenderer()
        bondRend.render(self.bondCoords, self.bondVectors, self.bondScalars, self.nspecies, colouringOptions,
                        bondsOptions, self.lut)
        
        # check result is correct type
        self.assertIsInstance(bondRend.getActor(), utils.ActorObject)


class TestDisplacementVectorCalculator(unittest.TestCase):
    """
    Test the displacement vector calculator

    """
    def setUp(self):
        """
        Called before each test

        """
        self.inputState = lattice.Lattice()
        self.inputState.addAtom("Fe", (2, 1, 3), 0.0)
        self.inputState.addAtom("Cr", (8, 4, 4), 0.0)
        self.inputState.addAtom("Cr", (4, 2, 4), 0.0)
        self.inputState.addAtom("Fe", (2, 1.9, 2), 0.0)
        self.inputState.addAtom("Fe", (3, 6, 9), 0.0)
        
        self.refState = lattice.Lattice()
        self.refState.addAtom("Fe", (0, 0, 0), 0.0)
        self.refState.addAtom("Cr", (11, 2, 6), 0.0)
        self.refState.addAtom("Cr", (1, 4, 5), 0.0)
        self.refState.addAtom("Fe", (2, 2, 2), 0.0)
        self.refState.addAtom("Fe", (8, 4, 1), 0.0)
        
        self.visibleAtoms = np.asarray([0, 2, 3], dtype=np.int32)
        self.scalars = np.asarray([0, 1, 0], dtype=np.float64)

    def tearDown(self):
        """
        Called after each test

        """
        self.inputState = None
        self.refState = None
        self.visibleAtoms = None
        self.scalars = None
    
    def test_calculateDisplacementVectors(self):
        """
        Displacement vector calculator
        
        """
        # calculate
        calc = bondRenderer.DisplacmentVectorCalculator()
        result = calc.calculateDisplacementVectors(self.inputState.pos, self.refState.pos, self.inputState.PBC,
                                                   self.inputState.cellDims, self.visibleAtoms, self.scalars)
        bondCoords, bondVectors, bondScalars = result
        
        # check result
        self.assertIsInstance(bondCoords, utils.NumpyVTKData)
        self.assertIsInstance(bondVectors, utils.NumpyVTKData)
        self.assertIsInstance(bondScalars, utils.NumpyVTKData)
        
        coords = bondCoords.getNumpy()
        vectors = bondVectors.getNumpy()
        scalars = bondScalars.getNumpy()
        self.assertEqual(len(coords.shape), 2)
        self.assertEqual(coords.shape[0], 2)
        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(len(vectors.shape), 2)
        self.assertEqual(vectors.shape[0], 2)
        self.assertEqual(vectors.shape[1], 3)
        self.assertEqual(len(scalars.shape), 1)
        self.assertEqual(scalars.shape[0], 2)
        self.assertTrue(np.array_equal(coords[0], [2, 1, 3]))
        self.assertTrue(np.array_equal(coords[1], [4, 2, 4]))
        self.assertTrue(np.array_equal(vectors[0], [-2, -1, -3]))
        self.assertTrue(np.array_equal(vectors[1], [-3, 2, 1]))
        self.assertEqual(scalars[0], 0)
        self.assertEqual(scalars[1], 1)
