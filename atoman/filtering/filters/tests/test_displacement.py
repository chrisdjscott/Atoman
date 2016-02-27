
"""
Unit tests for the displacement filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....system import lattice
from .. import displacementFilter
from .. import base


################################################################################

class TestDisplacementFilter(unittest.TestCase):
    """
    Test displacmenet filter
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice
        self.ref = lattice.Lattice()
        self.ref.addAtom("He", [0,0,0], 0)
        self.ref.addAtom("He", [0,0,4], 0)
        self.ref.addAtom("He", [2,0,0], 0)
        self.ref.addAtom("He", [0,2,0], 0)
        self.ref.addAtom("He", [4,0,0], 0)
        self.ref.addAtom("He", [0,0,2], 0)
        self.ref.addAtom("He", [0,4,0], 0)
        self.ref.addAtom("He", [4,4,4], 0)
        self.ref.addAtom("He", [99,99,99], 0)
        
        self.inp = lattice.Lattice()
        self.inp.addAtom("He", [1,0,0], 0) # 1.0
        self.inp.addAtom("He", [0,0,6], 0) # 2.0
        self.inp.addAtom("He", [2,1,2], 0) # 2.2
        self.inp.addAtom("He", [0,2,0], 0) # 0.0
        self.inp.addAtom("He", [4,0,1], 0) # 1.0
        self.inp.addAtom("He", [1,0,2], 0) # 1.0
        self.inp.addAtom("He", [0,4,2], 0) # 2.0
        self.inp.addAtom("He", [8,8,8], 0) # 6.9
        self.inp.addAtom("He", [99,0,99], 0) # 1.0
        
        self.inp2 = lattice.Lattice()
        self.inp2.addAtom("He", [1,0,0], 0)
        
        # filter
        self.filter = displacementFilter.DisplacementFilter("Displacement")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.ref = None
        self.inp = None
        self.filter = None
    
    def test_displacementFilter(self):
        """
        Displacement
        
        """
        # TEST 1
        
        # settings
        settings = displacementFilter.DisplacementFilterSettings()
        settings.updateSetting("minDisplacement", 1.1)
        settings.updateSetting("maxDisplacement", 6.0)
        settings.updateSetting("filteringEnabled", True)
        
        # set PBC
        self.inp.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.inp
        filterInput.refState = self.ref
        visibleAtoms = np.arange(self.inp.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 3)
        
        # make sure correct atoms selected
        self.assertTrue(1 in visibleAtoms)
        self.assertTrue(2 in visibleAtoms)
        self.assertTrue(6 in visibleAtoms)
        
        # TEST 2
        
        # settings
        settings = displacementFilter.DisplacementFilterSettings()
        settings.updateSetting("minDisplacement", 1.1)
        settings.updateSetting("maxDisplacement", 6.0)
        settings.updateSetting("filteringEnabled", False)
        
        # set PBC
        self.inp.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.inp
        filterInput.refState = self.ref
        visibleAtoms = np.arange(self.inp.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 9)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(2 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(5 in visibleAtoms)
        self.assertTrue(8 in visibleAtoms)
        
        # check scalars
        scalarsDict = result.getScalars()
        self.assertTrue("Displacement" in scalarsDict)
        scalars = scalarsDict["Displacement"]
        self.assertAlmostEqual(scalars[0], 1.0, places=3)
        self.assertAlmostEqual(scalars[1], 2.0, places=3)
        self.assertAlmostEqual(scalars[2], 2.236, places=3)
        self.assertAlmostEqual(scalars[3], 0.0, places=3)
        self.assertAlmostEqual(scalars[4], 1.0, places=3)
        self.assertAlmostEqual(scalars[5], 1.0, places=3)
        self.assertAlmostEqual(scalars[6], 2.0, places=3)
        self.assertAlmostEqual(scalars[7], 6.928, places=3)
        self.assertAlmostEqual(scalars[8], 1.0, places=3)
        
        # TEST 3
        
        settings = displacementFilter.DisplacementFilterSettings()
        settings.updateSetting("minDisplacement", 1.1)
        settings.updateSetting("maxDisplacement", 6.0)
        settings.updateSetting("filteringEnabled", False)
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.inp2
        filterInput.refState = self.ref
        visibleAtoms = np.arange(self.inp2.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        with self.assertRaises(RuntimeError):
            self.filter.apply(filterInput, settings)
