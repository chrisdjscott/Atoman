
"""
Unit tests for the crop sphere filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....system import lattice
from .. import cropSphereFilter
from .. import base


################################################################################

class TestCropSphereFilter(unittest.TestCase):
    """
    Test crop sphere filter
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("He", [0,0,0], 0)
        self.lattice.addAtom("He", [0,0,4], 0)
        self.lattice.addAtom("He", [2,0,0], 0)
        self.lattice.addAtom("He", [0,2,0], 0)
        self.lattice.addAtom("He", [4,0,0], 0)
        self.lattice.addAtom("He", [0,0,2], 0)
        self.lattice.addAtom("He", [0,4,0], 0)
        self.lattice.addAtom("He", [4,4,4], 0)
        self.lattice.addAtom("He", [99,99,99], 0)
        
        # filter
        self.filter = cropSphereFilter.CropSphereFilter("Crop sphere")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_cropBoxFilter(self):
        """
        Crop sphere
        
        """
        # TEST 1
        
        # settings - all clusters visible
        settings = cropSphereFilter.CropSphereFilterSettings()
        settings.updateSetting("xCentre", 0.0)
        settings.updateSetting("yCentre", 0.0)
        settings.updateSetting("zCentre", 0.0)
        settings.updateSetting("radius", 2.1)
        settings.updateSetting("invertSelection", False)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 4)
        
        # make sure correct atoms selected
        self.assertTrue(1 in visibleAtoms)
        self.assertTrue(4 in visibleAtoms)
        self.assertTrue(6 in visibleAtoms)
        self.assertTrue(7 in visibleAtoms)
        
        # TEST 2
        
        # settings - all clusters visible
        settings = cropSphereFilter.CropSphereFilterSettings()
        settings.updateSetting("xCentre", 0.0)
        settings.updateSetting("yCentre", 0.0)
        settings.updateSetting("zCentre", 0.0)
        settings.updateSetting("radius", 2.1)
        settings.updateSetting("invertSelection", True)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 5)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(2 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(5 in visibleAtoms)
        self.assertTrue(8 in visibleAtoms)
