
"""
Unit tests for the crop box filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....system import lattice
from .. import cropBoxFilter
from .. import base


################################################################################

class TestCropBoxAtomsFilter(unittest.TestCase):
    """
    Test crop box filter
    
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
        
        
        # filter
        self.filter = cropBoxFilter.CropBoxFilter("Crop box")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_cropBoxFilter(self):
        """
        Crop box atoms
        
        """
        # TEST 1
        
        # settings - all clusters visible
        settings = cropBoxFilter.CropBoxFilterSettings()
        settings.updateSetting("xEnabled", True)
        settings.updateSetting("xmin", 2.5)
        settings.updateSetting("xmax", 9.9)
        settings.updateSetting("yEnabled", True)
        settings.updateSetting("ymin", 2.5)
        settings.updateSetting("ymax", 9.9)
        settings.updateSetting("zEnabled", True)
        settings.updateSetting("zmin", 2.5)
        settings.updateSetting("zmax", 9.9)
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
        self.assertEqual(len(visibleAtoms), 1)
        
        # make sure correct atoms selected
        self.assertTrue(7 in visibleAtoms)
        
        # TEST 2
        
        # settings - all clusters visible
        settings = cropBoxFilter.CropBoxFilterSettings()
        settings.updateSetting("xEnabled", True)
        settings.updateSetting("xmin", 2.5)
        settings.updateSetting("xmax", 9.9)
        settings.updateSetting("yEnabled", True)
        settings.updateSetting("ymin", 2.5)
        settings.updateSetting("ymax", 9.9)
        settings.updateSetting("zEnabled", True)
        settings.updateSetting("zmin", 2.5)
        settings.updateSetting("zmax", 9.9)
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
        self.assertEqual(len(visibleAtoms), 7)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(1 in visibleAtoms)
        self.assertTrue(2 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(4 in visibleAtoms)
        self.assertTrue(5 in visibleAtoms)
        self.assertTrue(6 in visibleAtoms)

################################################################################

class TestCropBoxDefectsFilter(unittest.TestCase):
    """
    Test crop box filter (defects)
    
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
        
        self.ref = lattice.Lattice()
        self.ref.addAtom("H_", [0,0,0], 0)
        self.ref.addAtom("He", [4,0,4], 0)
        self.ref.addAtom("He", [2,0,2], 0)
        self.ref.addAtom("He", [0,2,0], 0)
        self.ref.addAtom("He", [4,0,0], 0)
        self.ref.addAtom("He", [0,0,2], 0)
        self.ref.addAtom("He", [4,4,0], 0)
        self.ref.addAtom("H_", [4,4,4], 0)
        
        self.vacancies = np.asarray([1,2,3,4,5,6], dtype=np.int32)
        self.vacancies = np.asarray([1,2,3,4,5,6], dtype=np.int32)
        
        # filter
        self.filter = cropBoxFilter.CropBoxFilter("Crop box")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.ref = None
        self.filter = None
        self.vacancies = None
    
#     def test_cropBoxFilter(self):
#         """
#         Crop box defects
#         
#         """
#         # TEST 1
#         
#         # settings - all clusters visible
#         settings = cropBoxFilter.CropBoxFilterSettings()
#         settings.updateSetting("xEnabled", True)
#         settings.updateSetting("xmin", 2.5)
#         settings.updateSetting("xmax", 9.9)
#         settings.updateSetting("yEnabled", True)
#         settings.updateSetting("ymin", 2.5)
#         settings.updateSetting("ymax", 9.9)
#         settings.updateSetting("zEnabled", True)
#         settings.updateSetting("zmin", 2.5)
#         settings.updateSetting("zmax", 9.9)
#         settings.updateSetting("invertSelection", False)
#         
#         # set PBC
#         self.lattice.PBC[:] = 1
#         
#         # filter input
#         filterInput = base.FilterInput()
#         filterInput.inputState = self.lattice
#         visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
#         filterInput.visibleAtoms = visibleAtoms
#         filterInput.NScalars = 0
#         filterInput.fullScalars = np.empty(0, np.float64)
#         filterInput.NVectors = 0
#         filterInput.fullVectors = np.empty(0, np.float64)
#         
#         # call filter
#         result = self.filter.apply(filterInput, settings)
#         self.assertIsInstance(result, base.FilterResult)
#         
#         # make sure num visible is correct
#         self.assertEqual(len(visibleAtoms), 1)
#         
#         # make sure correct atoms selected
#         self.assertTrue(7 in visibleAtoms)
#         
#         # TEST 2
#         
#         # settings - all clusters visible
#         settings = cropBoxFilter.CropBoxFilterSettings()
#         settings.updateSetting("xEnabled", True)
#         settings.updateSetting("xmin", 2.5)
#         settings.updateSetting("xmax", 9.9)
#         settings.updateSetting("yEnabled", True)
#         settings.updateSetting("ymin", 2.5)
#         settings.updateSetting("ymax", 9.9)
#         settings.updateSetting("zEnabled", True)
#         settings.updateSetting("zmin", 2.5)
#         settings.updateSetting("zmax", 9.9)
#         settings.updateSetting("invertSelection", True)
#         
#         # set PBC
#         self.lattice.PBC[:] = 1
#         
#         # filter input
#         filterInput = base.FilterInput()
#         filterInput.inputState = self.lattice
#         visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
#         filterInput.visibleAtoms = visibleAtoms
#         filterInput.NScalars = 0
#         filterInput.fullScalars = np.empty(0, np.float64)
#         filterInput.NVectors = 0
#         filterInput.fullVectors = np.empty(0, np.float64)
#         
#         # call filter
#         result = self.filter.apply(filterInput, settings)
#         self.assertIsInstance(result, base.FilterResult)
#         
#         # make sure num visible is correct
#         self.assertEqual(len(visibleAtoms), 7)
#         
#         # make sure correct atoms selected
#         self.assertTrue(0 in visibleAtoms)
#         self.assertTrue(1 in visibleAtoms)
#         self.assertTrue(2 in visibleAtoms)
#         self.assertTrue(3 in visibleAtoms)
#         self.assertTrue(4 in visibleAtoms)
#         self.assertTrue(5 in visibleAtoms)
#         self.assertTrue(6 in visibleAtoms)
