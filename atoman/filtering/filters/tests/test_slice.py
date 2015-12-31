
"""
Unit tests for the slice filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....system import lattice
from .. import sliceFilter
from .. import base


################################################################################

class TestSliceFilter(unittest.TestCase):
    """
    Test slice filter

    """
    def setUp(self):
        """
        Called before each test

        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("He", [0,0,3], 0)
        self.lattice.addAtom("He", [1,1,1], 0)
        self.lattice.addAtom("He", [2,3,1], 0)
        self.lattice.addAtom("He", [3,0,0], 0)
        self.lattice.addAtom("He", [4,1,4], 0)
        self.lattice.addAtom("He", [1,4,0], 0)
        self.lattice.addAtom("He", [2,1.8,4], 0)
        self.lattice.PBC[:] = 1

        # filter
        self.filter = sliceFilter.SliceFilter("Slice")

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.lattice = None
        self.filter = None

    def test_slice(self):
        """
        Slice filter

        """
        # test 1
        settings = sliceFilter.SliceFilterSettings()
        settings.updateSetting("x0", 2.0)
        settings.updateSetting("y0", 2.0)
        settings.updateSetting("z0", 0.0)
        settings.updateSetting("xn", 1.0)
        settings.updateSetting("yn", 1.0)
        settings.updateSetting("zn", 0.0)
        settings.updateSetting("invert", False)
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # run filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 4)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(1 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(6 in visibleAtoms)
        
        # test 2
        settings = sliceFilter.SliceFilterSettings()
        settings.updateSetting("x0", 2.0)
        settings.updateSetting("y0", 2.0)
        settings.updateSetting("z0", 0.0)
        settings.updateSetting("xn", -1.0)
        settings.updateSetting("yn", -1.0)
        settings.updateSetting("zn", 0.0)
        settings.updateSetting("invert", True)
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # run filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 4)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(1 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(6 in visibleAtoms)
        
        # test 3
        settings = sliceFilter.SliceFilterSettings()
        settings.updateSetting("x0", 2.0)
        settings.updateSetting("y0", 2.0)
        settings.updateSetting("z0", 0.0)
        settings.updateSetting("xn", -1.0)
        settings.updateSetting("yn", -1.0)
        settings.updateSetting("zn", 0.0)
        settings.updateSetting("invert", False)
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # run filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 3)
        
        # make sure correct atoms selected
        self.assertTrue(2 in visibleAtoms)
        self.assertTrue(4 in visibleAtoms)
        self.assertTrue(5 in visibleAtoms)
