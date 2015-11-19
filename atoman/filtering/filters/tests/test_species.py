
"""
Unit tests for the species filter

"""
import unittest

import numpy as np

from ....system import lattice
from .. import speciesFilter
from .. import base


################################################################################

class TestSpeciesFilter(unittest.TestCase):
    """
    Test species filter

    """
    def setUp(self):
        """
        Called before each test

        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("Au", [0,0,3], 0)
        self.lattice.addAtom("He", [1,1,1], 0)
        self.lattice.addAtom("He", [2,3,1], 0)
        self.lattice.addAtom("Au", [3,0,0], 0)
        self.lattice.addAtom("Au", [4,1,4], 0)
        self.lattice.addAtom("He", [1,4,0], 0)
        self.lattice.addAtom("He", [2,1.8,4], 0)
        self.lattice.addAtom("H_", [4,4,4], 0)
        self.lattice.PBC[:] = 1

        # filter
        self.filter = speciesFilter.SpeciesFilter("Species")

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.lattice = None
        self.filter = None

    def test_speciesFilter(self):
        """
        Species filter

        """
        # test 1
        settings = speciesFilter.SpeciesFilterSettings()
        settings.updateSetting("visibleSpeciesList", ["H_", "Au"])
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        
        # run filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 4)
        
        # make sure correct atoms selected
        self.assertTrue(0 in visibleAtoms)
        self.assertTrue(3 in visibleAtoms)
        self.assertTrue(4 in visibleAtoms)
        self.assertTrue(7 in visibleAtoms)
        
        # test 2
        settings = speciesFilter.SpeciesFilterSettings()
        settings.updateSetting("visibleSpeciesList", [])
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        
        # run filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 0)
