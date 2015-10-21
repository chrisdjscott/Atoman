
"""
Unit tests for the charge filter

"""
import unittest

import numpy as np

from ....state import lattice
from .. import chargeFilter
from .. import base


################################################################################

class TestChargeFilter(unittest.TestCase):
    """
    Test charge filter
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("Au", [0,0,0], -3.0)
        self.lattice.addAtom("Au", [1,0,0], -2.0)
        self.lattice.addAtom("Au", [0,1,0], 1.0)
        self.lattice.addAtom("Au", [0,0,1], -4.0)
        self.lattice.addAtom("Au", [1,1,0], 4.0)
        self.lattice.addAtom("Au", [0,1,1], 3.0)
        self.lattice.addAtom("Au", [1,1,1], 1.0)
        self.lattice.addAtom("Au", [2,0,0], -1.0)
        self.lattice.addAtom("Au", [0,2,0], -2.0)
        self.lattice.addAtom("Au", [0,0,2], 4.0)
        
        # filter
        self.filter = chargeFilter.ChargeFilter("Charge")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_chargeFilter(self):
        """
        Charge filter
        
        """
        # settings
        settings = chargeFilter.ChargeFilterSettings()
        settings.updateSetting("minCharge", -3.5)
        settings.updateSetting("maxCharge", 0)
        
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
        filterInput.ompNumThreads = 1
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 4)
        
        # check positions are correct
        self.assertListEqual(list(self.lattice.atomPos(visibleAtoms[0])), [0,0,0])
        self.assertListEqual(list(self.lattice.atomPos(visibleAtoms[1])), [1,0,0])
        self.assertListEqual(list(self.lattice.atomPos(visibleAtoms[2])), [2,0,0])
        self.assertListEqual(list(self.lattice.atomPos(visibleAtoms[3])), [0,2,0])
