
"""
Unit tests for filtering C module

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ..filters import _filtering


################################################################################

class TestSpecieFilter(unittest.TestCase):
    """
    Test specie filter C lib
    
    """
    def test_specieFilter(self):
        """
        Specie filter C lib
        
        """
        N = 5
        specieArray = np.asarray([0,0,1,0,1], dtype=np.int32)
        NScalars = 0
        fullScalars = np.asarray([], dtype=np.float64)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([1], dtype=np.int32)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, NScalars, fullScalars)
        self.assertEqual(nvis, 2)
        self.assertEqual(visibleAtoms[0], 2)
        self.assertEqual(visibleAtoms[1], 4)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0], dtype=np.int32)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, NScalars, fullScalars)
        self.assertEqual(nvis, 3)
        self.assertEqual(visibleAtoms[0], 0)
        self.assertEqual(visibleAtoms[1], 1)
        self.assertEqual(visibleAtoms[2], 3)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0,1], dtype=np.int32)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, NScalars, fullScalars)
        self.assertEqual(nvis, 5)
        self.assertEqual(visibleAtoms[0], 0)
        self.assertEqual(visibleAtoms[1], 1)
        self.assertEqual(visibleAtoms[2], 2)
        self.assertEqual(visibleAtoms[3], 3)
        self.assertEqual(visibleAtoms[4], 4)
        
        visibleSpecieArray = np.asarray([], dtype=np.int32)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, NScalars, fullScalars)
        self.assertEqual(nvis, 0)
    
    def test_specieFilterFullScalars(self):
        """
        Specie filter C lib full scalars
        
        """
        N = 5
        specieArray = np.asarray([0,0,1,0,1], dtype=np.int32)
        NScalars = 2
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0], dtype=np.int32)
        fullScalars = np.arange(NScalars*N, dtype=np.float64)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, 0, np.asarray([], dtype=np.float64))
        self.assertEqual(nvis, 3)
        self.assertEqual(visibleAtoms[0], 0)
        self.assertEqual(visibleAtoms[1], 1)
        self.assertEqual(visibleAtoms[2], 3)
        self.assertEqual(fullScalars[0], 0)
        self.assertEqual(fullScalars[1], 1)
        self.assertEqual(fullScalars[2], 3)
        self.assertEqual(fullScalars[5], 5)
        self.assertEqual(fullScalars[6], 6)
        self.assertEqual(fullScalars[7], 8)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([1], dtype=np.int32)
        fullScalars = np.arange(NScalars*N, dtype=np.float64)
        nvis = _filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars, 0, np.asarray([], dtype=np.float64))
        self.assertEqual(nvis, 2)
        self.assertEqual(visibleAtoms[0], 2)
        self.assertEqual(visibleAtoms[1], 4)
        self.assertEqual(fullScalars[0], 2)
        self.assertEqual(fullScalars[1], 4)
        self.assertEqual(fullScalars[5], 7)
        self.assertEqual(fullScalars[6], 9)

################################################################################

class TestCalculateDrift(unittest.TestCase):
    """
    Test calculate drift vector
    
    """
    def test_calculateDrift(self):
        """
        Calculate drift vector
        
        """
        N = 2
        p = np.asarray([2.0, 0.0, 99.0, 98.0, 1.0, 1.0], dtype=np.float64)
        r = np.asarray([3.0, 0.5, 1.0, 97.0, 99.0, 0.5], dtype=np.float64)
        cellDims = np.asarray([100, 100, 100], dtype=np.float64)
        pbc = np.ones(3, np.int32)
        driftVector = np.zeros(3, np.float64)
        
        ret = _filtering.calculate_drift_vector(N, p, r, cellDims, pbc, driftVector)
        
        self.assertEqual(ret, 0)
        self.assertEqual(driftVector[0], 0.0)
        self.assertEqual(driftVector[1], 0.75)
        self.assertEqual(driftVector[2], -0.75)
