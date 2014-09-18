
"""
Unit tests for filtering C module

"""
import unittest

import numpy as np

from ..visclibs import filtering


################################################################################

class TestSpecieFilterC(unittest.TestCase):
    """
    Test specie filter
    
    """
    def test_specieFilter(self):
        """
        Specie filter
        
        """
        N = 5
        specieArray = np.asarray([0,0,1,0,1], dtype=np.int32)
        NScalars = 0
        fullScalars = np.asarray([], dtype=np.float64)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([1], dtype=np.int32)
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
        self.assertEqual(nvis, 2)
        self.assertEqual(visibleAtoms[0], 2)
        self.assertEqual(visibleAtoms[1], 4)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0], dtype=np.int32)
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
        self.assertEqual(nvis, 3)
        self.assertEqual(visibleAtoms[0], 0)
        self.assertEqual(visibleAtoms[1], 1)
        self.assertEqual(visibleAtoms[2], 3)
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0,1], dtype=np.int32)
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
        self.assertEqual(nvis, 5)
        self.assertEqual(visibleAtoms[0], 0)
        self.assertEqual(visibleAtoms[1], 1)
        self.assertEqual(visibleAtoms[2], 2)
        self.assertEqual(visibleAtoms[3], 3)
        self.assertEqual(visibleAtoms[4], 4)
        
        visibleSpecieArray = np.asarray([], dtype=np.int32)
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
        self.assertEqual(nvis, 0)
    
    def test_specieFilterFullScalars(self):
        """
        Specie filter full scalars
        
        """
        N = 5
        specieArray = np.asarray([0,0,1,0,1], dtype=np.int32)
        NScalars = 2
        
        visibleAtoms = np.arange(N, dtype=np.int32)
        visibleSpecieArray = np.asarray([0], dtype=np.int32)
        fullScalars = np.arange(NScalars*N, dtype=np.float64)
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
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
        nvis = filtering.specieFilter(visibleAtoms, visibleSpecieArray, specieArray, NScalars, fullScalars)
        self.assertEqual(nvis, 2)
        self.assertEqual(visibleAtoms[0], 2)
        self.assertEqual(visibleAtoms[1], 4)
        self.assertEqual(fullScalars[0], 2)
        self.assertEqual(fullScalars[1], 4)
        self.assertEqual(fullScalars[5], 7)
        self.assertEqual(fullScalars[6], 9)
