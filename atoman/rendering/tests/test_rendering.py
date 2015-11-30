
"""
Unit tests for rendering.c

"""
import unittest

import numpy as np

from .. import _rendering


################################################################################

class TestRendering(unittest.TestCase):
    """
    Test _rendering

    """
    def test_countVisibleBySpecie(self):
        """
        Rendering countVisibleBySpecies
        
        """
        # set up inputs
        visatoms = np.asarray([1, 3, 4, 5, 8, 9], dtype=np.int32)
        nspecies = 4
        specie = np.asarray([0, 1, 0, 3, 1, 1, 0, 1, 3, 0, 0, 1, 1], dtype=np.int32)
        
        # call function
        visibleSpeciesCount = _rendering.countVisibleBySpecie(visatoms, nspecies, specie)
        
        # check result
        self.assertEqual(len(visibleSpeciesCount), 4)
        self.assertEqual(visibleSpeciesCount[0], 1)
        self.assertEqual(visibleSpeciesCount[1], 3)
        self.assertEqual(visibleSpeciesCount[2], 0)
        self.assertEqual(visibleSpeciesCount[3], 2)
    
    def test_makeVisibleRadiusArray(self):
        """
        Rendering makeVisibleRadiusArray
        
        """
        # set up inputs
        visatoms = np.asarray([1, 3, 4, 5, 8, 9], dtype=np.int32)
        specie = np.asarray([0, 1, 0, 3, 1, 1, 0, 1, 3, 0, 0, 1, 1], dtype=np.int32)
        specieCovRad = np.asarray([1.5, 2.5, 3.5, 4.5], dtype=np.float64)
        
        # call function
        radius = _rendering.makeVisibleRadiusArray(visatoms, specie, specieCovRad)
        
        # check result
        self.assertEqual(len(radius), len(visatoms))
        for i, index in enumerate(visatoms):
            self.assertEqual(radius[i], specieCovRad[specie[index]])
    
    def test_makeVisibleScalarArray(self):
        """
        Rendering makeVisibleScalarArray
        
        """
        # set up inputs
        visatoms = np.asarray([1, 3, 4, 5, 8, 9], dtype=np.int32)
        scalars = np.asarray([0.8, 1.2, 0.1, 3, 1, 1.8, 0, 1.9, 3.7, 0.22, 0.85, 1.38, 1], dtype=np.float64)
        
        # call function
        visScalars = _rendering.makeVisibleScalarArray(visatoms, scalars)
        
        # check result
        self.assertEqual(len(visScalars), len(visatoms))
        for i, index in enumerate(visatoms):
            self.assertEqual(visScalars[i], scalars[index])
    
    def test_makeVisiblePointsArray(self):
        """
        Rendering makeVisiblePointsArray
        
        """
        # set up inputs
        visatoms = np.asarray([1, 3, 4, 5, 8, 9], dtype=np.int32)
        pos = np.asarray([0.8, 1.2, 0.1, 3, 1, 1.8, 0, 1.9, 3.7, 0.22, 0.85, 1.38, 1,
                          0.8, 1.2, 0.1, 3, 1, 1.8, 0, 1.9, 3.7, 0.22, 0.85, 1.38, 1,
                          0.8, 1.2, 0.1, 3, 1, 1.8, 0, 1.9, 3.7, 0.22, 0.85, 1.38, 1])
        
        # call function
        vispoints = _rendering.makeVisiblePointsArray(visatoms, pos)
        
        # check result
        self.assertEqual(len(vispoints.shape), 2)
        self.assertEqual(vispoints.shape[0], len(visatoms))
        self.assertEqual(vispoints.shape[1], 3)
        for i, index in enumerate(visatoms):
            self.assertEqual(vispoints[i][0], pos[3 * index    ])
            self.assertEqual(vispoints[i][1], pos[3 * index + 1])
            self.assertEqual(vispoints[i][2], pos[3 * index + 2])
