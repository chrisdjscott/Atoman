
"""
Unit tests for vectors module

"""
import unittest

import numpy as np

from .. import _vectors


################################################################################

class TestVectors(unittest.TestCase):
    """
    Test vectors module
    
    """
    def test_magnitude(self):
        """
        Vector magnitude
        
        """
        vect = np.asarray([1.8, 5.4, 2.2, 0.0, 0.2], dtype=np.float64)
        mag = _vectors.magnitude(vect)
        self.assertAlmostEqual(mag, 6.1057350089895)
    
    def test_magnitudeCfNump(self):
        """
        Vector magnitude cf. NumPy
        
        """
        vect = np.random.rand(100)
        mag = _vectors.magnitude(vect)
        self.assertAlmostEqual(mag, np.linalg.norm(vect))
    
    def test_separationMagnitude(self):
        """
        Vector separation magnitude
        
        """
        cellDims = np.asarray([100,0,0,0,100,0,0,0,100], dtype=np.float64)
        
        p1 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        p2 = np.asarray([11.0, 5.0, 0.0], dtype=np.float64)
        
        sepMag = _vectors.separationMagnitude(p1, p2, cellDims, np.ones(3, np.int32))
        
        self.assertAlmostEqual(sepMag, 10.81665383)
    
    def test_separationMagnitudePBC(self):
        """
        Vector separation magnitude PBC
        
        """
        cellDims = np.asarray([100,0,0,0,100,0,0,0,100], dtype=np.float64)
        
        p1 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        p2 = np.asarray([99, 99, 99], dtype=np.float64)
        
        sepMag = _vectors.separationMagnitude(p1, p2, cellDims, np.ones(3, np.int32))
        
        self.assertAlmostEqual(sepMag, 3.464101615)
    
    def test_separationVector(self):
        """
        Vector separation vector
        
        """
        cellDims = np.asarray([100,0,0,0,100,0,0,0,100], dtype=np.float64)
        
        p1 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        p2 = np.asarray([11.0, 5.0, 0.0], dtype=np.float64)
        
        sepVec = np.empty(3, np.float64)
        
        status = _vectors.separationVector(sepVec, p1, p2, cellDims, np.ones(3, np.int32))
        
        self.assertEqual(status, 0)
        self.assertEqual(sepVec[0], 10)
        self.assertEqual(sepVec[1], 4)
        self.assertEqual(sepVec[2], -1)
    
    def test_separationVectorPBC(self):
        """
        Vector separation vector PBC
        
        """
        cellDims = np.asarray([100,0,0,0,100,0,0,0,100], dtype=np.float64)
        
        p1 = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        p2 = np.asarray([99, 99, 99], dtype=np.float64)
        
        sepVec = np.empty(3, np.float64)
        
        status = _vectors.separationVector(sepVec, p1, p2, cellDims, np.ones(3, np.int32))
        
        self.assertEqual(status, 0)
        self.assertEqual(sepVec[0], -2)
        self.assertEqual(sepVec[1], -2)
        self.assertEqual(sepVec[2], -2)
