
"""
Unit tests for vectors module

"""
import unittest

import numpy as np

from ..visclibs import vectors


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
        mag = vectors.magnitude(vect)
        self.assertAlmostEqual(mag, 6.1057350089895)
    
    def test_magnitudeCfNump(self):
        """
        Vector magnitude cf. NumPy
        
        """
        vect = np.random.rand(100)
        mag = vectors.magnitude(vect)
        self.assertAlmostEqual(mag, np.linalg.norm(vect))
