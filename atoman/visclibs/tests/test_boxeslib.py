
"""
Unit tests boxeslib

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ...lattice_gen import lattice_gen_fcc
from . import _test_boxeslib


################################################################################

class TestBoxeslib(unittest.TestCase):
    """
    Test boxeslib
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generator
        gen = lattice_gen_fcc.FCCLatticeGenerator()
        
        # lattice 1
        args = lattice_gen_fcc.Args(sym="Au", NCells=[10,10,10], a0=4.64, pbcx=True, pbcy=True, pbcz=True)
        status, self.lattice1 = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # lattice 2
        args = lattice_gen_fcc.Args(sym="Au", NCells=[6,11,8], a0=4.64, pbcx=True, pbcy=True, pbcz=True)
        status, self.lattice2 = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice1 = None
        self.lattice2 = None
    
    def test_boxeslib(self):
        """
        Boxeslib
        
        """
        pbc = np.ones(3, np.int32)
        
        # call C lib for first test
        numBoxes = np.empty(3, np.int32)
        cellLengths = np.empty(3, np.float64)
        approxWidth = 5.0
        status = _test_boxeslib.test_boxes(self.lattice2.pos, self.lattice2.cellDims, pbc, approxWidth, numBoxes, cellLengths)
        
        # check exit status
        if status:
            if status == 1:
                self.fail("Setup boxes failed")
            else:
                self.fail("Failed for unknown reason")
        
        # check boxes created as expected
        self.assertEqual(numBoxes[0], 5)
        self.assertEqual(numBoxes[1], 10)
        self.assertEqual(numBoxes[2], 7)
        self.assertAlmostEqual(cellLengths[0], 5.568)
        self.assertAlmostEqual(cellLengths[1], 5.104)
        self.assertAlmostEqual(cellLengths[2], 5.302857143)
        
        
        
        
        
        
        
        
        
