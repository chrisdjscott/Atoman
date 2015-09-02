
"""
Unit tests for bond order filter (Q4, Q6)

"""
import unittest

import numpy as np

from ..lattice_gen import lattice_gen_fcc, lattice_gen_bcc
from ..filtering import bond_order


################################################################################

class TestBondOrderBCC(unittest.TestCase):
    """
    Test bond order BCC
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # lattice generator args
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[10,10,10], a0=2.87, pbcx=True, pbcy=True, pbcz=True)
        
        # lattice generator
        gen = lattice_gen_bcc.BCCLatticeGenerator()
        
        # generate lattice
        status, self.lattice = gen.generateLattice(args)
        
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove ref to lattice
        self.lattice = None
    
    def test_bondOrderBCCQ4(self):
        """
        Bond order bcc Q4
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 1
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.036, scalarsQ4[i], places=3)
    
    def test_bondOrderBCCQ6(self):
        """
        Bond order bcc Q6
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 1
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims, 
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.511, scalarsQ6[i], places=3)
    
    def test_bondOrderBCCQ4_4threads(self):
        """
        Bond order bcc Q4 (4 threads)
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 4
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.036, scalarsQ4[i], places=3)
    
    def test_bondOrderBCCQ6_4threads(self):
        """
        Bond order bcc Q6 (4 threads)
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 4
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.511, scalarsQ6[i], places=3)

################################################################################

class TestBondOrderFCC(unittest.TestCase):
    """
    Test bond order FCC
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # lattice generator args
        args = lattice_gen_fcc.Args(sym="Au", NCells=[8,8,8], a0=4.078, pbcx=True, pbcy=True, pbcz=True)
        
        # lattice generator
        gen = lattice_gen_fcc.FCCLatticeGenerator()
        
        # generate lattice
        status, self.lattice = gen.generateLattice(args)
        
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove ref to lattice
        self.lattice = None
    
    def test_bondOrderFCCQ4(self):
        """
        Bond order fcc Q4
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 1
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.191, scalarsQ4[i], places=3)
    
    def test_bondOrderFCCQ6(self):
        """
        Bond order fcc Q6
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 1
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.575, scalarsQ6[i], places=3)
    
    def test_bondOrderFCCQ4_4threads(self):
        """
        Bond order fcc Q4 (4 threads)
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 4
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.191, scalarsQ4[i], places=3)
    
    def test_bondOrderFCCQ6_4threads(self):
        """
        Bond order fcc Q6 (4 threads)
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalarsQ4 = np.zeros(self.lattice.NAtoms, np.float64)
        scalarsQ6 = np.zeros(self.lattice.NAtoms, np.float64)
        filterQ4Enabled = 0
        minQ4 = 0.0
        maxQ4 = 0.0
        filterQ6Enabled = 0
        minQ6 = 0.0
        maxQ6 = 0.0
        
        # call bond order filter
        nthreads = 4
        NVis = bond_order.bondOrderFilter(visibleAtoms, self.lattice.pos, maxBondDistance, scalarsQ4, scalarsQ6, self.lattice.cellDims,
                                          np.ones(3, np.int32), 0, np.empty(0, np.float64), filterQ4Enabled, minQ4, maxQ4, filterQ6Enabled,
                                          minQ6, maxQ6, nthreads, 0, np.empty(0, np.float64))
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertAlmostEqual(0.575, scalarsQ6[i], places=3)
