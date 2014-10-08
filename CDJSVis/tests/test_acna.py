
"""
Unit tests for ACNA filter

"""
import unittest

import numpy as np

from ..lattice_gen import lattice_gen_fcc, lattice_gen_bcc
from ..filtering import acna
from ..filtering import filterer


################################################################################

class TestACNABCC(unittest.TestCase):
    """
    Test ACNA BCC
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # lattice generator args
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[10,10,10], a0=2.87, pbcx=True, pbcy=True, pbcz=True, quiet=False)
        
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
    
    def test_ACNABCC(self):
        """
        ACNA bcc
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalars = np.zeros(self.lattice.NAtoms, np.float64)
        filteringEnabled = 0
        structureVisibility = np.ones(len(filterer.Filterer.knownStructures), dtype=np.int32)
        counters = np.zeros(7, np.int32)
        
        # call bond order filter
        nthreads = 1
        NVis = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, self.lattice.pos, scalars, self.lattice.minPos, self.lattice.maxPos, 
                                                    self.lattice.cellDims, np.ones(3, np.int32), 0, np.empty(0, np.float64), 
                                                    maxBondDistance, counters, filteringEnabled, structureVisibility, nthreads)
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertEqual(3, scalars[i])
    
    def test_ACNABCC_4threads(self):
        """
        ACNA bcc (4 threads)
        
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 4.0
        scalars = np.zeros(self.lattice.NAtoms, np.float64)
        filteringEnabled = 0
        structureVisibility = np.ones(len(filterer.Filterer.knownStructures), dtype=np.int32)
        counters = np.zeros(7, np.int32)
        
        # call bond order filter
        nthreads = 4
        NVis = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, self.lattice.pos, scalars, self.lattice.minPos, self.lattice.maxPos, 
                                                    self.lattice.cellDims, np.ones(3, np.int32), 0, np.empty(0, np.float64), 
                                                    maxBondDistance, counters, filteringEnabled, structureVisibility, nthreads)
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertEqual(3, scalars[i])

################################################################################

class TestACNAFCC(unittest.TestCase):
    """
    Test ACNA FCC
     
    """
    def setUp(self):
        """
        Called before each test
         
        """
        # lattice generator args
        args = lattice_gen_fcc.Args(sym="Au", NCells=[8,8,8], a0=4.078, pbcx=True, pbcy=True, pbcz=True, quiet=False)
         
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
     
    def test_ACNAFCC(self):
        """
        ACNA fcc
         
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalars = np.zeros(self.lattice.NAtoms, np.float64)
        filteringEnabled = 0
        structureVisibility = np.ones(len(filterer.Filterer.knownStructures), dtype=np.int32)
        counters = np.zeros(7, np.int32)
         
        # call bond order filter
        nthreads = 1
        NVis = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, self.lattice.pos, scalars, self.lattice.minPos, self.lattice.maxPos, 
                                                    self.lattice.cellDims, np.ones(3, np.int32), 0, np.empty(0, np.float64), 
                                                    maxBondDistance, counters, filteringEnabled, structureVisibility, nthreads)
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertEqual(1, scalars[i])
    
    def test_ACNAFCC_4threads(self):
        """
        ACNA fcc (4 threads)
         
        """
        # arguments
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        maxBondDistance = 3.8
        scalars = np.zeros(self.lattice.NAtoms, np.float64)
        filteringEnabled = 0
        structureVisibility = np.ones(len(filterer.Filterer.knownStructures), dtype=np.int32)
        counters = np.zeros(7, np.int32)
         
        # call bond order filter
        nthreads = 4
        NVis = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, self.lattice.pos, scalars, self.lattice.minPos, self.lattice.maxPos, 
                                                    self.lattice.cellDims, np.ones(3, np.int32), 0, np.empty(0, np.float64), 
                                                    maxBondDistance, counters, filteringEnabled, structureVisibility, nthreads)
        
        # make sure num visible is same
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        for i in xrange(NVis):
            self.assertEqual(1, scalars[i])
