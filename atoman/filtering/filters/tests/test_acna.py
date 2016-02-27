
"""
Unit tests for ACNA filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....lattice_gen import lattice_gen_fcc, lattice_gen_bcc
from .. import acnaFilter
from .. import base
from ....gui import _preferences
from six.moves import range


################################################################################

class TestACNABCC(unittest.TestCase):
    """
    Test ACNA BCC
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[10,10,10], a0=2.87, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_bcc.BCCLatticeGenerator()
        status, self.lattice = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # filter
        self.filter = acnaFilter.AcnaFilter("ACNA")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_ACNABCC(self):
        """
        ACNA bcc
        
        """
        # settings
        settings = acnaFilter.AcnaFilterSettings()
        settings.updateSetting("maxBondDistance", 4.0)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        filterInput.visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # set number of threads
        _preferences.setNumThreads(1)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        scalars = result.getScalars()["ACNA"]
        for i in range(len(filterInput.visibleAtoms)):
            self.assertEqual(3, scalars[i])
    
    def test_ACNABCC_4threads(self):
        """
        ACNA bcc (4 threads)
        
        """
        # settings
        settings = acnaFilter.AcnaFilterSettings()
        settings.updateSetting("maxBondDistance", 4.0)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        filterInput.visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # set number of threads
        _preferences.setNumThreads(4)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        scalars = result.getScalars()["ACNA"]
        for i in range(len(filterInput.visibleAtoms)):
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
        # generate lattice
        args = lattice_gen_fcc.Args(sym="Au", NCells=[8,8,8], a0=4.078, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_fcc.FCCLatticeGenerator()
        status, self.lattice = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # filter
        self.filter = acnaFilter.AcnaFilter("ACNA")
     
    def tearDown(self):
        """
        Called after each test
         
        """
        # remove refs
        self.lattice = None
        self.filter = None
     
    def test_ACNAFCC(self):
        """
        ACNA fcc
         
        """
        # settings
        settings = acnaFilter.AcnaFilterSettings()
        settings.updateSetting("maxBondDistance", 3.8)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        filterInput.visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # set number of threads
        _preferences.setNumThreads(1)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        scalars = result.getScalars()["ACNA"]
        for i in range(len(filterInput.visibleAtoms)):
            self.assertEqual(1, scalars[i])
    
    def test_ACNAFCC_4threads(self):
        """
        ACNA fcc (4 threads)
         
        """
        # settings
        settings = acnaFilter.AcnaFilterSettings()
        settings.updateSetting("maxBondDistance", 3.8)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        filterInput.visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # set number of threads
        _preferences.setNumThreads(4)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.lattice.NAtoms)
        
        # check Q4 (all atoms same in perfect lattice...) (or should I check them all)
        scalars = result.getScalars()["ACNA"]
        for i in range(len(filterInput.visibleAtoms)):
            self.assertEqual(1, scalars[i])
