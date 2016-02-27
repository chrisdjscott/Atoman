
"""
Unit tests for bond order filter (Q4, Q6)

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from .. import base
from .. import bondOrderFilter
from ....lattice_gen import lattice_gen_fcc, lattice_gen_bcc
from ....gui import _preferences
from six.moves import range


################################################################################

class TestBondOrderBCC(unittest.TestCase):
    """
    Test bond order BCC
    
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
        self.filter = bondOrderFilter.BondOrderFilter("Bond order")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_bondOrderBCCQ4(self):
        """
        Bond order bcc Q4
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4
        scalarsQ4 = result.getScalars()["Q4"]
        for i in range(NVis):
            self.assertAlmostEqual(0.036, scalarsQ4[i], places=3)
    
    def test_bondOrderBCCQ6(self):
        """
        Bond order bcc Q6
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q6
        scalarsQ6 = result.getScalars()["Q6"]
        for i in range(NVis):
            self.assertAlmostEqual(0.511, scalarsQ6[i], places=3)
    
    def test_bondOrderBCCQ4_4threads(self):
        """
        Bond order bcc Q4 (4 threads)
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4
        scalarsQ4 = result.getScalars()["Q4"]
        for i in range(NVis):
            self.assertAlmostEqual(0.036, scalarsQ4[i], places=3)
    
    def test_bondOrderBCCQ6_4threads(self):
        """
        Bond order bcc Q6 (4 threads)
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q6
        scalarsQ6 = result.getScalars()["Q6"]
        for i in range(NVis):
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
        # generate lattice
        args = lattice_gen_fcc.Args(sym="Au", NCells=[8,8,8], a0=4.078, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_fcc.FCCLatticeGenerator()
        status, self.lattice = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # filter
        self.filter = bondOrderFilter.BondOrderFilter("Bond order")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
    
    def test_bondOrderFCCQ4(self):
        """
        Bond order fcc Q4
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4
        scalarsQ4 = result.getScalars()["Q4"]
        for i in range(NVis):
            self.assertAlmostEqual(0.191, scalarsQ4[i], places=3)
    
    def test_bondOrderFCCQ6(self):
        """
        Bond order fcc Q6
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q6
        scalarsQ6 = result.getScalars()["Q6"]
        for i in range(NVis):
            self.assertAlmostEqual(0.575, scalarsQ6[i], places=3)
    
    def test_bondOrderFCCQ4_4threads(self):
        """
        Bond order fcc Q4 (4 threads)
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q4
        scalarsQ4 = result.getScalars()["Q4"]
        for i in range(NVis):
            self.assertAlmostEqual(0.191, scalarsQ4[i], places=3)
    
    def test_bondOrderFCCQ6_4threads(self):
        """
        Bond order fcc Q6 (4 threads)
        
        """
        # settings
        settings = bondOrderFilter.BondOrderFilterSettings()
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
        
        # call bond order filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        NVis = len(filterInput.visibleAtoms)
        self.assertEqual(NVis, self.lattice.NAtoms)
        
        # check Q6
        scalarsQ6 = result.getScalars()["Q6"]
        for i in range(NVis):
            self.assertAlmostEqual(0.575, scalarsQ6[i], places=3)
