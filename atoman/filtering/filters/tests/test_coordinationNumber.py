
"""
Unit tests for the coordination number filter

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ....lattice_gen import lattice_gen_fcc, lattice_gen_bcc
from .. import coordinationNumberFilter
from .. import base
from ....system.atoms import elements
from six.moves import range


################################################################################

class TestCoordinationNumber(unittest.TestCase):
    """
    Test ACNA BCC
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice (BCC)
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[10,10,10], a0=2.87, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_bcc.BCCLatticeGenerator()
        status, self.latticeBCC = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # generate lattice (FCC)
        args = lattice_gen_fcc.Args(sym="Au", NCells=[8,8,8], a0=4.078, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_fcc.FCCLatticeGenerator()
        status, self.latticeFCC = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # set bond min/max
        elements.addBond("Fe", "Fe", 2.0, 2.6)
        elements.addBond("Au", "Au", 2.5, 3.0)
        
        # filter
        self.filter = coordinationNumberFilter.CoordinationNumberFilter("Coordination number")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.latticeBCC = None
        self.latticeFCC = None
        self.filter = None
    
    def test_coordinationNumber(self):
        """
        Coordination number
        
        """
        # FIRST BCC
        
        # settings
        settings = coordinationNumberFilter.CoordinationNumberFilterSettings()
        
        # set PBC
        self.latticeBCC.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.latticeBCC
        filterInput.visibleAtoms = np.arange(self.latticeBCC.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.latticeBCC.NAtoms)
        
        # check coordination
        scalars = result.getScalars()["Coordination number"]
        for i in range(len(filterInput.visibleAtoms)):
            self.assertEqual(8, scalars[i])
        
        # NOW FCC
        
        # settings
        settings = coordinationNumberFilter.CoordinationNumberFilterSettings()
        
        # set PBC
        self.latticeFCC.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.latticeFCC
        filterInput.visibleAtoms = np.arange(self.latticeFCC.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is same
        self.assertEqual(len(filterInput.visibleAtoms), self.latticeFCC.NAtoms)
        
        # check coordination
        scalars = result.getScalars()["Coordination number"]
        for i in range(len(filterInput.visibleAtoms)):
            self.assertEqual(12, scalars[i])
    
    def test_coordinationNumberFiltering(self):
        """
        Coordination number filtering
        
        """
        # FIRST BCC
        
        # settings
        settings = coordinationNumberFilter.CoordinationNumberFilterSettings()
        settings.updateSetting("filteringEnabled", True)
        settings.updateSetting("minCoordNum", 10)
        settings.updateSetting("maxCoordNum", 20)
        
        # set PBC
        self.latticeBCC.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.latticeBCC
        filterInput.visibleAtoms = np.arange(self.latticeBCC.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(filterInput.visibleAtoms), 0)
        
        # NOW FCC
        
        # settings
        settings = coordinationNumberFilter.CoordinationNumberFilterSettings()
        settings.updateSetting("filteringEnabled", True)
        settings.updateSetting("minCoordNum", 1)
        settings.updateSetting("maxCoordNum", 10)
        
        # set PBC
        self.latticeFCC.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.latticeFCC
        filterInput.visibleAtoms = np.arange(self.latticeFCC.NAtoms, dtype=np.int32)
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(filterInput.visibleAtoms), 0)
