
"""
Unit tests for the point defects filter

"""
import copy
import unittest

import numpy as np

from ....lattice_gen import lattice_gen_pu3ga
from ....system import lattice
from .. import pointDefectsFilter
from .. import base


################################################################################

class TestPointDefectsFilter(unittest.TestCase):
    """
    Test point defects
    
    - by type -- turning each off
    - split ints
    - vac rad ?
    - acna
    - big example too
    - filter species
    - clusters
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate reference lattice
        args = lattice_gen_pu3ga.Args(NCells=[10,10,10], pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_pu3ga.Pu3GaLatticeGenerator()
        status, self.ref = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # make input lattice
        self.inp = copy.deepcopy(self.ref)
        self.inp.PBC[:] = 1
        # vacancy and split
        self.inp.pos[0] = self.inp.pos[12]
        self.inp.pos[1] = self.inp.pos[13]
        self.inp.pos[2] = self.inp.pos[14] - 1.4
        self.inp.pos[14] += 1.4
        # antisite
        self.inp.specie[30] = 1
        
        # filter
        self.filter = pointDefectsFilter.PointDefectsFilter("Point defects")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.inp = None
        self.ref = None
        self.filter = None
    
    def makeFilterInput(self):
        """Make filter input"""
        filterInput = base.FilterInput()
        filterInput.inputState = self.inp
        filterInput.refState = self.ref
        filterInput.interstitials = np.empty(self.inp.NAtoms, dtype=np.int32)
        filterInput.vacancies = np.empty(self.ref.NAtoms, dtype=np.int32)
        filterInput.antisites = np.empty(self.ref.NAtoms, dtype=np.int32)
        filterInput.onAntisites = np.empty(self.ref.NAtoms, dtype=np.int32)
        filterInput.splitInterstitials = np.empty(3 * self.ref.NAtoms, dtype=np.int32)
        filterInput.defectFilterSelected = True
        
        return filterInput
    
    def test_pointDefectsSimple(self):
        """
        Point defects simple
        
        """
        # settings
        settings = pointDefectsFilter.PointDefectsFilterSettings()
        settings.updateSetting("vacancyRadius", 1.3)
        settings.updateSetting("showInterstitials", True)
        settings.updateSetting("showAntisites", True)
        settings.updateSetting("showVacancies", True)
        settings.updateSetting("findClusters", False)
        settings.updateSetting("neighbourRadius", 3.5)
        settings.updateSetting("minClusterSize", 3)
        settings.updateSetting("maxClusterSize", -1)
        settings.updateSetting("calculateVolumes", False)
        settings.updateSetting("calculateVolumesVoro", True)
        settings.updateSetting("calculateVolumesHull", False)
        settings.updateSetting("identifySplitInts", True)
        settings.updateSetting("useAcna", False)
        settings.updateSetting("acnaMaxBondDistance", 5.0)
        settings.updateSetting("acnaStructureType", 1)
        settings.updateSetting("filterSpecies", False)
        settings.updateSetting("visibleSpeciesList", [])
        
        # filter input
        filterInput = self.makeFilterInput()
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # check result
        self.assertEqual(len(filterInput.vacancies), 1)
        self.assertEqual(filterInput.vacancies[0], 0)
        self.assertEqual(len(filterInput.interstitials), 0)
        self.assertEqual(len(filterInput.antisites), 1)
        self.assertEqual(filterInput.antisites[0], 30)
        self.assertEqual(len(filterInput.onAntisites), 1)
        self.assertEqual(filterInput.onAntisites[0], 30)
        self.assertEqual(len(filterInput.splitInterstitials), 3)
        self.assertEqual(filterInput.splitInterstitials[0], 4)
        splits = filterInput.splitInterstitials
        self.assertTrue(splits[1] == 0 or splits[1] == 4)
        s2 = 0 if splits[1] == 4 else 0
        self.assertEqual(splits[2], s2)
        
        ### SPLIT INTS OFF ###
        
        # update settings
        settings.updateSetting("identifySplitInts", False)
        
        # filter input
        filterInput = self.makeFilterInput()
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # check result
        self.assertEqual(len(filterInput.vacancies), 2)
        vacs = filterInput.vacancies
        self.assertTrue(vacs[0] == 0 or vacs[0] == 4)
        v2 = 0 if vacs[0] == 4 else 4
        self.assertEqual(vacs[1], v2)
        self.assertEqual(len(filterInput.interstitials), 2)
        ints = filterInput.interstitials
        self.assertTrue(ints[0] == 0 or ints[0] == 4)
        i2 = 0 if ints[0] == 4 else 4
        self.assertEqual(ints[1], i2)
        self.assertEqual(len(filterInput.antisites), 1)
        self.assertEqual(filterInput.antisites[0], 30)
        self.assertEqual(len(filterInput.onAntisites), 1)
        self.assertEqual(filterInput.onAntisites[0], 30)
        self.assertEqual(len(filterInput.splitInterstitials), 0)
