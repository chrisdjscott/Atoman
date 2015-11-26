
"""
Unit tests for the Filterer

"""
import copy
import unittest

import numpy as np

from ..import filterer
from ...lattice_gen import lattice_gen_bcc
from ..filters import acnaFilter
from ..filters import bondOrderFilter
from ..filters import cropBoxFilter


################################################################################

def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing", path)

################################################################################

# required unless VoronoiOptionsWindow is rewritten to have a non GUI dependent settings object
class DummyVoroOpts(object):
    def __init__(self):
        self.dispersion = 10.0
        self.displayVoronoi = False
        self.useRadii = False
        self.opacity = 0.8
        self.outputToFile = False
        self.outputFilename = "voronoi.csv"
        self.faceAreaThreshold = 0.1

################################################################################

class TestFilterer(unittest.TestCase):
    """
    Test the Filterer

    """
    def setUp(self):
        """
        Called before each test

        """
        # generate lattice
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[10,10,10], a0=2.87, pbcx=True, pbcy=True, pbcz=True)
        gen = lattice_gen_bcc.BCCLatticeGenerator()
        status, self.inputState = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        self.inputState.PBC[:] = 1
        self.refState = copy.deepcopy(self.inputState)
        
        # voronoi options (required for filter)
        voroOpts = DummyVoroOpts()
        
        # the filterer
        self.filterer = filterer.Filterer(voroOpts)

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.inputState = None
        self.refState = None
        self.filterer = None

    def test_filterer(self):
        """
        Filterer
        
        TEST:
            - test running a couple of filters in a row
            - check outputs - vis atoms, defects, species counters
            - drift calculation
        
        """
        ## TEST 1 ##
        
        # filters
        filterNames = []
        filterSettings = []
        
        # add the acna filter
        acnaSettings = acnaFilter.AcnaFilterSettings()
        acnaSettings.updateSetting("maxBondDistance", 4.0)
        filterNames.append("ACNA")
        filterSettings.append(acnaSettings)
        
        # add the bond order filter
        bondOrderSettings = bondOrderFilter.BondOrderFilterSettings()
        bondOrderSettings.updateSetting("maxBondDistance", 4.0)
        filterNames.append("Bond order")
        filterSettings.append(bondOrderSettings)
        
        # apply the filterer (no filtering)
        self.filterer.runFilters(filterNames, filterSettings, self.inputState, self.refState)
        self.assertEqual(len(self.filterer.visibleAtoms), self.inputState.NAtoms)
        self.assertTrue("ACNA" in self.filterer.scalarsDict)
        self.assertTrue("Q4" in self.filterer.scalarsDict)
        self.assertTrue("Q6" in self.filterer.scalarsDict)
        for i in xrange(self.inputState.NAtoms):
            self.assertEqual(3, self.filterer.scalarsDict["ACNA"][i])
            self.assertAlmostEqual(0.036, self.filterer.scalarsDict["Q4"][i], places=3)
            self.assertAlmostEqual(0.511, self.filterer.scalarsDict["Q6"][i], places=3)
        self.assertTrue("ACNA structure count" in self.filterer.structureCounterDicts)
        self.assertTrue("BCC" in self.filterer.structureCounterDicts["ACNA structure count"])
        self.assertEqual(self.filterer.structureCounterDicts["ACNA structure count"]["BCC"], self.inputState.NAtoms)
        #TODO: test visible species list too
        
        ## TEST 2 ##
        
        # filters
        filterNames = []
        filterSettings = []
        
        # add the acna filter
        acnaSettings = acnaFilter.AcnaFilterSettings()
        acnaSettings.updateSetting("maxBondDistance", 4.0)
        filterNames.append("ACNA")
        filterSettings.append(acnaSettings)
        
        # add the bond order filter
        bondOrderSettings = bondOrderFilter.BondOrderFilterSettings()
        bondOrderSettings.updateSetting("maxBondDistance", 4.0)
        filterNames.append("Bond order")
        filterSettings.append(bondOrderSettings)
        
        # add crop box Filter
        cropBoxSettings = cropBoxFilter.CropBoxFilterSettings()
        cropBoxSettings.updateSetting("xEnabled", True)
        cropBoxSettings.updateSetting("xmin", 0.0)
        cropBoxSettings.updateSetting("xmax", 10.0)
        filterNames.append("Crop box")
        filterSettings.append(cropBoxSettings)
        
        # apply the filterer (no filtering)
        self.filterer.runFilters(filterNames, filterSettings, self.inputState, self.refState)
        nvis = 700
        self.assertEqual(len(self.filterer.visibleAtoms), nvis)
        self.assertTrue("ACNA" in self.filterer.scalarsDict)
        self.assertTrue("Q4" in self.filterer.scalarsDict)
        self.assertTrue("Q6" in self.filterer.scalarsDict)
        for i in xrange(nvis):
            self.assertEqual(3, self.filterer.scalarsDict["ACNA"][i])
            self.assertAlmostEqual(0.036, self.filterer.scalarsDict["Q4"][i], places=3)
            self.assertAlmostEqual(0.511, self.filterer.scalarsDict["Q6"][i], places=3)
            self.assertLessEqual(self.inputState.pos[3 * self.filterer.visibleAtoms[i]], 10.0)
        self.assertTrue("ACNA structure count" in self.filterer.structureCounterDicts)
        self.assertTrue("BCC" in self.filterer.structureCounterDicts["ACNA structure count"])
        # self.assertEqual(self.filterer.structureCounterDicts["ACNA structure count"]["BCC"], nvis)
        
        
