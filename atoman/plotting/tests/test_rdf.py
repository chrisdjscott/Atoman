
"""
Unit tests for the rdf module.

"""
import os
import unittest

import numpy as np

from .. import rdf
from ...lattice_gen import lattice_gen_pu3ga


################################################################################

class TestRDF(unittest.TestCase):
    """
    Test the RDF calculator.
    
    """
    def setUp(self):
        # load results
        dataDir = os.path.join(os.path.dirname(__file__), "data")
        self.allres = np.loadtxt(os.path.join(dataDir, "puga_rdf_all.csv"), skiprows=0, delimiter=",", unpack=True)
        
        # generate lattice
        args = lattice_gen_pu3ga.Args(NCells=[10,10,10])
        gen = lattice_gen_pu3ga.Pu3GaLatticeGenerator()
        status, self.lattice = gen.generateLattice(args)
        if status:
            raise unittest.SkipTest("Generate lattice failed (%d)" % status)
        
        # visible atoms
        self.visAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
    
    def tearDown(self):
        self.lattice = None
        self.allres = None
        self.visAtoms = None
    
    def test_calculateRDF(self):
        """
        Calculate RDF
        
        """
        # calculator
        calc = rdf.RDFCalculator()
        
        # test all (1 and 4 threads)
        xvals, yvals = calc.calculateRDF(self.visAtoms, self.lattice, 2.0, 15.0, 0.1, -1, -1, numThreads=1)
        self.assertTrue(np.allclose(xvals, self.allres[0]), msg="RDF xvals differ for ALL-ALL 1 proc")
        self.assertTrue(np.allclose(yvals, self.allres[1]), msg="RDF yvals differ for ALL-ALL 1 proc")
        
        xvals, yvals = calc.calculateRDF(self.visAtoms, self.lattice, 2.0, 15.0, 0.1, -1, -1, numThreads=4)
        self.assertTrue(np.allclose(xvals, self.allres[0]), msg="RDF xvals differ for ALL-ALL 4 proc")
        self.assertTrue(np.allclose(yvals, self.allres[1]), msg="RDF yvals differ for ALL-ALL 4 proc")
        
        # test out of range
        with self.assertRaises(RuntimeError):
            calc.calculateRDF(self.visAtoms, self.lattice, 2.0, 50.0, 0.1, -1, -1, numThreads=1)
