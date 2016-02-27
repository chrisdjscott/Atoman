
"""
Unit tests for the BCC lattice generator

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

import numpy as np

from ...system.lattice import Lattice
from .. import lattice_gen_bcc


################################################################################

class TestLatticeGenBCC(unittest.TestCase):
    """
    Test BCC lattice generator
    
    """
    def setUp(self):
        self.generator = lattice_gen_bcc.BCCLatticeGenerator()
    
    def tearDown(self):
        self.generator = None
    
    def test_latticeGenBCCPBC(self):
        """
        BCC lattice generator (PBCs)
        
        """
        args = lattice_gen_bcc.Args(sym="Fe", NCells=[2, 2, 2], a0=3.0, pbcx=True, pbcy=True, pbcz=True)
        status, lattice = self.generator.generateLattice(args)
        
        pos = np.asarray([ 0. ,  0. ,  0. ,  1.5,  1.5,  1.5,  0. ,  0. ,  3. ,  1.5,  1.5,
                          4.5,  0. ,  3. ,  0. ,  1.5,  4.5,  1.5,  0. ,  3. ,  3. ,  1.5,
                          4.5,  4.5,  3. ,  0. ,  0. ,  4.5,  1.5,  1.5,  3. ,  0. ,  3. ,
                          4.5,  1.5,  4.5,  3. ,  3. ,  0. ,  4.5,  4.5,  1.5,  3. ,  3. ,
                          3. ,  4.5,  4.5,  4.5], dtype=np.float64)
        
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, 16)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 1)
        self.assertEqual(lattice.specieList[0], "Fe")
        self.assertEqual(len(lattice.specieCount), 1)
        self.assertEqual(lattice.specieCount[0], 16)
        self.assertTrue(np.allclose(lattice.specie, np.zeros(16, np.int32)))
        self.assertTrue(np.allclose(lattice.cellDims, [6,6,6]))
    
    def test_latticeGenBCCNoPBC(self):
        """
        BCC lattice generator (no PBCs)
        
        """
        args = lattice_gen_bcc.Args(sym="Cr", NCells=[1, 3, 2], a0=3.0, pbcx=False, pbcy=False, pbcz=False)
        status, lattice = self.generator.generateLattice(args)
        
        pos = np.asarray([ 0. ,  0. ,  0. ,  1.5,  1.5,  1.5,  0. ,  0. ,  3. ,  1.5,  1.5,
                          4.5,  0. ,  0. ,  6. ,  0. ,  3. ,  0. ,  1.5,  4.5,  1.5,  0. ,
                          3. ,  3. ,  1.5,  4.5,  4.5,  0. ,  3. ,  6. ,  0. ,  6. ,  0. ,
                          1.5,  7.5,  1.5,  0. ,  6. ,  3. ,  1.5,  7.5,  4.5,  0. ,  6. ,
                          6. ,  0. ,  9. ,  0. ,  0. ,  9. ,  3. ,  0. ,  9. ,  6. ,  3. ,
                          0. ,  0. ,  3. ,  0. ,  3. ,  3. ,  0. ,  6. ,  3. ,  3. ,  0. ,
                          3. ,  3. ,  3. ,  3. ,  3. ,  6. ,  3. ,  6. ,  0. ,  3. ,  6. ,
                          3. ,  3. ,  6. ,  6. ,  3. ,  9. ,  0. ,  3. ,  9. ,  3. ,  3. ,
                          9. ,  6. ])
        
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, 30)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 1)
        self.assertEqual(lattice.specieList[0], "Cr")
        self.assertEqual(len(lattice.specieCount), 1)
        self.assertEqual(lattice.specieCount[0], 30)
        self.assertTrue(np.allclose(lattice.specie, np.zeros(30, np.int32)))
        self.assertTrue(np.allclose(lattice.cellDims, [3,9,6]))
