
"""
Unit tests for the Pu-Ga (Pu3Ga method) lattice generator

"""
import unittest

import numpy as np

from ..state.lattice import Lattice
from ..lattice_gen import lattice_gen_pu3ga


################################################################################

class TestLatticeGenPu3Ga(unittest.TestCase):
    """
    Test Pu-Ga (Pu3Ga method) lattice generator
    
    """
    def setUp(self):
        self.generator = lattice_gen_pu3ga.Pu3GaLatticeGenerator()
    
    def tearDown(self):
        self.generator = None
    
    def test_latticeGenPu3GaPBC(self):
        """
        Pu3Ga lattice generator (PBCs)
        
        """
        # 5 percent
        args = lattice_gen_pu3ga.Args(percGa=5, NCells=[2, 2, 2], a0=3.0, pbcx=True, pbcy=True, pbcz=True)
        status, lattice = self.generator.generateLattice(args)
        
        pos = np.asarray([ 0. ,  0. ,  0. ,  0. ,  1.5,  1.5,  1.5,  0. ,  1.5,  1.5,  1.5,
                          0. ,  0. ,  0. ,  3. ,  0. ,  1.5,  4.5,  1.5,  0. ,  4.5,  1.5,
                          1.5,  3. ,  0. ,  3. ,  0. ,  0. ,  4.5,  1.5,  1.5,  3. ,  1.5,
                          1.5,  4.5,  0. ,  0. ,  3. ,  3. ,  0. ,  4.5,  4.5,  1.5,  3. ,
                          4.5,  1.5,  4.5,  3. ,  3. ,  0. ,  0. ,  3. ,  1.5,  1.5,  4.5,
                          0. ,  1.5,  4.5,  1.5,  0. ,  3. ,  0. ,  3. ,  3. ,  1.5,  4.5,
                          4.5,  0. ,  4.5,  4.5,  1.5,  3. ,  3. ,  3. ,  0. ,  3. ,  4.5,
                          1.5,  4.5,  3. ,  1.5,  4.5,  4.5,  0. ,  3. ,  3. ,  3. ,  3. ,
                          4.5,  4.5,  4.5,  3. ,  4.5,  4.5,  4.5,  3. ])
        
        numatom = 32
        gacnt = int(args.percGa * 0.01 * float(numatom))
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, numatom)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 2)
        self.assertEqual(lattice.specieList[0], "Pu")
        self.assertEqual(lattice.specieList[1], "Ga")
        self.assertEqual(len(lattice.specieCount), 2)
        self.assertEqual(lattice.specieCount[0], numatom - gacnt)
        self.assertEqual(lattice.specieCount[1], gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 0]), numatom - gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 1]), gacnt)
        self.assertTrue(np.allclose(lattice.cellDims, [6,6,6]))
        
        # 2 percent
        args = lattice_gen_pu3ga.Args(percGa=2, NCells=[2, 2, 2], a0=3.0, pbcx=True, pbcy=True, pbcz=True)
        status, lattice = self.generator.generateLattice(args)
        
        numatom = 32
        gacnt = int(args.percGa * 0.01 * float(numatom))
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, numatom)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 2)
        self.assertEqual(lattice.specieList[0], "Pu")
        self.assertEqual(lattice.specieList[1], "Ga")
        self.assertEqual(len(lattice.specieCount), 2)
        self.assertEqual(lattice.specieCount[0], numatom - gacnt)
        self.assertEqual(lattice.specieCount[1], gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 0]), numatom - gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 1]), gacnt)
        self.assertTrue(np.allclose(lattice.cellDims, [6,6,6]))
    
    def test_latticeGenPu3GaNoPBC(self):
        """
        Pu3Ga lattice generator (no PBCs)
        
        """
        args = lattice_gen_pu3ga.Args(percGa=5, NCells=[1, 3, 2], a0=3.0, pbcx=False, pbcy=False, pbcz=False)
        status, lattice = self.generator.generateLattice(args)
        
        pos = np.asarray([ 0. ,  0. ,  0. ,  0. ,  1.5,  1.5,  1.5,  0. ,  1.5,  1.5,  1.5,
                          0. ,  0. ,  0. ,  3. ,  0. ,  1.5,  4.5,  1.5,  0. ,  4.5,  1.5,
                          1.5,  3. ,  0. ,  0. ,  6. ,  1.5,  1.5,  6. ,  0. ,  3. ,  0. ,
                          0. ,  4.5,  1.5,  1.5,  3. ,  1.5,  1.5,  4.5,  0. ,  0. ,  3. ,
                          3. ,  0. ,  4.5,  4.5,  1.5,  3. ,  4.5,  1.5,  4.5,  3. ,  0. ,
                          3. ,  6. ,  1.5,  4.5,  6. ,  0. ,  6. ,  0. ,  0. ,  7.5,  1.5,
                          1.5,  6. ,  1.5,  1.5,  7.5,  0. ,  0. ,  6. ,  3. ,  0. ,  7.5,
                          4.5,  1.5,  6. ,  4.5,  1.5,  7.5,  3. ,  0. ,  6. ,  6. ,  1.5,
                          7.5,  6. ,  0. ,  9. ,  0. ,  1.5,  9. ,  1.5,  0. ,  9. ,  3. ,
                          1.5,  9. ,  4.5,  0. ,  9. ,  6. ,  3. ,  0. ,  0. ,  3. ,  1.5,
                          1.5,  3. ,  0. ,  3. ,  3. ,  1.5,  4.5,  3. ,  0. ,  6. ,  3. ,
                          3. ,  0. ,  3. ,  4.5,  1.5,  3. ,  3. ,  3. ,  3. ,  4.5,  4.5,
                          3. ,  3. ,  6. ,  3. ,  6. ,  0. ,  3. ,  7.5,  1.5,  3. ,  6. ,
                          3. ,  3. ,  7.5,  4.5,  3. ,  6. ,  6. ,  3. ,  9. ,  0. ,  3. ,
                          9. ,  3. ,  3. ,  9. ,  6. ])

        numatom = 53
        gacnt = int(args.percGa * 0.01 * float(numatom))
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, numatom)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 2)
        self.assertEqual(lattice.specieList[0], "Pu")
        self.assertEqual(lattice.specieList[1], "Ga")
        self.assertEqual(len(lattice.specieCount), 2)
        self.assertEqual(lattice.specieCount[0], numatom - gacnt)
        self.assertEqual(lattice.specieCount[1], gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 0]), numatom - gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 1]), gacnt)
        self.assertTrue(np.allclose(lattice.cellDims, [3,9,6]))
        
        # 11 percent
        args = lattice_gen_pu3ga.Args(percGa=11, NCells=[1, 3, 2], a0=3.0, pbcx=False, pbcy=False, pbcz=False)
        status, lattice = self.generator.generateLattice(args)
        
        numatom = 53
        gacnt = int(args.percGa * 0.01 * float(numatom))
        self.assertEqual(status, 0)
        self.assertIsInstance(lattice, Lattice)
        self.assertEqual(lattice.NAtoms, numatom)
        self.assertTrue(np.allclose(lattice.pos, pos))
        self.assertEqual(len(lattice.specieList), 2)
        self.assertEqual(lattice.specieList[0], "Pu")
        self.assertEqual(lattice.specieList[1], "Ga")
        self.assertEqual(len(lattice.specieCount), 2)
        self.assertEqual(lattice.specieCount[0], numatom - gacnt)
        self.assertEqual(lattice.specieCount[1], gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 0]), numatom - gacnt)
        self.assertEqual(len([x for x in lattice.specie if x == 1]), gacnt)
        self.assertTrue(np.allclose(lattice.cellDims, [3,9,6]))
