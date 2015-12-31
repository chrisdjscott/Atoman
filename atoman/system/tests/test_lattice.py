
"""
Unit tests for Lattice object

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import unittest
import tempfile
import shutil

import numpy as np
from six.moves import range

from ..latticeReaders import LbomdDatReader, basic_displayError, basic_displayWarning, basic_log

   
def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing", path)
   
   
class TestLattice(unittest.TestCase):
    """
    Test Lattice
        
    """
    def setUp(self):
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
         
        # create reader
        reader = LbomdDatReader(self.tmpLocation, basic_log, basic_displayWarning, basic_displayError)
         
        status, self.lattice = reader.readFile(path_to_file("lattice.dat"))
        if status:
            self.fail("Error reading in lattice.dat")
     
    def tearDown(self):
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
         
        self.lattice = None
    
    def test_wrapAtoms(self):
        """
        Lattice wrapAtoms
        
        """
        # modify position
        self.lattice.pos[2] += self.lattice.cellDims[2]
        self.lattice.pos[3] -= self.lattice.cellDims[0]
        self.lattice.pos[4] += self.lattice.cellDims[1]
        
        # wrap
        self.lattice.wrapAtoms()
        
        # check result
        for i in range(self.lattice.NAtoms):
            i3 = 3 * i
            for j in range(3):
                pos = self.lattice.pos[i3 + j]
                self.assertTrue(pos <= self.lattice.cellDims[0] and self.lattice.pos[3 * i + j] >= 0.0)
    
    def test_atomSeparation(self):
        """
        Lattice atomSeparation
        
        """
        sep = self.lattice.atomSeparation(0, self.lattice.NAtoms - 1, np.ones(3, np.int32))
        self.assertAlmostEqual(sep, 4.9945095855)
        with self.assertRaises(IndexError):
            self.lattice.atomSeparation(0, self.lattice.NAtoms, np.ones(3, np.int32))
        with self.assertRaises(IndexError):
            self.lattice.atomSeparation(self.lattice.NAtoms, 0, np.ones(3, np.int32))
    
    def test_density(self):
        """
        Lattice density
        
        """
        dens = self.lattice.density()
        self.assertAlmostEqual(dens, 0.0589818414)
    
    def test_volume(self):
        """
        Lattice volume
            
        """
        lattice = self.lattice
        
        v1 = lattice.volume()
        v2 = lattice.cellDims[0] * lattice.cellDims[1] * lattice.cellDims[2]
        
        self.assertEqual(v1, v2)
    
    def test_atomPos(self):
        """
        Lattice atomPos
        
        """
        pos = self.lattice.atomPos(11)
        self.assertEqual(pos[0], 2.039)
        self.assertEqual(pos[1], 2.039)
        self.assertEqual(pos[2], 8.156)
    
    def test_writeLattice(self):
        """
        Lattice writeLattice
        
        """
        with tempfile.NamedTemporaryFile() as tmpf:
            fn = tmpf.name
            self.lattice.writeLattice(fn)
            line = tmpf.readline()
            natom = int(line)
            self.assertEqual(natom, self.lattice.NAtoms)
            line = tmpf.readline()
            dims = line.split()
            self.assertEqual(float(dims[0]), self.lattice.cellDims[0])
            self.assertEqual(float(dims[1]), self.lattice.cellDims[1])
            self.assertEqual(float(dims[2]), self.lattice.cellDims[2])
            for i, line in enumerate(tmpf):
                array = line.split()
                self.assertEqual(array[0].decode('utf-8'), "Au")
                for j in range(3):
                    self.assertEqual(float(array[j + 1]), self.lattice.pos[3 * i + j])
                self.assertEqual(float(array[4]), self.lattice.charge[i])
            self.assertEqual(i + 1, self.lattice.NAtoms)
        
        with tempfile.NamedTemporaryFile() as tmpf:
            visatoms = np.asarray([1, 3, 5], dtype=np.int32)
            fn = tmpf.name
            self.lattice.writeLattice(fn, visibleAtoms=visatoms)
            line = tmpf.readline()
            natom = int(line)
            self.assertEqual(natom, len(visatoms))
            line = tmpf.readline()
            dims = line.split()
            self.assertEqual(float(dims[0]), self.lattice.cellDims[0])
            self.assertEqual(float(dims[1]), self.lattice.cellDims[1])
            self.assertEqual(float(dims[2]), self.lattice.cellDims[2])
            for i, line in enumerate(tmpf):
                index = visatoms[i]
                array = line.split()
                self.assertEqual(array[0].decode('utf-8'), "Au")
                for j in range(3):
                    self.assertEqual(float(array[j + 1]), self.lattice.pos[3 * index + j])
                self.assertEqual(float(array[4]), self.lattice.charge[index])
            self.assertEqual(i + 1, len(visatoms))
