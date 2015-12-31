
"""
Unit tests for latticeReaders module

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import unittest
import tempfile
import shutil

from .. import latticeReaders
from ..lattice import Lattice


################################################################################

def log_output(*args, **kwargs):
    print(args[0])

def log_warning(*args, **kwargs):
    print("DISPLAY WARNING: %s" % args[0])

def log_error(*args, **kwargs):
    print("DISPLAY ERROR: %s" % args[0])

################################################################################

def path_to_file(path):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..", "testing", path)

################################################################################

def specie_index(specieList, symIn):
    index = None
    for count, sym in enumerate(specieList):
        if sym == symIn:
            index = count
            break
    
    return index

################################################################################

class TestLbomdDatReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # create reader
        self.reader = latticeReaders.LbomdDatReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
    
    def test_readGzipLattice(self):
        """
        Read dat gzipped
        
        """
        filename = path_to_file("postcascref.dat")
        
        status, state = self.reader.readFile(filename)
        
        self.assertIsInstance(state, Lattice)
        self.assertEqual(status, 0)
        self.assertEqual(state.cellDims[0], 111.36)
        self.assertEqual(state.cellDims[1], 111.36)
        self.assertEqual(state.cellDims[2], 111.36)
        self.assertEqual(len(state.specieList), 2)
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Pu")], 52532)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Ga")], 2764)
    
    def test_readLattice(self):
        """
        Read dat
        
        """
        filename = path_to_file("kenny_lattice.dat")
        
        status, state = self.reader.readFile(filename)
        self.assertEqual(status, 0)
        self.assertEqual(state.NAtoms, 1140)
        self.assertEqual(state.cellDims[0], 26.3781222148)
        self.assertEqual(state.cellDims[1], 26.3781222148)
        self.assertEqual(state.cellDims[2], 26.3781222148)
        self.assertEqual(len(state.specieList), 3)
        self.assertIn("Si", state.specieList)
        self.assertIn("B_", state.specieList)
        self.assertIn("O_", state.specieList)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Si")], 280)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "B_")], 120)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "O_")], 740)
    
################################################################################

class TestLbomdRefReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # create reader
        self.reader = latticeReaders.LbomdRefReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
    
    def test_readLattice(self):
        """
        Read ref
        
        """
        filename = path_to_file("anim-ref-Hdiff.xyz")
        
        status, state = self.reader.readFile(filename)
        self.assertEqual(status, 0)
        self.assertEqual(state.NAtoms, 16392)
        self.assertEqual(state.cellDims[0], 74.24)
        self.assertEqual(state.cellDims[1], 74.24)
        self.assertEqual(state.cellDims[2], 74.24)
        self.assertEqual(len(state.specieList), 3)
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertIn("H_", state.specieList)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Pu")], 15565)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Ga")], 819)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "H_")], 8)

################################################################################

def skipIfStatus(status):
    if not status:
        return unittest.skip("Skipping XYZ test because readRef failed")

################################################################################

class TestLbomdXYZReader(unittest.TestCase):
    """
    Test LBOMD DAT reader
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # tmp dir
        self.tmpLocation = tempfile.mkdtemp(prefix="atomanTest")
        
        # create reader
        refReader = latticeReaders.LbomdRefReader(self.tmpLocation, log_output, log_warning, log_error)
        refStatus, self.refState = refReader.readFile(path_to_file("anim-ref-Hdiff.xyz"))
        
        if refStatus:
            raise unittest.SkipTest("read ref failed ('%s'; %d; %r)" % (path_to_file("anim-ref-Hdiff.xyz"), refStatus, self.refState))
        
        self.reader = latticeReaders.LbomdXYZReader(self.tmpLocation, log_output, log_warning, log_error)
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove tmp dir
        shutil.rmtree(self.tmpLocation)
        
        # remove ref to reader
        self.reader = None
        self.refState = None
    
    def test_readGzipLattice(self):
        """
        Read xyz
        
        """
        filename = path_to_file("input-HDiff.xyz")
        
        status, state = self.reader.readFile(filename, self.refState)
        
        self.assertIsInstance(state, Lattice)
        self.assertEqual(status, 0)
        self.assertEqual(state.NAtoms, 16392)
        self.assertEqual(state.cellDims[0], 74.24)
        self.assertEqual(state.cellDims[1], 74.24)
        self.assertEqual(state.cellDims[2], 74.24)
        self.assertEqual(len(state.specieList), 3)
        self.assertIn("Pu", state.specieList)
        self.assertIn("Ga", state.specieList)
        self.assertIn("H_", state.specieList)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Pu")], 15565)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "Ga")], 819)
        self.assertEqual(state.specieCount[specie_index(state.specieList, "H_")], 8)
