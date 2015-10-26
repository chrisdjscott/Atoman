
"""
Unit tests for the cluster filter

"""
import unittest

import numpy as np

from ....system import lattice
from .. import clusterFilter
from .. import base


################################################################################

class TestClusterFilter(unittest.TestCase):
    """
    Test cluster filter
    
    - check right number of clusters/right atoms
    - check min/max size option
    - check cluster list constructed properly
    - check resizing of full scalars/vectors array
    
    """
    def setUp(self):
        """
        Called before each test
        
        """
        # generate lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("He", [0,0,0], 0)
        self.lattice.addAtom("He", [2,0,0], 0)
        self.lattice.addAtom("He", [0,2,0], 0)
        self.lattice.addAtom("He", [0,0,2], 0)
        self.lattice.addAtom("He", [9,9,9], 0)
        self.lattice.addAtom("He", [2,2,0], 0)
        self.lattice.addAtom("He", [2,0,2], 0)
        self.lattice.addAtom("He", [0,2,2], 0)
        self.lattice.addAtom("He", [2,2,2], 0)
        
        # indexes of cluster atoms
        self.bigClusterIndexes = [0,1,2,3,5,6,7,8]
        self.smallClusterIndexes = [4]
        
        # filter
        self.filter = clusterFilter.ClusterFilter("Cluster")
    
    def tearDown(self):
        """
        Called after each test
        
        """
        # remove refs
        self.lattice = None
        self.filter = None
        self.bigClusterIndexes = None
        self.smallClusterIndexes = None
    
    def test_clusterFilter(self):
        """
        Cluster filter
        
        """
        # settings - all clusters visible
        settings = clusterFilter.ClusterFilterSettings()
        settings.updateSetting("neighbourRadius", 2.1)
        settings.updateSetting("minClusterSize", 1)
        settings.updateSetting("maxClusterSize", -1)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), len(self.bigClusterIndexes) + len(self.smallClusterIndexes))
        
        # check clusters are correct
        clusterList = result.getClusterList()
        self.assertEqual(len(clusterList), 2)
        for cluster in clusterList:
            if len(cluster) == 1:
                self.assertTrue(self.smallClusterIndexes[0] in cluster)
            
            else:
                for index in self.bigClusterIndexes:
                    self.assertTrue(index in cluster)
    
    def test_clusterFilterMinSize(self):
        """
        Cluster filter min size
        
        """
        # settings - all clusters visible
        settings = clusterFilter.ClusterFilterSettings()
        settings.updateSetting("neighbourRadius", 2.1)
        settings.updateSetting("minClusterSize", 2)
        settings.updateSetting("maxClusterSize", -1)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), len(self.bigClusterIndexes))
        
        # check clusters are correct
        clusterList = result.getClusterList()
        self.assertEqual(len(clusterList), 1)
        cluster = clusterList[0]
        self.assertEqual(len(cluster), len(self.bigClusterIndexes))
        for index in self.bigClusterIndexes:
            self.assertTrue(index in cluster)
    
    def test_clusterFilterMaxSize(self):
        """
        Cluster filter max size
        
        """
        # settings - all clusters visible
        settings = clusterFilter.ClusterFilterSettings()
        settings.updateSetting("neighbourRadius", 2.1)
        settings.updateSetting("minClusterSize", 1)
        settings.updateSetting("maxClusterSize", 2)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), len(self.smallClusterIndexes))
        
        # check clusters are correct
        clusterList = result.getClusterList()
        self.assertEqual(len(clusterList), 1)
        cluster = clusterList[0]
        self.assertEqual(len(cluster), len(self.smallClusterIndexes))
        for index in self.smallClusterIndexes:
            self.assertTrue(index in cluster)
    
    def test_clusterFilterNebRad(self):
        """
        Cluster filter neighbour radius
        
        """
        # settings - all clusters visible
        settings = clusterFilter.ClusterFilterSettings()
        settings.updateSetting("neighbourRadius", 1.9)
        settings.updateSetting("minClusterSize", 2)
        settings.updateSetting("maxClusterSize", -1)
        
        # set PBC
        self.lattice.PBC[:] = 1
        
        # filter input
        filterInput = base.FilterInput()
        filterInput.inputState = self.lattice
        visibleAtoms = np.arange(self.lattice.NAtoms, dtype=np.int32)
        filterInput.visibleAtoms = visibleAtoms
        filterInput.NScalars = 0
        filterInput.fullScalars = np.empty(0, np.float64)
        filterInput.NVectors = 0
        filterInput.fullVectors = np.empty(0, np.float64)
        
        # call filter
        result = self.filter.apply(filterInput, settings)
        self.assertIsInstance(result, base.FilterResult)
        
        # make sure num visible is correct
        self.assertEqual(len(visibleAtoms), 0)
