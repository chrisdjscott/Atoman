
"""
Unit tests for the cluster renderer

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import unittest

from .. import clusterRenderer
from ... import utils
from ....filtering import clusters
from ....filtering.filters import clusterFilter
from ....system import lattice


class TestClusterRenderer(unittest.TestCase):
    """
    Test the cluster renderer

    """
    def setUp(self):
        """
        Called before each test

        """
        # lattice
        self.lattice = lattice.Lattice()
        self.lattice.addAtom("He", [0, 0, 0], 0)
        self.lattice.addAtom("He", [0, 0, 8], 0)
        self.lattice.addAtom("He", [3, 0, 0], 0)
        self.lattice.addAtom("He", [0, 11, 22], 0)
        self.lattice.addAtom("He", [4, 7, 1], 0)
        self.lattice.addAtom("He", [1.5, 3, 0], 0)
        self.lattice.addAtom("He", [0, 4, 30], 0)
        self.lattice.addAtom("He", [1.5, 1.5, 3], 0)
        self.lattice.addAtom("He", [99, 99, 99], 0)
        
        # reference
        self.ref = lattice.Lattice()
        self.ref.addAtom("He", [0, 0, 0], 0)
        self.ref.addAtom("He", [0, 0, 8], 0)
        self.ref.addAtom("He", [3, 0, 0], 0)
        self.ref.addAtom("He", [0, 11, 22], 0)
        self.ref.addAtom("He", [4, 7, 1], 0)
        self.ref.addAtom("He", [1.5, 3, 0], 0)
        self.ref.addAtom("He", [0, 4, 30], 0)
        self.ref.addAtom("He", [1.5, 1.5, 3], 0)
        self.ref.addAtom("He", [99, 99, 99], 0)
        
        # cluster list
        cluster = clusters.AtomCluster(self.lattice)
        cluster.addAtom(0)
        cluster.addAtom(2)
        cluster.addAtom(5)
        cluster.addAtom(7)
        self.atomClusters = [cluster]
        
        cluster = clusters.DefectCluster(self.lattice, self.ref)
        cluster.addInterstitial(2)
        cluster.addInterstitial(5)
        cluster.addVacancy(0)
        cluster.addVacancy(7)
        self.defectClusters = [cluster]
        
        # settings
        self.settings = clusterFilter.ClusterFilterSettings()

    def tearDown(self):
        """
        Called after each test

        """
        # remove refs
        self.lattice = None
        self.ref = None
        self.atomClusters = None
        self.defectClusters = None
        self.settings = None

    def test_clusterRendererAtoms(self):
        """
        Cluster renderer (atoms)
        
        """
        # run the renderer
        renderer = clusterRenderer.ClusterRenderer()
        renderer.render(self.atomClusters, self.settings, refState=None)
        
        # check result is correct type
        self.assertIsInstance(renderer.getActor(), utils.ActorObject)
    
    def test_clusterRendererDefects(self):
        """
        Cluster renderer (defects)
        
        """
        # run the renderer
        renderer = clusterRenderer.ClusterRenderer()
        renderer.render(self.defectClusters, self.settings, refState=self.ref)
        
        # check result is correct type
        self.assertIsInstance(renderer.getActor(), utils.ActorObject)
