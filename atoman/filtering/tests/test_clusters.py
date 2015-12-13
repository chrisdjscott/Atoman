
"""
Test for the clusters module.

"""
import unittest

import numpy as np

from .. import clusters
from .. import voronoi
from ..filters import clusterFilter
from ..filters import pointDefectsFilter
from ...system import lattice


class DummyVoronoiOpts(object):
    def __init__(self):
        self.dispersion = 10.0
        self.displayVoronoi = False
        self.useRadii = False
        self.opacity = 0.8
        self.outputToFile = False
        self.outputFilename = "voronoi.csv"
        self.faceAreaThreshold = 0.1


class TestClusters(unittest.TestCase):
    """
    Tests for clusters
    
    """
    def setUp(self):
        """Set up the tests."""
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
        
        # defects
        self.inp = lattice.Lattice()
        self.inp.addAtom("He", [0, 0, 0], 0)
        self.inp.addAtom("He", [3, 0, 0], 0)
        self.inp.addAtom("He", [0, 3, 0], 0)
        self.inp.addAtom("He", [0, 0, 3], 0)
        self.inp.addAtom("He", [3, 3, 0], 0)
        self.inp.addAtom("He", [0, 3, 3], 0)
        self.inp.addAtom("He", [3, 0, 3], 0)
        self.inp.addAtom("He", [3, 3, 3], 0)
        self.ref = lattice.Lattice()
        self.ref.addAtom("He", [0, 0, 0], 0)
        self.ref.addAtom("He", [3, 0, 0], 0)
        self.ref.addAtom("He", [0, 3, 0], 0)
        self.ref.addAtom("He", [0, 0, 3], 0)
        self.ref.addAtom("He", [3, 3, 0], 0)
        self.ref.addAtom("He", [0, 3, 3], 0)
        self.ref.addAtom("He", [3, 0, 3], 0)
        self.ref.addAtom("He", [3, 3, 3], 0)
    
    def tearDown(self):
        """Clean up."""
        self.lattice = None
        self.ref = None
    
    def test_atomCluster(self):
        """
        AtomCluster
        
        """
        # create cluster and add atoms
        cluster = clusters.AtomCluster(self.lattice)
        cluster.addAtom(0)
        cluster.addAtom(2)
        cluster.addAtom(5)
        cluster.addAtom(7)
        
        # check
        self.assertEqual(len(cluster), 4)
        self.assertEqual(cluster[0], 0)
        self.assertEqual(cluster[1], 2)
        self.assertEqual(cluster[2], 5)
        self.assertEqual(cluster[3], 7)
        self.assertTrue(0 in cluster)
        self.assertTrue(2 in cluster)
        self.assertTrue(5 in cluster)
        self.assertTrue(7 in cluster)
        
        # make cluster pos
        pos = cluster.makeClusterPos()
        expectedPos = np.asarray([0, 0, 0, 3, 0, 0, 1.5, 3, 0, 1.5, 1.5, 3], dtype=np.float64)
        self.assertEqual(len(pos), len(expectedPos))
        for p1, p2 in zip(pos, expectedPos):
            self.assertEqual(p1, p2)
        
        # get volume/facet area
        cluster._facetArea = 44.3
        self.assertEqual(cluster.getFacetArea(), 44.3)
        cluster._facetArea = None
        cluster._volume = 11.6
        self.assertEqual(cluster.getVolume(), 11.6)
        cluster._volume = None
        
        # check calculate volume method(s)
        # voroOpts = DummyVoronoiOpts()
        # voroCalc = voronoi.VoronoiDefectsCalculator(voroOpts)
        # settings = pointDefectsFilter.PointDefectsFilterSettings()
        # settings.updateSetting("calculateVolumes", True)
        # settings.updateSetting("calculateVolumesHull", True)
        # settings.updateSetting("calculateVolumesVoro", False)
        # cluster.calculateVolume(voroCalc, settings)
        # self.assertEqual(cluster.getVolume(), 27.0)
        # settings.updateSetting("calculateVolumesHull", False)
        # settings.updateSetting("calculateVolumesVoro", True)
        # cluster.calculateVolume(voroCalc, settings)
        # self.assertEqual(cluster.getVolume(), 54.0)
    
    def test_defectCluster(self):
        """
        DefectCluster
        
        """
        # create cluster and add defects
        cluster = clusters.DefectCluster(self.inp, self.ref)
        cluster.addInterstitial(0)
        cluster.addSplitInterstitial(2, 4, 7)
        cluster.addVacancy(1)
        cluster.addVacancy(6)
        cluster.addAntisite(5, 3)
        cluster.addAntisite(3, 5)
        
        # check
        self.assertEqual(cluster.getNDefects(), 6)
        self.assertEqual(cluster.getNDefectsFull(), 8)
        self.assertEqual(cluster.getNVacancies(), 2)
        self.assertEqual(cluster.getNInterstitials(), 1)
        self.assertEqual(cluster.getNAntisites(), 2)
        self.assertEqual(cluster.getNSplitInterstitials(), 1)
        self.assertEqual(cluster.getVacancy(0), 1)
        self.assertEqual(cluster.getVacancy(1), 6)
        self.assertEqual(cluster.getInterstitial(0), 0)
        self.assertEqual(cluster.getAntisite(0)[0], 5)
        self.assertEqual(cluster.getAntisite(0)[1], 3)
        self.assertEqual(cluster.getAntisite(1)[0], 3)
        self.assertEqual(cluster.getAntisite(1)[1], 5)
        self.assertEqual(cluster.getSplitInterstitial(0)[0], 2)
        self.assertEqual(cluster.getSplitInterstitial(0)[1], 4)
        self.assertEqual(cluster.getSplitInterstitial(0)[2], 7)
        
        # make cluster pos
        pos = cluster.makeClusterPos()
        expectedPos = np.asarray([3, 0, 0, 3, 0, 3, 0, 3, 3, 0, 0, 3,
                                  0, 0, 0, 0, 3, 0, 3, 3, 0, 3, 3, 3], dtype=np.float64)
        self.assertEqual(len(pos), len(expectedPos))
        for p1, p2 in zip(pos, expectedPos):
            self.assertEqual(p1, p2)
        
        # check calculate volumes successful
    
    def test_convexHullFacetArea(self):
        """
        Convex hull facet area
           
        """
        try:
            import pyhull
        except ImportError:
            self.skipTest("Pyhull (optional) is not available so cannot compute facet area.")
        
        # make points
        N = 8
        pts = [0, 0, 0,
               3, 0, 0,
               0, 3, 0,
               0, 0, 3,
               3, 3, 0,
               0, 3, 3,
               3, 0, 3,
               3, 3, 3]
        
        # calc volume
        volume, facetArea = clusters.findConvexHullVolume(N, pts)
        
        self.assertAlmostEqual(facetArea, 54.0)
    
    def test_convexHullVolume(self):
        """
        Convex hull volume
           
        """
        # make points
        N = 8
        pts = [0, 0, 0,
               3, 0, 0,
               0, 3, 0,
               0, 0, 3,
               3, 3, 0,
               0, 3, 3,
               3, 0, 3,
               3, 3, 3]
        
        # calc volume
        volume, facetArea = clusters.findConvexHullVolume(N, pts)
        
        self.assertAlmostEqual(volume, 27.0)
