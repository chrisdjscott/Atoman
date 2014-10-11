
"""
Additional routines to do with clusters (hulls, etc...)

@author: Chris Scott

"""
import logging

import numpy as np
import pyhull
from scipy import spatial


################################################################################
def findConvexHullFacets(N, pos):
    """
    Find convex hull of given points
    
    """
    # construct pts list
    pts = []
    for i in xrange(N):
        pts.append([pos[3*i], pos[3*i+1], pos[3*i+2]])
    
    # call scipy
    hull = spatial.ConvexHull(pts)
    facets = hull.simplices
    
    return facets


################################################################################
def findConvexHullVolume(N, pos, posIsPts=False):
    """
    Find convex hull of given points
    
    """
    # construct pts list
    if posIsPts:
        pts = pos
    
    else:
        #TODO: this should be written in C!
        pts = []
        for i in xrange(N):
            pts.append([pos[3*i], pos[3*i+1], pos[3*i+2]])
    
    # call pyhull library
    output = pyhull.qconvex("Qt FA", pts)
    
    # parse output
    volume = -999.0
    facetArea = -999.0
    cnt1 = 0
    cnt2 = 0
    for line in output:
        if "facet area" in line:
            array = line.split(":")
            facetArea = float(array[1])
            
            cnt1 += 1
        
        if "volume" in line:
            array = line.split(":")
            volume = float(array[1])
            
            cnt2 += 1
    
    assert cnt1 == 1, "ERROR: 'facet area' found %d times in pyhull output" % cnt1
    assert cnt2 == 1, "ERROR: 'volume' found %d times in pyhull output" % cnt2
    
    return volume, facetArea


################################################################################
def applyPBCsToCluster(clusterPos, cellDims, appliedPBCs):
    """
    Apply PBCs to cluster.
    
    """
    for i in xrange(7):
        if appliedPBCs[i]:
            # apply in x direction
            if i == 0:
                if min(clusterPos[0::3]) < 0.0:
                    clusterPos[0::3] += cellDims[0]
                
                else:
                    clusterPos[0::3] -= cellDims[0]
            
            elif i == 1:
                if min(clusterPos[1::3]) < 0.0:
                    clusterPos[1::3] += cellDims[1]
                
                else:
                    clusterPos[1::3] -= cellDims[1]
            
            elif i == 2:
                if min(clusterPos[2::3]) < 0.0:
                    clusterPos[2::3] += cellDims[2]
                
                else:
                    clusterPos[2::3] -= cellDims[2]
            
            elif i == 3:
                if min(clusterPos[0::3]) < 0.0:
                    clusterPos[0::3] += cellDims[0]
                
                else:
                    clusterPos[0::3] -= cellDims[0]
                
                if min(clusterPos[1::3]) < 0.0:
                    clusterPos[1::3] += cellDims[1]
                
                else:
                    clusterPos[1::3] -= cellDims[1]
            
            elif i == 4:
                if min(clusterPos[0::3]) < 0.0:
                    clusterPos[0::3] += cellDims[0]
                
                else:
                    clusterPos[0::3] -= cellDims[0]
                
                if min(clusterPos[2::3]) < 0.0:
                    clusterPos[2::3] += cellDims[2]
                
                else:
                    clusterPos[2::3] -= cellDims[2]
            
            elif i == 5:
                if min(clusterPos[1::3]) < 0.0:
                    clusterPos[1::3] += cellDims[1]
                
                else:
                    clusterPos[1::3] -= cellDims[1]
                
                if min(clusterPos[2::3]) < 0.0:
                    clusterPos[2::3] += cellDims[2]
                
                else:
                    clusterPos[2::3] -= cellDims[2]
            
            elif i == 6:
                if min(clusterPos[0::3]) < 0.0:
                    clusterPos[0::3] += cellDims[0]
                
                else:
                    clusterPos[0::3] -= cellDims[0]
                
                if min(clusterPos[1::3]) < 0.0:
                    clusterPos[1::3] += cellDims[1]
                
                else:
                    clusterPos[1::3] -= cellDims[1]
                
                if min(clusterPos[2::3]) < 0.0:
                    clusterPos[2::3] += cellDims[2]
                
                else:
                    clusterPos[2::3] -= cellDims[2]
            
            appliedPBCs[i] = 0
            
            break


################################################################################
def checkFacetsPBCs(facetsIn, clusterPos, excludeRadius, PBC, cellDims):
    """
    Remove facets that are far from cell
    
    """
    facets = []
    for facet in facetsIn:
        includeFlag = True
        for i in xrange(3):
            index = facet[i]
            
            for j in xrange(3):
                if PBC[j]:
                    if clusterPos[3*index+j] > cellDims[j] + excludeRadius or clusterPos[3*index+j] < 0.0 - excludeRadius:
                        includeFlag = False
                        break
            
            if not includeFlag:
                break
        
        if includeFlag:
            facets.append(facet)
    
    return facets

################################################################################

class AtomCluster(object):
    """
    Cluster of atoms
    
    """
    def __init__(self):
        self.indexes = []
        self.volume = None
        self.facetArea = None
    
    def __len__(self):
        return len(self.indexes)
    
    def __getitem__(self, i):
        return self.indexes[i]
    
    def __contains__(self, item):
        return item in self.indexes

################################################################################
class DefectCluster(object):
    """
    Defect cluster info.
    
    """
    def __init__(self):
        self.vacancies = []
        self.vacAsIndex = []
        self.interstitials = []
        self.antisites = []
        self.onAntisites = []
        self.splitInterstitials = []
        self.volume = None
        self.facetArea = None
    
    def belongsInCluster(self, defectType, defectIndex):
        """
        Check if the given defect belongs to this cluster
        
        defectType:
            1 = vacancy
            2 = interstitial
            3 = antisite
            4 = split interstitial
        
        """
        logger = logging.getLogger(__name__)
        logger.debug("Checking if defect belongs in cluster (%d, %d)", defectType, defectIndex)
        
        if defectType == 1 and defectIndex in self.vacancies:
            logger.debug("Defect is in cluster (vacancy)")
            return True
        
        elif defectType == 2 and defectIndex in self.interstitials:
            logger.debug("Defect is in cluster (interstitial)")
            return True
        
        elif defectType == 3 and defectIndex in self.antisites:
            logger.debug("Defect is in cluster (antisite)")
            return True
        
        elif defectType == 4 and defectIndex in self.splitInterstitials[::3]:
            logger.debug("Defect is in cluster (split interstitial)")
            return True
        
        return False
    
    def __len__(self):
        return self.getNDefects()
    
    def getNDefects(self):
        """
        Return total number of defects.
        
        """
        return len(self.vacancies) + len(self.interstitials) + len(self.antisites) + len(self.splitInterstitials) / 3
    
    def getNDefectsFull(self):
        """
        Return total number of defects, where a split interstitial counts as 3 defects.
        
        """
        return len(self.vacancies) + len(self.interstitials) + len(self.antisites) + len(self.splitInterstitials)
    
    def getNVacancies(self):
        """
        Return number of vacancies
        
        """
        return len(self.vacancies)
    
    def getNInterstitials(self):
        """
        Return number of interstitials
        
        """
        return len(self.interstitials)
    
    def getNAntisites(self):
        """
        Return number of antisites
        
        """
        return len(self.antisites)
    
    def getNSplitInterstitials(self):
        """
        Return number of split interstitials
        
        """
        return len(self.splitInterstitials) / 3
    
    def makeClusterPos(self, inputLattice, refLattice):
        """
        Make cluster pos array
        
        """
        clusterPos = np.empty(3 * self.getNDefectsFull(), np.float64)
        
        # vacancy positions
        count = 0
        for i in xrange(self.getNVacancies()):
            index = self.vacancies[i]
            
            clusterPos[3*count] = refLattice.pos[3*index]
            clusterPos[3*count+1] = refLattice.pos[3*index+1]
            clusterPos[3*count+2] = refLattice.pos[3*index+2]
            
            count += 1
        
        # antisite positions
        for i in xrange(self.getNAntisites()):
            index = self.antisites[i]
            
            clusterPos[3*count] = refLattice.pos[3*index]
            clusterPos[3*count+1] = refLattice.pos[3*index+1]
            clusterPos[3*count+2] = refLattice.pos[3*index+2]
            
            count += 1
        
        # interstitial positions
        for i in xrange(self.getNInterstitials()):
            index = self.interstitials[i]
            
            clusterPos[3*count] = inputLattice.pos[3*index]
            clusterPos[3*count+1] = inputLattice.pos[3*index+1]
            clusterPos[3*count+2] = inputLattice.pos[3*index+2]
            
            count += 1
        
        # split interstitial positions
        for i in xrange(self.getNSplitInterstitials()):
            index = self.splitInterstitials[3*i]
            clusterPos[3*count] = refLattice.pos[3*index]
            clusterPos[3*count+1] = refLattice.pos[3*index+1]
            clusterPos[3*count+2] = refLattice.pos[3*index+2]
            count += 1
            
            index = self.splitInterstitials[3*i+1]
            clusterPos[3*count] = refLattice.pos[3*index]
            clusterPos[3*count+1] = refLattice.pos[3*index+1]
            clusterPos[3*count+2] = refLattice.pos[3*index+2]
            count += 1
            
            index = self.splitInterstitials[3*i+2]
            clusterPos[3*count] = refLattice.pos[3*index]
            clusterPos[3*count+1] = refLattice.pos[3*index+1]
            clusterPos[3*count+2] = refLattice.pos[3*index+2]
            count += 1
        
        return clusterPos
