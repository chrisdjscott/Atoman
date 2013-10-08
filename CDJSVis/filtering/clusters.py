
"""
Additional routines to do with clusters (hulls, etc...)

@author: Chris Scott

"""
import pyhull
import pyhull.convex_hull
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
def findConvexHullVolume(N, pos):
    """
    Find convex hull of given points
    
    """
    # construct pts list
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
        includeFlag = 1
        for i in xrange(3):
            index = facet[i]
            
            for j in xrange(3):
                if PBC[j]:
                    if clusterPos[3*index+j] > cellDims[j] + excludeRadius or clusterPos[3*index+j] < 0.0 - excludeRadius:
                        includeFlag = 0
                        break
            
            if not includeFlag:
                break
        
        if includeFlag:
            facets.append(facet)
    
    return facets


################################################################################
class DefectCluster:
    """
    Defect cluster info.
    
    """
    def __init__(self):
        self.vacancies = []
        self.interstitials = []
        self.antisites = []
        self.onAntisites = []
    
    def getNDefects(self):
        """
        Return total number of defects.
        
        """
        return len(self.vacancies) + len(self.interstitials) + len(self.antisites)
    
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
