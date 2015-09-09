
"""
Point defects
=============

Identify point defects in the lattice by comparison to a reference lattice...

"""
import numpy as np

from . import base
from . import _defects
from . import acnaFilter
from .. import clusters
from ...algebra import vectors


class PointDefectsFilterSettings(base.BaseSettings):
    """
    Settings for the point defects filter
    
    """
    def __init__(self):
        super(PointDefectsFilterSettings, self).__init__()
        
        self.registerSetting("vacancyRadius", default=1.3)
        self.registerSetting("showInterstitials", default=True)
        self.registerSetting("showAntisites", default=True)
        self.registerSetting("showVacancies", default=True)
        self.registerSetting("findClusters", default=False)
        self.registerSetting("neighbourRadius", default=3.5)
        self.registerSetting("minClusterSize", default=3)
        self.registerSetting("maxClusterSize", default=-1)
        self.registerSetting("hullCol", default=[0,0,1])
        self.registerSetting("hullOpacity", default=0.5)
        self.registerSetting("calculateVolumes", default=False)
        self.registerSetting("calculateVolumesVoro", default=True)
        self.registerSetting("calculateVolumesHull", default=False)
        self.registerSetting("drawConvexHulls", default=False)
        self.registerSetting("hideDefects", default=False)
        self.registerSetting("identifySplitInts", default=True)
        self.registerSetting("vacScaleSize", default=0.75)
        self.registerSetting("vacOpacity", default=0.8)
        self.registerSetting("vacSpecular", default=0.4)
        self.registerSetting("vacSpecularPower", default=10)
        self.registerSetting("useAcna", default=False)
        self.registerSetting("acnaMaxBondDistance", default=5.0)
        self.registerSetting("acnaStructureType", default=1)
        self.registerSetting("filterSpecies", default=False)
        self.registerSetting("visibleSpeciesList", default=[])
        self.registerSetting("drawDisplacementVectors", default=False)
        self.registerSetting("bondThicknessVTK", default=0.4)
        self.registerSetting("bondThicknessPOV", default=0.4)
        self.registerSetting("bondNumSides", default=5)


class PointDefectsFilter(base.BaseFilter):
    """
    Point defects filter.
    
    """
    def apply(self, filterInput, settings):
        """Apply the filter."""
        # unpack inputs
        inputLattice = filterInput.inputState
        refLattice = filterInput.refState
        ompNumThreads = filterInput.ompNumThreads
        vacancies = filterInput.vacancies
        interstitials = filterInput.interstitials
        splitInterstitials = filterInput.splitInterstitials
        antisites = filterInput.antisites
        onAntisites = filterInput.onAntisites
        driftCompensation = filterInput.driftCompensation
        driftVector = filterInput.driftVector
        
        # compute ACNA if required
        acnaArray = None
        if settings.getSetting("useAcna"):
            self.logger.debug("Computing ACNA from point defects filter...")
            
            # acna settings
            acnaSettings = acnaFilter.AcnaFilterSettings()
            acnaSettings.updateSetting("maxBondDistance", settings.getSetting("acnaMaxBondDistance"))
            
            # acna input
            acnaInput = base.FilterInput()
            acnaInput.inputState = inputLattice
            acnaInput.NScalars = 0
            acnaInput.fullScalars = np.empty(acnaInput.NScalars, np.float64)
            acnaInput.NVectors = 0
            acnaInput.fullVectors = np.empty(acnaInput.NVectors, np.float64)
            acnaInput.ompNumThreads = ompNumThreads
            acnaInput.visibleAtoms = np.arange(inputLattice.NAtoms, dtype=np.int32)
            
            # acna filter
            acna = acnaFilter.AcnaFilter("ACNA - Defects")
            
            # run filter
            acnaResult = acna.apply(acnaInput, acnaSettings)
            
            # get scalars array from result
            acnaArray = acnaResult.getScalars()["ACNA"]
            
            # structure counters
            sd = acnaResult.getStructureCounterDict()
            self.logger.debug("  %r", sd)
        
        # check ACNA array is valid
        if acnaArray is None or len(acnaArray) != inputLattice.NAtoms:
            acnaArray = np.empty(0, np.float64)
        
        # set up excluded specie arrays
        visibleSpecieList = settings.getSetting("visibleSpeciesList")
        self.logger.debug("Visible species list: %r", visibleSpecieList)
        
        exclSpecsInput = []
        for i, spec in enumerate(inputLattice.specieList):
            if spec not in visibleSpecieList:
                exclSpecsInput.append(i)
        exclSpecsInput = np.asarray(exclSpecsInput, dtype=np.int32)
        
        exclSpecsRef = []
        for i, spec in enumerate(refLattice.specieList):
            if spec not in visibleSpecieList:
                exclSpecsRef.append(i)
        exclSpecsRef = np.asarray(exclSpecsRef, dtype=np.int32)
        
        # specie counter arrays
        vacSpecCount = np.zeros(len(refLattice.specieList), np.int32)
        intSpecCount = np.zeros(len(inputLattice.specieList), np.int32)
        antSpecCount = np.zeros(len(refLattice.specieList), np.int32)
        onAntSpecCount = np.zeros((len(refLattice.specieList), len(inputLattice.specieList)), np.int32)
        splitIntSpecCount = np.zeros((len(inputLattice.specieList), len(inputLattice.specieList)), np.int32)
        
        NDefectsByType = np.zeros(6, np.int32)
        
        if settings.getSetting("findClusters"):
            defectCluster = np.empty(inputLattice.NAtoms + refLattice.NAtoms, np.int32)
        else:
            defectCluster = np.empty(0, np.int32)
        
        # settings for C call
        showVacancies = settings.getSetting("showVacancies")
        showInterstitials = settings.getSetting("showInterstitials")
        showAntisites = settings.getSetting("showAntisites")
        vacancyRadius = settings.getSetting("vacancyRadius")
        findClusters = settings.getSetting("findClusters")
        neighbourRadius = settings.getSetting("neighbourRadius")
        minClusterSize = settings.getSetting("minClusterSize")
        maxClusterSize = settings.getSetting("maxClusterSize")
        identifySplitInts = settings.getSetting("identifySplitInts")
        acnaStructureType = settings.getSetting("acnaStructureType")
        
        # call C library
        self.logger.debug("Calling C library")
        _defects.findDefects(showVacancies, showInterstitials, showAntisites, NDefectsByType, vacancies, interstitials,
                             antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                             inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                             refLattice.pos, refLattice.cellDims, inputLattice.PBC, vacancyRadius, findClusters, neighbourRadius,
                             defectCluster, vacSpecCount, intSpecCount, antSpecCount, onAntSpecCount, splitIntSpecCount,
                             minClusterSize, maxClusterSize, splitInterstitials, identifySplitInts, driftCompensation,
                             driftVector, acnaArray, acnaStructureType)
        
        # summarise
        NDef = NDefectsByType[0]
        NVac = NDefectsByType[1]
        NInt = NDefectsByType[2]
        NAnt = NDefectsByType[3]
        NSplit = NDefectsByType[5]
        vacancies.resize(NVac, refcheck=False)
        interstitials.resize(NInt, refcheck=False)
        antisites.resize(NAnt, refcheck=False)
        onAntisites.resize(NAnt, refcheck=False)
        splitInterstitials.resize(NSplit * 3, refcheck=False)
        
        # report counters
        self.logger.info("Found %d defects", NDef)
        
        if settings.getSetting("showVacancies"):
            self.logger.info("  %d vacancies", NVac)
            for i in xrange(len(refLattice.specieList)):
                self.logger.info("    %d %s vacancies", vacSpecCount[i], refLattice.specieList[i])
        
        if settings.getSetting("showInterstitials"):
            self.logger.info("  %d interstitials", NInt + NSplit)
            for i in xrange(len(inputLattice.specieList)):
                self.logger.info("    %d %s interstitials", intSpecCount[i], inputLattice.specieList[i])
        
            if settings.getSetting("identifySplitInts"):
                self.logger.info("    %d split interstitials", NSplit)
                for i in xrange(len(inputLattice.specieList)):
                    for j in xrange(i, len(inputLattice.specieList)):
                        if j == i:
                            N = splitIntSpecCount[i][j]
                        else:
                            N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                        self.logger.info("      %d %s - %s split interstitials", N, inputLattice.specieList[i], inputLattice.specieList[j])
        
        if settings.getSetting("showAntisites"):
            self.logger.info("  %d antisites", NAnt)
            for i in xrange(len(refLattice.specieList)):
                for j in xrange(len(inputLattice.specieList)):
                    if inputLattice.specieList[j] == refLattice.specieList[i]:
                        continue
                    
                    self.logger.info("    %d %s on %s antisites", onAntSpecCount[i][j], inputLattice.specieList[j], refLattice.specieList[i])
        
        if settings.getSetting("identifySplitInts"):
            self.logger.info("Splint interstitial analysis")
            
            PBC = inputLattice.PBC
            cellDims = inputLattice.cellDims
            
            for i in xrange(NSplit):
                ind1 = splitInterstitials[3*i+1]
                ind2 = splitInterstitials[3*i+2]
                
                pos1 = inputLattice.pos[3*ind1:3*ind1+3]
                pos2 = inputLattice.pos[3*ind2:3*ind2+3]
                
                sepVec = vectors.separationVector(pos1, pos2, cellDims, PBC)
                norm = vectors.normalise(sepVec)
                
                self.logger.info("  Orientation of split int %d: (%.3f %.3f %.3f)", i, norm[0], norm[1], norm[2])
        
        # sort clusters here
        clusterList = []
        if settings.getSetting("findClusters"):
            NClusters = NDefectsByType[4]
            
            defectCluster.resize(NDef)
            
            # build cluster lists
            for i in xrange(NClusters):
                clusterList.append(clusters.DefectCluster())
            
            # add atoms to cluster lists
            clusterIndexMapper = {}
            count = 0
            for i in xrange(NVac):
                atomIndex = vacancies[i]
                clusterIndex = defectCluster[i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].vacancies.append(atomIndex)
                clusterList[clusterListIndex].vacAsIndex.append(i)
            
            for i in xrange(NInt):
                atomIndex = interstitials[i]
                clusterIndex = defectCluster[NVac + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].interstitials.append(atomIndex)
            
            for i in xrange(NAnt):
                atomIndex = antisites[i]
                atomIndex2 = onAntisites[i]
                clusterIndex = defectCluster[NVac + NInt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                clusterList[clusterListIndex].antisites.append(atomIndex)
                clusterList[clusterListIndex].onAntisites.append(atomIndex2)
            
            for i in xrange(NSplit):
                clusterIndex = defectCluster[NVac + NInt + NAnt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                
                atomIndex = splitInterstitials[3*i]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
                
                atomIndex = splitInterstitials[3*i+1]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
                
                atomIndex = splitInterstitials[3*i+2]
                clusterList[clusterListIndex].splitInterstitials.append(atomIndex)
        
        # make result
        result = base.FilterResult()
        result.setClusterList(clusterList)
        
        return result
