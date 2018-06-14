
"""
Point defects
=============

Identify point defects in the lattice by comparison to a reference lattice.

This filter will identify vacancies, interstitials (including split interstitials) and antisites. A vacancy radius is
used to determine if an atom in the input lattice is associated with a site in the reference lattice. If an input atom
is not associated with a reference site then it is an interstitial. If a reference site has no input atom associated
with it then it is a vacancy. If an input atom is sitting on a reference site of a different species then it is an
anitiste.

Split interstitials are identified by looking for vacancies that have two neighbouring interstitials, forming the split
interstitial.

Defects can be refined using ACNA, which checks if interstitials have the ideal structure type for the system. If they
do, and if the interstitial is very close to a vacancy, then we remove the pair from the list of defects. We have found
this to be important in alloys where the crystal structure is distorted due to the alloying element. In these cases the
use of a fixed vacancy radius does not always work well by itself.

.. glossary::

    Vacancy radius
        This parameter is used to determine if an input atom is associated with a reference site.
    
    Show interstitials
        If checked include interstitials in the list of defects.
    
    Render spaghetti
        Render spaghetti as described in [1]_.

.. [1] A. F. Calder et al. *Philos. Mag.* **90** (2010) 863-884;
       `doi: 10.1080/14786430903117141 <http://dx.doi.org/10.1080/14786430903117141>`_.

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import numpy as np

from . import base
from . import _defects
from . import acnaFilter
from . import displacementFilter
from .. import clusters
from ...algebra import vectors
from six.moves import range


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
        self.registerSetting("hullCol", default=[0, 0, 1])
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
        self.registerSetting("drawSpaghetti", default=False)
        self.registerSetting("writeDefectsFile", default=False)
        self.registerSetting("defectsFile", default="defects.dat")
        
        # old methods for calculating certain things
        self.registerSetting("splitIntsOld", default=False)
        self.registerSetting("acnaOld", default=False)


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
        filterSpecies = settings.getSetting("filterSpecies")
        splitOld = settings.getSetting("splitIntsOld")
        acnaOld = settings.getSetting("acnaOld")
        
        # call C library
        self.logger.debug("Calling C library")
        _defects.findDefects(showVacancies, showInterstitials, showAntisites, NDefectsByType, vacancies, interstitials,
                             antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms,
                             inputLattice.specieList, inputLattice.specie, inputLattice.pos, refLattice.NAtoms,
                             refLattice.specieList, refLattice.specie, refLattice.pos, refLattice.cellDims,
                             inputLattice.PBC, vacancyRadius, findClusters, neighbourRadius, defectCluster,
                             vacSpecCount, intSpecCount, antSpecCount, onAntSpecCount, splitIntSpecCount,
                             minClusterSize, maxClusterSize, splitInterstitials, identifySplitInts, driftCompensation,
                             driftVector, acnaArray, acnaStructureType, int(filterSpecies), int(splitOld), int(acnaOld))
        
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
            for i in range(len(refLattice.specieList)):
                self.logger.info("    %d %s vacancies", vacSpecCount[i], refLattice.specieList[i])
        
        if settings.getSetting("showInterstitials"):
            self.logger.info("  %d interstitials", NInt + NSplit)
            for i in range(len(inputLattice.specieList)):
                self.logger.info("    %d %s interstitials", intSpecCount[i], inputLattice.specieList[i])
        
            if settings.getSetting("identifySplitInts"):
                self.logger.info("    %d split interstitials", NSplit)
                for i in range(len(inputLattice.specieList)):
                    for j in range(i, len(inputLattice.specieList)):
                        if j == i:
                            N = splitIntSpecCount[i][j]
                        else:
                            N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                        self.logger.info("      %d %s - %s split interstitials", N, inputLattice.specieList[i],
                                         inputLattice.specieList[j])
        
        if settings.getSetting("showAntisites"):
            self.logger.info("  %d antisites", NAnt)
            for i in range(len(refLattice.specieList)):
                for j in range(len(inputLattice.specieList)):
                    if inputLattice.specieList[j] == refLattice.specieList[i]:
                        continue
                    
                    self.logger.info("    %d %s on %s antisites", onAntSpecCount[i][j], inputLattice.specieList[j],
                                     refLattice.specieList[i])
        
        if settings.getSetting("showInterstitials") and settings.getSetting("identifySplitInts"):
            self.logger.info("Split interstitial analysis")
            
            PBC = inputLattice.PBC
            cellDims = inputLattice.cellDims
            
            for i in range(NSplit):
                ind1 = splitInterstitials[3 * i + 1]
                ind2 = splitInterstitials[3 * i + 2]
                
                pos1 = inputLattice.pos[3 * ind1:3 * ind1 + 3]
                pos2 = inputLattice.pos[3 * ind2:3 * ind2 + 3]
                
                sepVec = vectors.separationVector(pos1, pos2, cellDims, PBC)
                norm = vectors.normalise(sepVec)
                
                self.logger.info("  Orientation of split int %d: (%.3f %.3f %.3f)", i, norm[0], norm[1], norm[2])
        
        # sort clusters here
        clusterList = []
        if settings.getSetting("findClusters"):
            NClusters = NDefectsByType[4]
            
            defectCluster.resize(NDef)
            
            # build cluster lists
            for i in range(NClusters):
                clusterList.append(clusters.DefectCluster(inputLattice, refLattice))
            
            # add atoms to cluster lists
            clusterIndexMapper = {}
            count = 0
            for i in range(NVac):
                atomIndex = vacancies[i]
                clusterIndex = defectCluster[i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                clusterList[clusterListIndex].addVacancy(atomIndex)
            
            for i in range(NInt):
                atomIndex = interstitials[i]
                clusterIndex = defectCluster[NVac + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                clusterList[clusterListIndex].addInterstitial(atomIndex)
            
            for i in range(NAnt):
                atomIndex = antisites[i]
                atomIndex2 = onAntisites[i]
                clusterIndex = defectCluster[NVac + NInt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                clusterList[clusterListIndex].addAntisite(atomIndex, atomIndex2)
            
            for i in range(NSplit):
                clusterIndex = defectCluster[NVac + NInt + NAnt + i]
                
                if clusterIndex not in clusterIndexMapper:
                    clusterIndexMapper[clusterIndex] = count
                    count += 1
                
                clusterListIndex = clusterIndexMapper[clusterIndex]
                index0 = splitInterstitials[3 * i]
                index1 = splitInterstitials[3 * i + 1]
                index2 = splitInterstitials[3 * i + 2]
                clusterList[clusterListIndex].addSplitInterstitial(index0, index1, index2)
            
            # calculate volumes
            calcVols = settings.getSetting("calculateVolumes")
            if calcVols:
                self.logger.debug("Calculating defect cluster volumes")
                for i, cluster in enumerate(clusterList):
                    cluster.calculateVolume(filterInput.voronoiDefects, settings)
                    volume = cluster.getVolume()
                    if volume is not None:
                        self.logger.debug("Cluster %d: volume is %f", i, volume)
                    area = cluster.getFacetArea()
                    if area is not None:
                        self.logger.debug("Cluster %d: facet area is %f", i, area)
            
            # hide defects if required
            if settings.getSetting("hideDefects"):
                vacancies.resize(0, refcheck=False)
                antisites.resize(0, refcheck=False)
                onAntisites.resize(0, refcheck=False)
                interstitials.resize(0, refcheck=False)
                splitInterstitials.resize(0, refcheck=False)
        
        # make result
        result = base.FilterResult()
        result.setClusterList(clusterList)
        
        # get spaghetti atoms
        if settings.getSetting("drawSpaghetti"):
            spaghettiAtoms = self.getSpaghettiAtoms(filterInput, vacancyRadius)
            result.setSpaghettiAtoms(spaghettiAtoms)
        
        # write a file containing the defects
        if settings.getSetting("writeDefectsFile"):
            filename = settings.getSetting("defectsFile")
            self.logger.info("Writing file containing defects: %s" % filename)
            with open(filename, "w") as fout:
                # total number of defects
                fout.write("%d total defects\n" % NDef)
                
                # vacancies
                fout.write("%d vacancies\n" % NVac)
                for i, sym in enumerate(refLattice.specieList):
                    fout.write("%d %s " % (vacSpecCount[i], sym))
                fout.write("\n")
                for index in vacancies:
                    pos = refLattice.atomPos(index)
                    fout.write("%s %f %f %f\n" % (refLattice.atomSym(index), pos[0], pos[1], pos[2]))
                
                # interstitials
                fout.write("%d interstitials\n" % NInt)
                for i, sym in enumerate(inputLattice.specieList):
                    fout.write("%d %s " % (intSpecCount[i], sym))
                fout.write("\n")
                for index in interstitials:
                    pos = inputLattice.atomPos(index)
                    fout.write("%s %f %f %f\n" % (inputLattice.atomSym(index), pos[0], pos[1], pos[2]))
                
                # split interstitials
                fout.write("%d split interstitials\n" % NSplit)
                for i in range(len(inputLattice.specieList)):
                    for j in range(i, len(inputLattice.specieList)):
                        if j == i:
                            N = splitIntSpecCount[i][j]
                        else:
                            N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                        fout.write("%d %s %s " % (N, inputLattice.specieList[i], inputLattice.specieList[j]))
                fout.write("\n")
                for i in range(NSplit):
                    ind1 = splitInterstitials[3 * i + 1]
                    ind2 = splitInterstitials[3 * i + 2]
                    
                    pos1 = inputLattice.pos[3 * ind1:3 * ind1 + 3]
                    pos2 = inputLattice.pos[3 * ind2:3 * ind2 + 3]
                    
                    sym1 = inputLattice.atomSym(ind1)
                    sym2 = inputLattice.atomSym(ind2)
                    
                    fout.write("%s %f %f %f %s %f %f %f\n" % (sym1, pos1[0], pos1[1], pos1[2], sym2, pos2[0], pos2[1],
                                                              pos2[2]))
                
                # antisites
                fout.write("%d antisites\n" % NAnt)
                for i, symref in enumerate(refLattice.specieList):
                    for j, syminp in enumerate(inputLattice.specieList):
                        if symref != syminp:
                            fout.write("%d %s %s " % (onAntSpecCount[i][j], syminp, symref))
                fout.write("\n")
                for index, refIndex in zip(onAntisites, antisites):
                    pos = inputLattice.atomPos(index)
                    refPos = refLattice.atomPos(refIndex)
                    fout.write("%s %f %f %f %s %f %f %f\n" % (inputLattice.atomSym(index), pos[0], pos[1], pos[2],
                                                              refLattice.atomSym(index), refPos[0], refPos[1],
                                                              refPos[2]))
        
        return result
    
    def getSpaghettiAtoms(self, filterInput, vacancyRadius):
        """
        Find atoms for rendering spaghetti.
        
        This means atoms that are displaced from their original site by more than the vacancy radius.
        
        """
        inputLattice = filterInput.inputState
        refLattice = filterInput.refState
        
        if inputLattice.NAtoms != refLattice.NAtoms:
            self.logger.warning("Cannot find spaghetti atoms if number of atoms in input and ref differ")
            spaghettiAtoms = np.empty(0, np.int32)
        
        else:
            # displacement filter settings
            dispSettings = displacementFilter.DisplacementFilterSettings()
            dispSettings.updateSetting("minDisplacement", vacancyRadius)
            dispSettings.updateSetting("filteringEnabled", True)
            
            # displacmeent filter input
            dispInput = base.FilterInput()
            dispInput.inputState = inputLattice
            dispInput.refState = refLattice
            dispInput.NScalars = 0
            dispInput.fullScalars = np.empty(dispInput.NScalars, np.float64)
            dispInput.NVectors = 0
            dispInput.fullVectors = np.empty(dispInput.NVectors, np.float64)
            dispInput.ompNumThreads = filterInput.ompNumThreads
            dispInput.visibleAtoms = np.arange(inputLattice.NAtoms, dtype=np.int32)
            dispInput.driftCompensation = filterInput.driftCompensation
            dispInput.driftVector = filterInput.driftVector
            
            # displacement filter
            disp = displacementFilter.DisplacementFilter("Displacement (spaghetti)")
            
            # run filter
            disp.apply(dispInput, dispSettings)
            
            # spaghetti atoms are the visible atoms
            spaghettiAtoms = dispInput.visibleAtoms
            self.logger.debug("Found %d 'spaghetti' atoms (atoms that have moved from their original site)",
                              len(spaghettiAtoms))
        
        return spaghettiAtoms
