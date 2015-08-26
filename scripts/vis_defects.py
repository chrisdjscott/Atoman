#!/usr/bin/env python

"""
You can use this module to find defects using the visualiser.
Examples are in the main() method. 

At some point I will rewrite the filters as objects so it will
be easier to use them separately

"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging

import numpy as np

from CDJSVis.filtering import _defects
from CDJSVis.filtering import _filtering
from CDJSVis.filtering import filterer
from CDJSVis.filtering import acna
from CDJSVis.state import latticeReaders
from CDJSVis.algebra import vectors
from CDJSVis.filtering import voronoi
from CDJSVis.filtering import clusters


class Settings(object):
    """
    Defects filter settings
    
    """
    def __init__(self, driftCompensation=0, findClusters=0, showVacancies=1, showInterstitials=1, showAntisites=1, vacancyRadius=1.0, 
                 neighbourRadius=5.0, minClusterSize=1, maxClusterSize=99999, identifySplitInts=1, allSpeciesSelected=1, useAcna=False,
                 acnaMaxBondDistance=5.0, acnaFilteringEnabled=0, acnaStructureType=1):
        self.driftCompensation = driftCompensation
        self.showVacancies = showVacancies
        self.showInterstitials = showInterstitials
        self.showAntisites = showAntisites
        self.vacancyRadius = vacancyRadius
        self.neighbourRadius = neighbourRadius
        self.minClusterSize = minClusterSize
        self.maxClusterSize = maxClusterSize
        self.identifySplitInts = identifySplitInts
        self.findClusters = findClusters
        self.allSpeciesSelected = allSpeciesSelected
        self.useAcna = useAcna
        self.acnaMaxBondDistance = acnaMaxBondDistance
        self.acnaFilteringEnabled = acnaFilteringEnabled
        if acnaStructureType < 0 or acnaStructureType >= len(filterer.Filterer.knownStructures):
            print "Error: invalid value for acnaStructureType => turning of ACNA filtering"
            self.acnaFilteringEnabled = 0
            self.acnaStructureType = 0
        else:
            self.acnaStructureType = acnaStructureType

class VoronoiOptions(object):
    """
    Dummy Voronoi options class
    
    """
    def __init__(self, useRadii=False, faceAreaThreshold=0.1):
        self.useRadii = useRadii
        self.faceAreaThreshold = faceAreaThreshold

def computeDriftVector(inputState, refState, cellDims, pbc):
    """
    Compute drift vector
    
    """
    driftVector = np.empty(3, np.float64)
    
    assert inputState.NAtoms == refState.NAtoms
    
    _filtering.calculate_drift_vector(refState.NAtoms, inputState.pos, refState.pos, cellDims, pbc, driftVector)
    
    logging.debug("Calculated drift vector: %r", driftVector)
    
    return driftVector

def calculateDefectClusterVolumes(inputLattice, refLattice, vacancies, clusterList, voronoiOptions, pbc=np.ones(3, np.int32)):
    """
    Calculate volumes of defect clusters
    
    """
    logging.debug("Calculating volumes of defect clusters")
    
    if len(inputLattice.cellDims) == 9:
        dims_mod = True
        cellDims = inputLattice.cellDims
        inputLattice.cellDims = np.empty(3, np.float64)
        inputLattice.cellDims[0] = cellDims[0]
        inputLattice.cellDims[1] = cellDims[4]
        inputLattice.cellDims[2] = cellDims[8]
    else:
        dims_mod = False
    
    # compute Voronoi
    vor = voronoi.computeVoronoiDefects(inputLattice, refLattice, vacancies, voronoiOptions, pbc)
    
    count = 0
    for cluster in clusterList:
        volume = 0.0
        
        # add volumes of interstitials
        for i in xrange(cluster.getNInterstitials()):
            index = cluster.interstitials[i]
            volume += vor.atomVolume(index)
        
        # add volumes of split interstitial atoms
        for i in xrange(cluster.getNSplitInterstitials()):
            index = cluster.splitInterstitials[3*i+1]
            volume += vor.atomVolume(index)
            index = cluster.splitInterstitials[3*i+2]
            volume += vor.atomVolume(index)
        
        # add volumes of on antisite atoms
        for i in xrange(cluster.getNAntisites()):
            index = cluster.onAntisites[i]
            volume += vor.atomVolume(index)
        
        # add volumes of vacancies
        for i in xrange(cluster.getNVacancies()):
            vacind = cluster.vacAsIndex[i]
            index = inputLattice.NAtoms + vacind
            volume += vor.atomVolume(index)
        
        cluster.volume = volume
        
        logging.debug("  Cluster %d (%d defects)", count, cluster.getNDefects())
        logging.debug("    volume is %f", volume)
        
        count += 1
    
    if dims_mod:
        inputLattice.cellDims = cellDims

def findDefects(inputLattice, refLattice, settings, acnaArray=None, pbc=np.ones(3, np.int32)):
    """
    Point defects filter
    
    """
    if len(refLattice.cellDims) == 9:
        cellDims = np.empty(3, np.float64)
        cellDims[0] = refLattice.cellDims[0]
        cellDims[1] = refLattice.cellDims[4]
        cellDims[2] = refLattice.cellDims[8]
    else:
        cellDims = refLattice.cellDims
    
    if settings.useAcna:
        logging.debug("Computing ACNA from point defects filter...")
        
        # dummy visible atoms and scalars arrays
        visAtoms = np.arange(inputLattice.NAtoms, dtype=np.int32)
        acnaArray = np.empty(inputLattice.NAtoms, np.float64)
        NScalars = 0
        fullScalars = np.empty(NScalars, np.float64)
        NVectors = 0
        fullVectors = np.empty(NVectors, np.float64)
        structVis = np.ones(len(filterer.Filterer.knownStructures), np.int32)
        
        # counter array
        counters = np.zeros(7, np.int32)
        
        # number of threads
        numThreads = 1
        
        acna.adaptiveCommonNeighbourAnalysis(visAtoms, inputLattice.pos, acnaArray, inputLattice.minPos, inputLattice.maxPos, 
                                             cellDims, pbc	, NScalars, fullScalars, settings.acnaMaxBondDistance, counters, 
                                             0, structVis, numThreads, NVectors, fullVectors) 
        
        # store counters
        d = {}
        for i, structure in enumerate(filterer.Filterer.knownStructures):
            if counters[i] > 0:
                d[structure] = counters[i]
        
        logging.debug("  %r", d)
    
    elif acnaArray is None or len(acnaArray) != inputLattice.NAtoms:
        acnaArray = np.empty(0, np.float64)
    
    if settings.driftCompensation:
        driftVector = computeDriftVector(inputLattice, refLattice, cellDims, pbc)
    else:
        driftVector = np.zeros(3, np.float64)
    
    # set up arrays
    interstitials = np.empty(inputLattice.NAtoms, np.int32)
    
    if settings.identifySplitInts:
        splitInterstitials = np.empty(inputLattice.NAtoms, np.int32)
    
    else:
        splitInterstitials = np.empty(0, np.int32)
            
    vacancies = np.empty(refLattice.NAtoms, np.int32)
    
    antisites = np.empty(refLattice.NAtoms, np.int32)
    onAntisites = np.empty(refLattice.NAtoms, np.int32)
    
    # set up excluded specie arrays
    if settings.allSpeciesSelected:
        exclSpecsInput = np.zeros(0, np.int32)
        exclSpecsRef = np.zeros(0, np.int32)
    
    else:
        exclSpecs = []
#         for i in xrange(len(inputLattice.specieList)):
#             spec = inputLattice.specieList[i]
#             if spec not in settings.visibleSpecieList:
#                 exclSpecs.append(i)
        exclSpecsInput = np.empty(len(exclSpecs), np.int32)
        for i in xrange(len(exclSpecs)):
            exclSpecsInput[i] = exclSpecs[i]
        
        exclSpecs = []
#         for i in xrange(len(refLattice.specieList)):
#             spec = refLattice.specieList[i]
#             if spec not in settings.visibleSpecieList:
#                 exclSpecs.append(i)
        exclSpecsRef = np.empty(len(exclSpecs), np.int32)
        for i in xrange(len(exclSpecs)):
            exclSpecsRef[i] = exclSpecs[i]
    
    # specie counter arrays
    vacSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
    intSpecCount = np.zeros( len(inputLattice.specieList), np.int32 )
    antSpecCount = np.zeros( len(refLattice.specieList), np.int32 )
    onAntSpecCount = np.zeros( (len(refLattice.specieList), len(inputLattice.specieList)), np.int32 )
    splitIntSpecCount = np.zeros( (len(inputLattice.specieList), len(inputLattice.specieList)), np.int32 )
    
    NDefectsByType = np.zeros(6, np.int32)
    
    if settings.findClusters:
        defectCluster = np.empty(inputLattice.NAtoms + refLattice.NAtoms, np.int32)
    
    else:
        defectCluster = np.empty(0, np.int32)
    
    # call C library
    status = _defects.findDefects(settings.showVacancies, settings.showInterstitials, settings.showAntisites, NDefectsByType, vacancies, 
                                  interstitials, antisites, onAntisites, exclSpecsInput, exclSpecsRef, inputLattice.NAtoms, inputLattice.specieList,
                                  inputLattice.specie, inputLattice.pos, refLattice.NAtoms, refLattice.specieList, refLattice.specie, 
                                  refLattice.pos, cellDims, pbc, settings.vacancyRadius,
                                  settings.findClusters, settings.neighbourRadius, defectCluster, vacSpecCount, intSpecCount, antSpecCount,
                                  onAntSpecCount, splitIntSpecCount, settings.minClusterSize, settings.maxClusterSize, splitInterstitials, 
                                  settings.identifySplitInts, settings.driftCompensation, driftVector, acnaArray, settings.acnaStructureType)
    
    # summarise
    NDef = NDefectsByType[0]
    NVac = NDefectsByType[1]
    NInt = NDefectsByType[2]
    NAnt = NDefectsByType[3]
    NSplit = NDefectsByType[5]
    vacancies.resize(NVac)
    interstitials.resize(NInt)
    antisites.resize(NAnt)
    onAntisites.resize(NAnt)
    splitInterstitials.resize(NSplit*3)
    
    # report counters
    logging.info("Found %d defects", NDef)
    
    if settings.showVacancies:
        logging.debug("  %d vacancies", NVac)
        for i in xrange(len(refLattice.specieList)):
            logging.debug("    %d %s vacancies", vacSpecCount[i], refLattice.specieList[i])
    
    if settings.showInterstitials:
        logging.debug("  %d interstitials", NInt + NSplit)
        for i in xrange(len(inputLattice.specieList)):
            logging.debug("    %d %s interstitials", intSpecCount[i], inputLattice.specieList[i])
    
        if settings.identifySplitInts:
            logging.debug("    %d split interstitials", NSplit)
            for i in xrange(len(inputLattice.specieList)):
                for j in xrange(i, len(inputLattice.specieList)):
                    if j == i:
                        N = splitIntSpecCount[i][j]
                    else:
                        N = splitIntSpecCount[i][j] + splitIntSpecCount[j][i]
                    logging.debug("      %d %s - %s split interstitials", N, inputLattice.specieList[i], inputLattice.specieList[j])
    
    if settings.showAntisites:
        logging.debug("  %d antisites", NAnt)
        for i in xrange(len(refLattice.specieList)):
            for j in xrange(len(inputLattice.specieList)):
                if inputLattice.specieList[j] == refLattice.specieList[i]:
                    continue
                
                logging.debug("    %d %s on %s antisites", onAntSpecCount[i][j], inputLattice.specieList[j], refLattice.specieList[i])
    
    if settings.identifySplitInts:
        logging.debug("Split interstitial analysis")
        
        PBC = pbc
        
        for i in xrange(NSplit):
            ind1 = splitInterstitials[3*i+1]
            ind2 = splitInterstitials[3*i+2]
            
            pos1 = inputLattice.pos[3*ind1:3*ind1+3]
            pos2 = inputLattice.pos[3*ind2:3*ind2+3]
            
            sepVec = vectors.separationVector(pos1, pos2, cellDims, PBC)
            norm = vectors.normalise(sepVec)
            
            logging.debug("  Orientation of split int %d: (%.3f %.3f %.3f)", i, norm[0], norm[1], norm[2])
    
    # sort clusters here
    clusterList = []
    if settings.findClusters:
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
    
    return NDef, interstitials, vacancies, antisites, onAntisites, splitInterstitials, vacSpecCount, intSpecCount, splitIntSpecCount, clusterList

def computeACNA(inputState, settings):
    """
    ACNA filter
    
    """
    if len(inputState.cellDims) == 9:
        cellDims = np.empty(3, np.float64)
        cellDims[0] = inputState.cellDims[0]
        cellDims[1] = inputState.cellDims[4]
        cellDims[2] = inputState.cellDims[8]
    else:
        cellDims = inputState.cellDims
    
    visibleAtoms = np.arange(inputState.NAtoms, dtype=np.int32)
    scalars = np.zeros(inputState.NAtoms, dtype=np.float64)
    NScalars = 0
    fullScalars = np.empty(NScalars, np.float64)
    NVectors = 0
    fullVectors = np.empty(NVectors, np.float64)
    
    pbc = np.ones(3, np.int32)
    
    # counter array
    counters = np.zeros(7, np.int32)
    
    NVisible = acna.adaptiveCommonNeighbourAnalysis(visibleAtoms, inputState.pos, scalars, cellDims, pbc, NScalars, fullScalars,
                                                    settings.maxBondDistance, counters, settings.filteringEnabled, settings.structureVisibility,
                                                    1, NVectors, fullVectors)
    
    # resize visible atoms
    visibleAtoms.resize(NVisible, refcheck=False)

    # store scalars
    scalars.resize(NVisible, refcheck=False)
    
    # store counters
    d = {}
    for i, structure in enumerate(filterer.Filterer.knownStructures):
        if counters[i] > 0:
            d[structure] = counters[i]
    
    return scalars, d

def main():
    """
    This is an example of how to use this module...
    
    * Read in a reference (lattice, animation-reference, whatever)
    * make your settings object
    * Read in your input (or loop over them)
    * Call findDefects
    
    Obviously these examples will only run on my computer.
    
    """
    # set logging level
#    logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    # read in lattice files
    dat_reader = latticeReaders.LbomdDatReader("/tmp", latticeReaders.basic_log, latticeReaders.basic_displayWarning, 
                                                   latticeReaders.basic_displayError)
    ref_reader = latticeReaders.LbomdRefReader("/tmp", latticeReaders.basic_log, latticeReaders.basic_displayWarning, 
                                               latticeReaders.basic_displayError)
    xyz_reader = latticeReaders.LbomdXYZReader("/tmp", latticeReaders.basic_log, latticeReaders.basic_displayWarning, 
                                                   latticeReaders.basic_displayError)
    
    work_dir = os.getcwd()
    
    fin_file = sys.argv[1]
    ref_file = sys.argv[2]
    
#     ref_file = os.path.join(work_dir, "ref.dat")
#     animref_file = os.path.join(work_dir, "animation-reference.xyz")
#     fin_file = os.path.join(work_dir, "final-relaxed.dat")
#     fin_file2 = os.path.join(work_dir, "PuGaH0200.xyz")
#     early_file = os.path.join(work_dir, "PuGaH0020.xyz")
    
    status, ref = dat_reader.readFile(ref_file)
    assert not status
#     status, animref = ref_reader.readFile(animref_file)
#     assert not status
    status, finrel = dat_reader.readFile(fin_file)
    assert not status
#     status, fin = xyz_reader.readFile(fin_file2, animref)
#     assert not status
#     status, early = xyz_reader.readFile(early_file, animref)
#     assert not status
    
    print "="*120
    
    # compute normal defects
    settings = Settings(showAntisites=0, vacancyRadius=1.3, driftCompensation=1)
    res = findDefects(finrel, ref, settings)
    print "NDEF", res[0]
    
    print "="*120
    
    # compute defects with ACNA
    settings = Settings(showAntisites=0, vacancyRadius=1.3, driftCompensation=1, useAcna=True)
    res = findDefects(finrel, ref, settings)
    print "NDEF", res[0]
    


if __name__ == "__main__":
    main()
