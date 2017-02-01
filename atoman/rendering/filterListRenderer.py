
"""
Renderer for the FilterList object.

@author: Chris Scott

"""
from __future__ import absolute_import
from __future__ import unicode_literals
import copy
import logging

import numpy as np
from PySide import QtCore

from . import _rendering
from . import utils
from ..algebra import vectors as vectorslib
from .renderers import atomRenderer
from .renderers import vectorRenderer
from .renderers import bondRenderer
from .renderers import vacancyRenderer
from .renderers import antisiteRenderer
from .renderers import clusterRenderer
from .renderers import voronoiRenderer
from ..system.atoms import elements
import six
from six.moves import range
from six.moves import zip


class FilterListRenderer(object):
    """
    Renderer for a filter list.
    
    """
    def __init__(self, filterList):
        # logger
        self._logger = logging.getLogger(__name__)
        
        # the filter list we belong to
        self._filterList = filterList
        
        # the filterer that we are rendering
        self._filterer = filterList.filterer
        
        # temporary directory
        self._tmpDirectory = filterList.mainWindow.tmpDirectory
        
        # dictionaries for storing current actors
        self._renderersDict = {}
        self._traceCoords = np.empty((0, 3), dtype=np.float64)
        self._traceVectors = np.empty((0, 3), dtype=np.float64)
        self._traceScalars = np.empty(0, dtype=np.float64)
        self._tracePreviousPos = None
        self._scalarBarWhite = None
        self._scalarBarBlack = None
        self.scalarBarAdded = False
        self.povrayAtomsWritten = False
        
        # get required refs from filter list
        self.pipelinePage = filterList.pipelinePage
        self.rendererWindows = filterList.pipelinePage.rendererWindows
        self.colouringOptions = filterList.colouringOptions
        self.displayOptions = filterList.displayOptions
        self.vectorsOptions = filterList.vectorsOptions
        self.bondsOptions = filterList.bondsOptions
        self.actorsOptions = filterList.actorsOptions
        self.traceOptions = filterList.traceOptions
        self.voronoiOptions = filterList.voronoiOptions
    
    def render(self, sequencer=False):
        """
        Render the data provided by the Filterer.
        
        Workflow:
            - Render atoms
            - Write povray atoms...
            - Render interstitials
            - Render vacancies
            - Render antisites/onAntisites
            - Render clusters
            - Render bubbles
            - Render bonds (calculate?)
            - Render Voronoi?
            - Make scalar bar etc...
        
        Just go through everything and render what is available.
        
        Onscreen info is probably going to come from here too
        
        """
        self._logger.debug("Rendering filter list")
        
        # local refs
        inputState = self._filterer.inputState
        visibleAtoms = self._filterer.visibleAtoms
        
        # set resolution
        numForRes = len(visibleAtoms) + len(self._filterer.interstitials) + len(self._filterer.onAntisites)
        resolution = utils.setRes(numForRes, self.displayOptions)
        self._logger.debug("Setting resolution to %d (num %d)", resolution, numForRes)
        
        # make points data
        atomPoints = _rendering.makeVisiblePointsArray(visibleAtoms, inputState.pos)
        atomPoints = utils.NumpyVTKData(atomPoints)
        
        # make radius array
        radiusArray = _rendering.makeVisibleRadiusArray(visibleAtoms, inputState.specie,
                                                        inputState.specieCovalentRadius)
        radiusArray = utils.NumpyVTKData(radiusArray, name="radius")
        
        # get the scalars array
        scalarsArray = self._getScalarsArray(inputState, visibleAtoms)
        
        # make the look up table
        lut = utils.setupLUT(inputState.specieList, inputState.specieRGB, self.colouringOptions)
        
        # render atoms
        self._renderAtoms(atomPoints, scalarsArray, radiusArray, lut, resolution)
        
        # render defects
        self._renderDefects(lut, resolution)
        
        # bonds
        self._renderBonds(scalarsArray, lut)
        
        # render vectors
        self._renderVectors(atomPoints, scalarsArray, lut)
            
        # displacement vectors
        self._renderDisplacementVectors(lut)
        
        # trace
        self._renderTrace(scalarsArray, lut)
        
        # voronoi
        self._renderVoronoi(scalarsArray, lut)
        
        # render clusters
        self._renderClusters()
        
        # render bubbles
        self._renderBubbles(lut, resolution)
        
        # scalar bar
        self._createScalarBar(lut)
        
        # refresh actors options
        self.actorsOptions.refresh(self.getActorsDict())
    
    def _renderBubbles(self, lut, resolution):
        """Render bubbles."""
        bubbleList = self._filterer.bubbleList
        if len(bubbleList):
            self._logger.debug("Rendering bubbles")
            
            # get the bubble filter settings
            found = False
            for name, settings in zip(self._filterer.currentFilters, self._filterer.currentSettings):
                if name == "Bubbles":
                    found = True
                    break
            if not found:
                raise RuntimeError("Could not find bubbles filter settings")
            
            # get a list of all bubble vacancies and atoms
            bubbleVacs, bubbleAtoms = self._filterer.getBubblesIndices()
            
            # render the vacancies
            if len(bubbleVacs):
                self._renderAntisites(lut, antisites=bubbleVacs, name="Bubble vacancies")
            
            # render the atoms
            if len(bubbleAtoms):
                self._renderDefectAtoms(lut, resolution, bubbleAtoms, "Bubble atoms")
    
    def _renderVoronoi(self, scalars, lut):
        """Render Voronoi cells."""
        visibleAtoms = self._filterer.visibleAtoms
        if self.voronoiOptions.displayVoronoi and len(visibleAtoms):
            self._logger.debug("Rendering Voronoi cells")
            
            # warn the user if they are rendering a large number of cells
            if len(visibleAtoms) > 2000:
                # warn that this will be slow
                msg = "Rendering a large number of Voronoi cells (%d); this will be slow" % len(visibleAtoms)
                self._logger.warning(msg)
            
            # get the Voronoi object
            lattice = self._filterer.inputState
            voro = self._filterer.voronoiAtoms.getVoronoi(lattice)
            
            # render
            rend = voronoiRenderer.VoronoiRenderer()
            rend.render(lattice, visibleAtoms, scalars.getNumpy(), lut, voro, self.voronoiOptions,
                        self.colouringOptions)
            self._renderersDict["Voronoi"] = rend
    
    def _createScalarBar(self, lut):
        """Create the scalar bars."""
        if self.colouringOptions.colourBy != "Species" and self.colouringOptions.colourBy != "Solid colour":
            prefs = self._filterList.mainWindow.preferences.renderingForm
            self._scalarBarWhite = utils.makeScalarBar(lut, self.colouringOptions, (0, 0, 0), prefs)
            self._scalarBarBlack = utils.makeScalarBar(lut, self.colouringOptions, (1, 1, 1), prefs)
    
    def addScalarBar(self):
        """Show the scalar bar."""
        haveScalarBar = self._scalarBarWhite is not None
        scalarBarChecked = self._filterList.scalarBarButton.isChecked()
        alreadyAdded = self._filterList.filterTab.scalarBarAdded
        if haveScalarBar and scalarBarChecked and not alreadyAdded:
            toolbar = self._filterList.pipelinePage.mainToolbar
            for rw in self.rendererWindows:
                if rw.currentPipelineString == toolbar.currentPipelineString:
                    # which scalar bar to add
                    if rw.blackBackground:
                        scalarBar = self._scalarBarBlack
                    else:
                        scalarBar = self._scalarBarWhite
                    
                    rw.vtkRen.AddActor2D(scalarBar)
                    rw.vtkRenWinInteract.ReInitialize()
            
            self._filterList.filterTab.scalarBarAdded = True
            self.scalarBarAdded = True
        
        return self.scalarBarAdded
    
    def hideScalarBar(self):
        """Hide the scalar bar."""
        toolbar = self._filterList.pipelinePage.mainToolbar
        if self.scalarBarAdded:
            for rw in self.rendererWindows:
                if rw.currentPipelineString == toolbar.currentPipelineString:
                    # which scalar bar was added
                    if rw.blackBackground:
                        scalarBar = self._scalarBarBlack
                    else:
                        scalarBar = self._scalarBarWhite
                    
                    rw.vtkRen.RemoveActor2D(scalarBar)
                    rw.vtkRenWinInteract.ReInitialize()
            
            self._filterList.pipelinePage.scalarBarAdded = False
            self.scalarBarAdded = False
    
    def _renderTrace(self, scalars, lut):
        """Render trace vectors."""
        visibleAtoms = self._filterer.visibleAtoms
        if self.traceOptions.drawTraceVectors and len(visibleAtoms):
            self._logger.debug("Rendering trace")
            
            # refs
            inputState = self._filterer.inputState
            refState = self._filterer.refState
            
            # previous positions to draw trace vector from
            if self._tracePreviousPos is None:
                self._tracePreviousPos = refState.pos
            
            # check lengths are the same
            if len(self._tracePreviousPos) == len(inputState.pos):
                # calculate displacements from previous positions
                calc = bondRenderer.DisplacmentVectorCalculator()
                result = calc.calculateDisplacementVectors(inputState.pos, self._tracePreviousPos, inputState.PBC,
                                                           inputState.cellDims, visibleAtoms, scalars.getNumpy())
                traceCoords, traceVectors, traceScalars = result
                
                # append to previously stored data
                traceCoords = traceCoords.getNumpy()
                traceVectors = traceVectors.getNumpy()
                traceScalars = traceScalars.getNumpy()
                self._traceCoords = np.concatenate((self._traceCoords, traceCoords))
                traceCoords = utils.NumpyVTKData(self._traceCoords)
                self._traceVectors = np.concatenate((self._traceVectors, traceVectors))
                traceVectors = utils.NumpyVTKData(self._traceVectors, name="vectors")
                self._traceScalars = np.concatenate((self._traceScalars, traceScalars))
                traceScalars = utils.NumpyVTKData(self._traceScalars, name="colours")
                
                if not len(self._traceCoords):
                    self._logger.debug("No trace vectors to render")
                
                else:
                    # draw trace vectors
                    if self.traceOptions.drawAsArrows:
                        vecRend = vectorRenderer.VectorRenderer()
                        vecRend.render(traceCoords, traceScalars, traceVectors, len(inputState.specieList),
                                       self.colouringOptions, self.traceOptions, lut, invert=True)
                    
                    else:
                        self._logger.debug("Size of trace: %d", len(self._traceCoords))
                        vecRend = bondRenderer.BondRenderer()
                        vecRend.render(traceCoords, traceVectors, traceScalars, len(inputState.specieList),
                                       self.colouringOptions, self.traceOptions, lut)
                    
                    self._renderersDict["Trace vectors"] = vecRend
            
            else:
                self._logger.warning("Cannot compute trace with differing number of atoms between steps")
            
            # store positions for next time
            self._tracePreviousPos = copy.deepcopy(inputState.pos)
    
    def _renderDisplacmentVectorsList(self, atomList, lut, name):
        """Render displacement for the given list of atoms."""
        if not len(atomList):
            self._logger.debug("No %s displacement vectors to draw", name)
        
        else:
            self._logger.debug("Rendering %s displacement vectors", name)
            
            # local refs
            inputState = self._filterer.inputState
            refState = self._filterer.refState
            
            # scalars array
            scalars = self._getScalarsArray(inputState, atomList)
            
            # calculate displacement vectors
            calc = bondRenderer.DisplacmentVectorCalculator()
            result = calc.calculateDisplacementVectors(inputState.pos, refState.pos, inputState.PBC,
                                                       inputState.cellDims, atomList, scalars.getNumpy())
            bondCoords, bondVectors, bondScalars = result
            if not len(bondCoords.getNumpy()):
                self._logger.debug("No displacement vectors were calculated")
                return
            
            # draw displacement vectors
            vecRend = bondRenderer.BondRenderer()
            vecRend.render(bondCoords, bondVectors, bondScalars, len(inputState.specieList), self.colouringOptions,
                           self.bondsOptions, lut)
            self._renderersDict["{0} displacement vectors".format(name)] = vecRend
    
    def _renderDisplacementVectors(self, lut):
        """Render displacement vectors for atoms/interstitials."""
        # input/ref lattices
        inputState = self._filterer.inputState
        refState = self._filterer.refState
        
        # check if the number of atoms match
        numAtomsMatch = inputState.NAtoms == refState.NAtoms
        
        # loop over settings, drawing displacement vectors as required
        for name, settings in zip(self._filterer.currentFilters, self._filterer.currentSettings):
            if name == "Displacement" and settings.getSetting("drawDisplacementVectors"):
                if not numAtomsMatch:
                    msg = "Cannot render atom displacement vectors when num atoms in ref and input differ"
                    self._logger.warning(msg)
                else:
                    self._renderDisplacmentVectorsList(self._filterer.visibleAtoms, lut, "Atom")
            
            elif name == "Point defects" and settings.getSetting("drawDisplacementVectors"):
                if not numAtomsMatch:
                    msg = "Cannot render interstitial displacement vectors when num atoms in ref and input differ"
                    self._logger.warning(msg)
                else:
                    # TODO: split ints should be done too
                    self._renderDisplacmentVectorsList(self._filterer.interstitials, lut, "Interstitial")
            
            elif name == "Point defects" and settings.getSetting("drawSpaghetti"):
                if not numAtomsMatch:
                    self._logger.warning("Cannot render spaghetti when num atoms in ref and input differ")
                else:
                    self._renderDisplacmentVectorsList(self._filterer.spaghettiAtoms, lut, "Spaghetti")
    
    def _renderDefects(self, lut, resolution):
        """Render defects."""
        self._logger.debug("Rendering defects")
        
        # local refs
        interstitials = self._filterer.interstitials
        vacancies = self._filterer.vacancies
        onAntisites = self._filterer.onAntisites
        splitInterstitials = self._filterer.splitInterstitials
        
        # render interstitials
        self._renderDefectAtoms(lut, resolution, interstitials, "Interstitials")
        
        # render on antisite atoms
        self._renderDefectAtoms(lut, resolution, onAntisites, "Antisites occupying")
        
        # split interstitials arrays of ints and vacs
        splitInts = np.concatenate((splitInterstitials[1::3], splitInterstitials[2::3]))
        splitVacs = splitInterstitials[::3]
        
        # render split interstitial atoms
        self._renderDefectAtoms(lut, resolution, splitInts, "Split interstitial atoms")
        
        # render split vacancies
        self._renderVacancies(lut, splitVacs, actorName="Split interestitial vacancies")
        
        # render vacancies
        self._renderVacancies(lut, vacancies)
        
        # render antisites
        self._renderAntisites(lut)
    
    def _renderAntisites(self, lut, antisites=None, name="Antisites"):
        """Render antisites."""
        # local refs
        refState = self._filterer.refState
        if antisites is None:
            antisites = self._filterer.antisites
        if not len(antisites):
            return
        self._logger.debug("Rendering %s", name)
        
        # points
        points = _rendering.makeVisiblePointsArray(antisites, refState.pos)
        points = utils.NumpyVTKData(points)
        
        # radii
        radius = _rendering.makeVisibleRadiusArray(antisites, refState.specie, refState.specieCovalentRadius)
        radius = utils.NumpyVTKData(radius, name="radius")
        
        # scalars
        scalars = self._getScalarsArray(refState, antisites)
        
        # renderer
        rend = antisiteRenderer.AntisiteRenderer()
        rend.render(points, scalars, radius, len(refState.specieList), self.colouringOptions,
                    self.displayOptions.atomScaleFactor, lut)
        self._renderersDict[name] = rend
    
    def _renderVacancies(self, lut, vacancies, actorName="Vacancies", settings=None):
        """Render vacancies."""
        # local refs
        refState = self._filterer.refState
        
        if not len(vacancies):
            return
        self._logger.debug("Rendering %s", actorName)
        
        # points
        points = _rendering.makeVisiblePointsArray(vacancies, refState.pos)
        points = utils.NumpyVTKData(points)
        
        # radii
        radius = _rendering.makeVisibleRadiusArray(vacancies, refState.specie, refState.specieCovalentRadius)
        radius = utils.NumpyVTKData(radius, name="radius")
        
        # scalars
        scalars = self._getScalarsArray(refState, vacancies)
        
        # get vacancy scale setting
        if settings is None:
            found = False
            for name, settings in zip(self._filterer.currentFilters, self._filterer.currentSettings):
                if name == "Point defects":
                    found = True
                    break
            if not found:
                raise RuntimeError("Could not find point defects filter settings")
        
        # render
        rend = vacancyRenderer.VacancyRenderer()
        rend.render(points, scalars, radius, len(refState.specieList), self.colouringOptions,
                    self.displayOptions.atomScaleFactor, lut, settings)
        self._renderersDict[actorName] = rend
    
    def _renderDefectAtoms(self, lut, resolution, atomList, actorName):
        """Render defect atoms, eg interstitials."""
        if not len(atomList):
            return
        self._logger.debug("Rendering %s", actorName)
        
        # local refs
        inputState = self._filterer.inputState
        
        # points
        points = _rendering.makeVisiblePointsArray(atomList, inputState.pos)
        points = utils.NumpyVTKData(points)
        
        # radii
        radius = _rendering.makeVisibleRadiusArray(atomList, inputState.specie, inputState.specieCovalentRadius)
        radius = utils.NumpyVTKData(radius, name="radius")
        
        # scalars
        scalars = self._getScalarsArray(inputState, atomList)
        
        # render
        rend = atomRenderer.AtomRenderer()
        rend.render(points, scalars, radius, len(inputState.specieList), self.colouringOptions,
                    self.displayOptions.atomScaleFactor, lut, resolution)
        self._renderersDict[actorName] = rend
    
    def _renderBonds(self, scalarsArray, lut):
        """Calculate and render bonds between atoms."""
        # check if we need to calculate
        if not self.bondsOptions.drawBonds:
            return
        
        haveBonds = False
        for i in range(self.bondsOptions.bondsList.count()):
            item = self.bondsOptions.bondsList.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                haveBonds = True
                break
        if not haveBonds:
            return
        
        self._logger.info("Calculating bonds")
        
        # visible atoms array
        visibleAtoms = self._filterer.visibleAtoms
        
        if not len(visibleAtoms):
            self._logger.info("No visible atoms so no bonds to render")
            return
        
        # local refs
        inputState = self.pipelinePage.inputState
        specieList = inputState.specieList
        NSpecies = len(specieList)
        bondDict = elements.bondDict
        
        # bonds arrays
        bondMinArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        bondMaxArray = np.zeros((NSpecies, NSpecies), dtype=np.float64)
        
        # construct bonds array
        calcBonds = False
        maxBond = -1
        drawList = []
        for i in range(self.bondsOptions.bondsList.count()):
            item = self.bondsOptions.bondsList.item(i)
            
            if item.checkState() == QtCore.Qt.Checked:
                syma = item.syma
                symb = item.symb
                
                # check if in current specie list and if so what indexes
                if syma in specieList:
                    indexa = inputState.getSpecieIndex(syma)

                    if symb in specieList:
                        indexb = inputState.getSpecieIndex(symb)
                
                        if syma in bondDict:
                            d = bondDict[syma]
                            
                            if symb in d:
                                bondMin, bondMax = d[symb]
                                bondMinArray[indexa][indexb] = bondMin
                                bondMinArray[indexb][indexa] = bondMin
                                bondMaxArray[indexa][indexb] = bondMax
                                bondMaxArray[indexb][indexa] = bondMax
                                maxBond = max(bondMax, maxBond)
                                if bondMax > 0:
                                    calcBonds = True
                                drawList.append("%s-%s" % (syma, symb))
                                self._logger.info("Pair: %s - %s; bond range: %f -> %f", syma, symb, bondMin, bondMax)
        
        assert bondMaxArray.max() == bondMax
        
        if not calcBonds:
            self._logger.info("No bonds to calculate")
            return
        
        # calculate bonds
        bonds = bondRenderer.BondCalculator()
        result = bonds.calculateBonds(inputState, visibleAtoms, scalarsArray, bondMinArray, bondMaxArray, drawList)
        bondCoords, bondVectors, bondScalars, bondSpecieCounter = result
        if len(bondCoords.getNumpy()) == 0:
            self._logger.info("No bonds to render")
            return
        
        # draw bonds
        bondRend = bondRenderer.BondRenderer()
        bondRend.render(bondCoords, bondVectors, bondScalars, NSpecies, self.colouringOptions, self.bondsOptions, lut)
        self._renderersDict["Bonds"] = bondRend
    
    def _renderClusters(self):
        """Render clusters."""
        clusterList = self._filterer.clusterList
        if not len(clusterList):
            return
        self._logger.debug("Rendering clusters")
        
        # get filter settings
        found = False
        for name, settings in zip(self._filterer.currentFilters, self._filterer.currentSettings):
            if name == "Point defects" or name == "Cluster":
                found = True
                break
        if not found:
            raise RuntimeError("Could not find clusters or point defects filter settings")
        refState = self._filterer.refState if name == "Point defects" else None
        
        # check if we are supposed to be rendering clusters
        if settings.getSetting("drawConvexHulls") and settings.getSetting("hullOpacity") > 0:
            # render
            rend = clusterRenderer.ClusterRenderer()
            rend.render(clusterList, settings, refState=refState)
            self._renderersDict["Clusters"] = rend
    
    def _renderAtoms(self, atomPoints, scalarsArray, radiusArray, lut, resolution):
        """Render atoms."""
        if not len(self._filterer.visibleAtoms):
            return
        self._logger.debug("Rendering atoms")
        
        inputState = self._filterer.inputState
        atomRend = atomRenderer.AtomRenderer()
        atomRend.render(atomPoints, scalarsArray, radiusArray, len(inputState.specieList), self.colouringOptions,
                        self.displayOptions.atomScaleFactor, lut, resolution)
        self._renderersDict["Atoms"] = atomRend
    
    def _renderVectors(self, atomPoints, scalarsArray, lut):
        """Render vectors."""
        vectorsName = self.vectorsOptions.selectedVectorsName
        vectorsDict = self._filterer.vectorsDict
        if vectorsName is not None and vectorsName not in vectorsDict:
            self._logger.warning("Skipping adding vectors because could not find array: '%s'", vectorsName)
        
        elif vectorsName is not None:
            self._logger.debug("Rendering arrows for vector data: '%s'", vectorsName)
            
            # input lattice
            inputState = self._filterer.inputState
            
            # make the VTK vectors array
            vectors = vectorsDict[vectorsName]
            if self.vectorsOptions.vectorNormalise:
                vectors = vectorslib.normalise(vectors)
            vectors = utils.NumpyVTKData(vectors, name="vectors")
            
            # render vectors
            vectorRend = vectorRenderer.VectorRenderer()
            vectorRend.render(atomPoints, scalarsArray, vectors, len(inputState.specieList), self.colouringOptions,
                              self.vectorsOptions, lut)
            self._renderersDict["Vectors"] = vectorRend
    
    def _getScalarsArray(self, lattice, atomList):
        """
        Returns the scalars array (np and vtk versions)
        
        """
        # local refs
        colouringOptions = self.colouringOptions
        
        # scalar type
        scalarType = utils.getScalarsType(colouringOptions)
        
        # scalars array
        if scalarType == 5:
            if colouringOptions.colourBy.startswith("Lattice: "):
                scalarsArray = self._filterer.latticeScalarsDict[colouringOptions.colourBy[9:]]
            else:
                scalarsArray = self._filterer.scalarsDict[colouringOptions.colourBy]
        
        else:
            if scalarType == 0:
                scalarsFull = np.asarray(lattice.specie, dtype=np.float64)
            elif scalarType == 1:
                scalarsFull = lattice.pos[colouringOptions.heightAxis::3]
            elif scalarType == 4:
                scalarsFull = lattice.charge
            else:
                self._logger.error("Unrecognised scalar type (%d): defaulting to specie", scalarType)
                scalarsFull = lattice.specie
            scalarsArray = _rendering.makeVisibleScalarArray(atomList, scalarsFull)
        
        scalarsArray = utils.NumpyVTKData(scalarsArray, name="colours")
        
        return scalarsArray
    
    def _getCurrentRendererWindows(self):
        """Returns a list of the current renderer windows."""
        rws = []
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                rws.append(rw)
        
        return rws
    
    def removeActors(self, sequencer=False):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self._renderersDict = {}
        if not sequencer:
            self._traceCoords = np.empty((0, 3), dtype=np.float64)
            self._traceVectors = np.empty((0, 3), dtype=np.float64)
            self._traceScalars = np.empty(0, dtype=np.float64)
            self._tracePreviousPos = None
        
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        self.povrayAtomsWritten = False
        
        # refresh actors options
        self.actorsOptions.refresh(self.getActorsDict())
    
    def getActorsDict(self):
        """Return dict containing current actors."""
        actorsDict = {}
        for name, renderer in six.iteritems(self._renderersDict):
            actorsDict[name] = renderer.getActor()
        
        return actorsDict
    
    def hideActors(self):
        """
        Hide all actors
        
        """
        # renderer windows associated with this filter list
        rendererWindows = self._getCurrentRendererWindows()
        
        # actors dict
        actorsDict = self.getActorsDict()
        
        # loop over actors
        for actorName, actorObj in six.iteritems(actorsDict):
            if actorObj.visible:
                self._logger.debug("Removing actor: '%s'", actorName)
                for rw in rendererWindows:
                    rw.vtkRen.RemoveActor(actorObj.actor)
                
                actorObj.visible = False
        
        # reinitialise renderer windows
        for rw in rendererWindows:
            rw.vtkRenWinInteract.ReInitialize()
        
        self.hideScalarBar()
    
    def setActorAmbient(self, actorName, ambient, reinit=True):
        """Set ambient property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        actorObj.actor.GetProperty().SetAmbient(ambient)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecular(self, actorName, specular, reinit=True):
        """Set specular property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        actorObj.actor.GetProperty().SetSpecular(specular)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecularPower(self, actorName, specularPower, reinit=True):
        """Set specular power property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        actorObj.actor.GetProperty().SetSpecularPower(specularPower)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def getActorAmbient(self, actorName):
        """Get ambient property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        ambient = actorObj.actor.GetProperty().GetAmbient()
        
        return ambient
    
    def getActorSpecular(self, actorName):
        """Get specular property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        specular = actorObj.actor.GetProperty().GetSpecular()
        
        return specular
    
    def getActorSpecularPower(self, actorName):
        """Get specular power property on actor."""
        actorObj = self._renderersDict[actorName].getActor()
        specularPower = actorObj.actor.GetProperty().GetSpecularPower()
        
        return specularPower
    
    def addActor(self, actorName, reinit=True):
        """Add individual actor."""
        actorObj = self._renderersDict[actorName].getActor()
        changes = False
        if not actorObj.visible:
            self._logger.debug("Adding actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                    rw.vtkRen.AddActor(actorObj.actor)
                    changes = True
            
            actorObj.visible = True
        
        if changes and reinit:
            self.reinitialiseRendererWindows()
        
        return changes
    
    def hideActor(self, actorName, reinit=True):
        """Remove individual actor."""
        actorObj = self._renderersDict[actorName].getActor()
        changes = False
        if actorObj.visible:
            self._logger.debug("Removing actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                    rw.vtkRen.RemoveActor(actorObj.actor)
                    changes = True
            
            actorObj.visible = False
        
        if changes and reinit:
            self.reinitialiseRendererWindows()
        
        return changes
    
    def reinitialiseRendererWindows(self):
        """Reinit renderer windows."""
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                rw.vtkRenWinInteract.ReInitialize()
    
    def addActors(self):
        """Add all actors."""
        # current renderer windows
        rendererWindows = self._getCurrentRendererWindows()
        
        # actors dict
        actorsDict = self.getActorsDict()
        
        # loop over actors
        for actorName, actorObj in six.iteritems(actorsDict):
            if not actorObj.visible:
                self._logger.debug("Adding actor: '%s'", actorName)
                for rw in rendererWindows:
                    rw.vtkRen.AddActor(actorObj.actor)
                
                actorObj.visible = True
        
        self.addScalarBar()
    
    def renderers(self):
        """Iterate over renderers."""
        for key in self._renderersDict:
            yield self._renderersDict[key]
