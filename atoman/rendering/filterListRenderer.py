
"""
Renderer for the FilterList object.

@author: Chris Scott

"""
import logging

import numpy as np
import vtk
from vtk.util import numpy_support
from PySide import QtCore

from . import _rendering
from . import utils
from ..algebra import vectors as vectorslib
from .renderers import atomRenderer
from .renderers import vectorRenderer
from .renderers import bondRenderer
from .renderers import vacancyRenderer
from ..system.atoms import elements


################################################################################

class FilterListRenderer(object):
    """
    Renderer for a filter list.
    
    """
    def __init__(self, filterList):
        # logger
        self._logger = logging.getLogger(__name__)
        
        # the filterer that we are rendering
        self._filterer = filterList.filterer
        
        # temporary directory
        self._tmpDirectory = filterList.mainWindow.tmpDirectory
        
        # dictionaries for storing current actors
        self.actorsDict = {}
        self.traceDict = {}
        self.previousPosForTrace = None
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        self.povrayAtomsWritten = False
        
        # get required refs from filter list
        self.pipelinePage = filterList.pipelinePage
        self.rendererWindows = filterList.pipelinePage.rendererWindows
        self.colouringOptions = filterList.colouringOptions
        self.displayOptions = filterList.displayOptions
        self.vectorsOptions = filterList.vectorsOptions
        self.bondsOptions = filterList.bondsOptions
        self.actorsOptions = filterList.actorsOptions
    
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
        
        ** Do we need to store lots of refs, eg lattices, scalarsDicts, etc
        ** Should filterer be passed to __init__
        
        """
        self._logger.debug("Rendering filter list")
        
        # local refs
        inputState = self._filterer.inputState
        refState = self._filterer.refState
        visibleAtoms = self._filterer.visibleAtoms
        
        # set resolution
        numForRes = len(visibleAtoms) + len(self._filterer.interstitials) + len(self._filterer.onAntisites)
        resolution = utils.setRes(numForRes, self.displayOptions)
        self._logger.debug("Setting resolution to %d (num %d)", resolution, numForRes)
        
        # make points array
        atomPointsNp = _rendering.makeVisiblePointsArray(visibleAtoms, inputState.pos)
        atomPoints = vtk.vtkPoints()
        atomPoints.SetData(numpy_support.numpy_to_vtk(atomPointsNp, deep=1))
        
        # make radius array
        radiusArrayNp = _rendering.makeVisibleRadiusArray(visibleAtoms, inputState.specie, inputState.specieCovalentRadius)
        radiusArray = numpy_support.numpy_to_vtk(radiusArrayNp, deep=1)
        radiusArray.SetName("radius")
        
        # get the scalars array
        scalarsArrayNp, scalarsArray = self._getScalarsArray(inputState, visibleAtoms)
        
        # make the look up table
        lut = utils.setupLUT(inputState.specieList, inputState.specieRGB, self.colouringOptions)
        
        # render atoms
        self._renderAtoms(atomPoints, scalarsArray, radiusArray, lut, resolution)
        
        # render defects
        self._renderDefects(lut, resolution)
        
        # render vectors
        self._renderVectors(lut)
            
        # displacement vectors
        
        
        # trace
        
        
        # voronoi
        
        
        # bonds
        self._renderBonds(scalarsArrayNp, lut)
        
        # render clusters
        
        
        # render bubbles (or already done?)
        
        
        
        # refresh actors options
        self.actorsOptions.refresh(self.actorsDict)
    
    def _renderDefects(self, lut, resolution):
        """Render defects."""
        self._logger.debug("Rendering defects")
        
        # local refs
        interstitials = self._filterer.interstitials
        vacancies = self._filterer.vacancies
        antisites = self._filterer.antisites
        onAntisites = self._filterer.onAntisites
        splitInterstitials = self._filterer.splitInterstitials
        refState = self._filterer.refState
        
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
        print "SPLIT VACS", len(splitVacs), splitVacs
        self._renderVacancies(lut, splitVacs, actorName="Split interestitial vacancies")
        
        # render vacancies
        print "VACS", len(vacancies), vacancies
        self._renderVacancies(lut, vacancies)
        
        # render antisite frames
        
    
    def _renderVacancies(self, lut, vacancies, actorName="Vacancies"):
        """Render vacancies."""
        # local refs
        refState = self._filterer.refState
        
        if not len(vacancies):
            return
        self._logger.debug("Rendering %s", actorName)
        
        # points
        pointsNp = _rendering.makeVisiblePointsArray(vacancies, refState.pos)
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(pointsNp, deep=1))
        
        # radii
        radiusNp = _rendering.makeVisibleRadiusArray(vacancies, refState.specie, refState.specieCovalentRadius)
        radius = numpy_support.numpy_to_vtk(radiusNp, deep=1)
        radius.SetName("radius")
        
        # scalars
        scalarsNp, scalars = self._getScalarsArray(refState, vacancies)
        
        # get vacancy scale setting
        found = False
        for name, settings in zip(self._filterer.currentFilters, self._filterer.currentSettings):
            if name == "Point defects":
                found = True
                break
        if not found:
            raise RuntimeError("Could not find point defects filter settings")
        
        # render
        rend = vacancyRenderer.VacancyRenderer()
        actor = rend.render(points, scalars, radius, len(refState.specieList), self.colouringOptions,
                            self.displayOptions.atomScaleFactor, lut, settings)
        self.actorsDict[actorName] = actor
    
    def _renderDefectAtoms(self, lut, resolution, atomList, actorName):
        """Render defect atoms, eg interstitials."""
        if not len(atomList):
            return
        self._logger.debug("Rendering %s", actorName)
        
        # local refs
        inputState = self._filterer.inputState
        
        # points
        pointsNp = _rendering.makeVisiblePointsArray(atomList, inputState.pos)
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(pointsNp, deep=1))
        
        # radii
        radiusNp = _rendering.makeVisibleRadiusArray(atomList, inputState.specie, inputState.specieCovalentRadius)
        radius = numpy_support.numpy_to_vtk(radiusNp, deep=1)
        radius.SetName("radius")
        
        # scalars
        scalarsNp, scalars = self._getScalarsArray(inputState, atomList)
        
        # render
        rend = atomRenderer.AtomRenderer()
        actor = rend.render(points, scalars, radius, len(inputState.specieList), self.colouringOptions, 
                            self.displayOptions.atomScaleFactor, lut, resolution)
        self.actorsDict[actorName] = actor
    
    def _renderBonds(self, scalarsArray, lut):
        """Calculate and render bonds between atoms."""
        # check if we need to calculate
        if not self.bondsOptions.drawBonds:
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
        for i in xrange(self.bondsOptions.bondsList.count()):
            item = self.bondsOptions.bondsList.item(i)
            
            if item.checkState() == QtCore.Qt.Checked:
                syma = item.syma
                symb = item.symb
                
                # check if in current specie list and if so what indexes
                if syma in specieList:
                    indexa = inputState.getSpecieIndex(syma)
                else:
                    continue
                
                if symb in specieList:
                    indexb = inputState.getSpecieIndex(symb)
                else:
                    continue
                
                if syma in bondDict:
                    d = bondDict[syma]
                    
                    if symb in d:
                        bondMin, bondMax = d[symb]
                        bondMinArray[indexa][indexb] = bondMin
                        bondMinArray[indexb][indexa] = bondMin
                        bondMaxArray[indexa][indexb] = bondMax
                        bondMaxArray[indexb][indexa] = bondMax
                        if bondMax > maxBond:
                            maxBond = bondMax
                        if bondMax > 0:
                            calcBonds = True
                        drawList.append("%s-%s" % (syma, symb))
                        self._logger.info("    Pair: %s - %s; bond range: %f -> %f", syma, symb, bondMin, bondMax)
        
        assert bondMaxArray.max() == bondMax
        
        if not calcBonds:
            self._logger.info("No bonds to calculate")
            return
        
        # calculate bonds
        bonds = bondRenderer.BondCalculator()
        result = bonds.calculateBonds(inputState, visibleAtoms, bondMinArray, bondMaxArray, drawList)
        NBondsTotal, bondArray, NBondsArray, bondVectorArray, bondSpecieCounter = result
        if NBondsTotal == 0:
            self._logger.info("No bonds to render")
            return
        
        # draw bonds
        bondRend = bondRenderer.BondRenderer()
        actor = bondRend.render(inputState, visibleAtoms, NBondsArray, bondArray, bondVectorArray, scalarsArray,
                                self.colouringOptions, self.bondsOptions, lut)
        self.actorsDict["Bonds"] = actor
    
    def _renderClusters(self):
        """Render clusters."""
        
        
        
    
    def _renderAtoms(self, atomPoints, scalarsArray, radiusArray, lut, resolution):
        """Render atoms."""
        if not len(self._filterer.visibleAtoms):
            return
        self._logger.debug("Rendering atoms")
        
        inputState = self._filterer.inputState
        #TODO: should we store the renderer!?
        atomRend = atomRenderer.AtomRenderer()
        actor = atomRend.render(atomPoints, scalarsArray, radiusArray, len(inputState.specieList), self.colouringOptions, 
                                self.displayOptions.atomScaleFactor, lut, resolution)
        self.actorsDict["Atoms"] = actor
    
    def _renderVectors(self, lut):
        """Render vectors."""
        vectorsName = self.vectorsOptions.selectedVectorsName
        vectorsDict = self._filterer.vectorsDict
        if vectorsName is not None and vectorsName not in vectorsDict:
            self._logger.warning("Skipping adding vectors because could not find array: '%s'", vectorsName)
        
        elif vectorsName is not None:
            self._logger.debug("Rendering arrows for vector data: '%s'", vectorsName)
            
            # make the VTK vectors array
            vectorsNp = vectorsDict[vectorsName]
            if self.vectorsOptions.vectorNormalise:
                vectorsNp = vectorslib.normalise(vectorsNp)
            vectors = numpy_support.numpy_to_vtk(vectorsNp, deep=1)
            vectors.SetName("vectors")
            
            # render vectors
            vectorRend = vectorRenderer.VectorRenderer()
            actor = vectorRend.render(atomPoints, scalarsArray, vectors, len(inputState.specieList), self.colouringOptions,
                                      self.vectorsOptions, lut)
            self.actorsDict["Vectors"] = actor
    
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
                logger.error("Unrecognised scalar type (%d): defaulting to specie", scalarType)
                scalarsFull = lattice.specie
            scalarsArray = _rendering.makeVisibleScalarArray(atomList, scalarsFull)
        
        scalarsArrayVTK = numpy_support.numpy_to_vtk(scalarsArray, deep=1)
        scalarsArrayVTK.SetName("colours")
        
        return scalarsArray, scalarsArrayVTK
    
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
        
        self.actorsDict = {}
        if not sequencer:
            self.traceDict = {}
            self.previousPosForTrace = None
        
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        self.povrayAtomsWritten = False
    
    def hideActors(self):
        """
        Hide all actors
        
        """
        rendererWindows = self._getCurrentRendererWindows()
        
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self._logger.debug("Removing actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if actorObj.visible:
                        self._logger.debug("  Removing actor: '%s'", actorName2)
                        for rw in rendererWindows:
                            rw.vtkRen.RemoveActor(actorObj.actor)
                        
                        actorObj.visible = False
            
            else:
                actorObj = val
                if actorObj.visible:
                    self._logger.debug("Removing actor: '%s'", actorName)
                    for rw in rendererWindows:
                        rw.vtkRen.RemoveActor(actorObj.actor)
                    
                    actorObj.visible = False
        
        for rw in rendererWindows:
            rw.vtkRenWinInteract.ReInitialize()
        
        # self.hideScalarBar()
    
    def setActorAmbient(self, actorName, parentName, ambient, reinit=True):
        """
        Set ambient property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetAmbient(ambient)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecular(self, actorName, parentName, specular, reinit=True):
        """
        Set specular property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetSpecular(specular)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def setActorSpecularPower(self, actorName, parentName, specularPower, reinit=True):
        """
        Set specular power property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        actorObj.actor.GetProperty().SetSpecularPower(specularPower)
        
        if reinit:
            self.reinitialiseRendererWindows()
    
    def getActorAmbient(self, actorName, parentName):
        """
        Get ambient property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        ambient = actorObj.actor.GetProperty().GetAmbient()
        
        return ambient
    
    def getActorSpecular(self, actorName, parentName):
        """
        Get specular property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        specular = actorObj.actor.GetProperty().GetSpecular()
        
        return specular
    
    def getActorSpecularPower(self, actorName, parentName):
        """
        Get specular power property on actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
        specularPower = actorObj.actor.GetProperty().GetSpecularPower()
        
        return specularPower
    
    def addActor(self, actorName, parentName=None, reinit=True):
        """
        Add individual actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
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
    
    def hideActor(self, actorName, parentName=None, reinit=True):
        """
        Remove individual actor
        
        """
        if parentName is not None:
            d = self.actorsDict[parentName]
        else:
            d = self.actorsDict
        
        actorObj = d[actorName]
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
        """
        Reinit renderer windows
        
        """
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                rw.vtkRenWinInteract.ReInitialize()
    
    def addActors(self):
        """
        Add all actors
        
        """
        rendererWindows = self._getCurrentRendererWindows()
        
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self._logger.debug("Adding actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if not actorObj.visible:
                        self._logger.debug("  Adding actor: '%s'", actorName2)
                        for rw in rendererWindows:
                            rw.vtkRen.AddActor(actorObj.actor)
                        
                        actorObj.visible = True
            
            else:
                actorObj = val
                if not actorObj.visible:
                    self._logger.debug("Adding actor: '%s'", actorName)
                    for rw in rendererWindows:
                        rw.vtkRen.AddActor(actorObj.actor)
                    
                    actorObj.visible = True
        
        # self.addScalarBar()
