
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
        numForRes = len(visibleAtoms) #TODO: plus interstitials etc...
        resolution = utils.setRes(numForRes, self.displayOptions)
        
        # make points array
        atomPointsNp = _rendering.makeVisiblePointsArray(visibleAtoms, inputState.pos)
        atomPoints = vtk.vtkPoints()
        atomPoints.SetData(numpy_support.numpy_to_vtk(atomPointsNp, deep=1))
        
        # make radius array
        radiusArrayNp = _rendering.makeVisibleRadiusArray(visibleAtoms, inputState.specie, inputState.specieCovalentRadius)
        radiusArray = numpy_support.numpy_to_vtk(radiusArrayNp, deep=1)
        radiusArray.SetName("radius")
        
        # get the scalars array
        scalarsArrayNp, scalarsArray = self._getScalarsArray()
        
        # make the look up table
        lut = utils.setupLUT(inputState.specieList, inputState.specieRGB, self.colouringOptions)
        
        # render atoms
        self._renderAtoms(atomPoints, scalarsArray, radiusArray, lut, resolution)
        
        # render defects
        
        
        # render vectors
        self._renderVectors(lut)
            
        # displacement vectors
        
        
        # trace
        
        
        # voronoi
        
        
        # bonds
        self._renderBonds(scalarsArrayNp, lut)
        
        # render clusters
        
        
        # render bubbles (or already done?)
        
        
        
        
        
    
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
    
    def _getScalarsArray(self):
        """
        Returns the scalars array (np and vtk versions)
        
        """
        # local refs
        inputState = self._filterer.inputState
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
                scalarsFull = np.asarray(inputState.specie, dtype=np.float64)
            elif scalarType == 1:
                scalarsFull = inputState.pos[colouringOptions.heightAxis::3]
            elif scalarType == 4:
                scalarsFull = inputState.charge
            else:
                logger.error("Unrecognised scalar type (%d): defaulting to specie", scalarType)
                scalarsFull = inputState.specie
            scalarsArray = _rendering.makeVisibleScalarArray(self._filterer.visibleAtoms, scalarsFull)
        
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
