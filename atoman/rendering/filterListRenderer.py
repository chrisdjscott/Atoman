
"""
Renderer for the FilterList object.

@author: Chris Scott

"""
import logging

import numpy as np
import vtk
import vtk.util

from . import _rendering
from . import utils
from .renderers import atomRenderer


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
        atomPoints.SetData(vtk.util.numpy_support.numpy_to_vtk(atomPointsNp, deep=1))
        
        # make radius array
        radiusArrayNp = _rendering.makeVisibleRadiusArray(visibleAtoms, inputState.specie, inputState.specieCovalentRadius)
        radiusArray = vtk.util.numpy_support.numpy_to_vtk(radiusArrayNp, deep=1)
        radiusArray.SetName("radius")
        
        # get the scalars array
        scalarsArrayNp, scalarsArray = self._getScalarsArray()
        
        # make the look up table
        lut = utils.setupLUT(inputState.specieList, inputState.specieRGB, self.colouringOptions)
        
        # render atoms
        #TODO: should we store the renderer!?
        atomRend = atomRenderer.AtomRenderer()
        actor = atomRend.render(atomPoints, scalarsArray, radiusArray, len(inputState.specieList), self.colouringOptions, 
                                self.displayOptions.atomScaleFactor, lut, resolution)
        self.actorsDict["Atoms"] = actor
        
        
        
        
        
        
        
    
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
        
        scalarsArrayVTK = vtk.util.numpy_support.numpy_to_vtk(scalarsArray, deep=1)
        scalarsArrayVTK.SetName("colours")
        
        return scalarsArray, scalarsArrayVTK
    
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
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self._logger.debug("Removing actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if actorObj.visible:
                        self._logger.debug("  Removing actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                                rw.vtkRen.RemoveActor(actorObj.actor)
                        
                        actorObj.visible = False
            
            else:
                actorObj = val
                if actorObj.visible:
                    self._logger.debug("Removing actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                            rw.vtkRen.RemoveActor(actorObj.actor)
                    
                    actorObj.visible = False
        
        for rw in self.rendererWindows:
            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                rw.vtkRenWinInteract.ReInitialize()
        
        self.hideScalarBar()
    
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
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self._logger.debug("Adding actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if not actorObj.visible:
                        self._logger.debug("  Adding actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                                rw.vtkRen.AddActor(actorObj.actor)
                        
                        actorObj.visible = True
            
            else:
                actorObj = val
                if not actorObj.visible:
                    self._logger.debug("Adding actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineIndex == self.pipelinePage.pipelineIndex:
                            rw.vtkRen.AddActor(actorObj.actor)
                    
                    actorObj.visible = True
        
        # self.addScalarBar()
