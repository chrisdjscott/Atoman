
"""
Renderer for the FilterList object.

@author: Chris Scott

"""







class FilterListRenderer(object):
    """
    Renderer for a filter list.
    
    """
    def __init__(self):
        # dictionaries for storing current actors
        self.actorsDict = {}
        self.traceDict = {}
        self.previousPosForTrace = None
        
        
        
    
    def render(self, filterer):
        """
        Render the data provided by the Filterer
        
        """
    
    def removeActors(self, sequencer=False):
        """
        Remove actors.
        
        """
        self.hideActors()
        
        self.actorsDict = {}
        if not sequencer:
            self.traceDict = {}
            self.previousPosForTrace = None
        
        self.scalarsDict = {}
        self.latticeScalarsDict = {}
        self.vectorsDict = {}
        self.scalarBar_white_bg = None
        self.scalarBar_black_bg = None
        
        self.NVis = 0
        self.NVac = 0
        self.NInt = 0
        self.NAnt = 0
        self.visibleAtoms = np.asarray([], dtype=np.int32)
        self.interstitials = np.asarray([], dtype=np.int32)
        self.vacancies = np.asarray([], dtype=np.int32)
        self.antisites = np.asarray([], dtype=np.int32)
        self.onAntisites = np.asarray([], dtype=np.int32)
        self.splitInterstitials = np.asarray([], dtype=np.int32)
        self.visibleSpecieCount = np.asarray([], dtype=np.int32)
        self.vacancySpecieCount = np.asarray([], dtype=np.int32)
        self.interstitialSpecieCount = np.asarray([], dtype=np.int32)
        self.antisiteSpecieCount = np.asarray([], dtype=np.int32)
        self.splitIntSpecieCount = np.asarray([], dtype=np.int32)
        self.driftVector = np.zeros(3, np.float64)
        
        self.povrayAtomsWritten = False
        self.clusterList = []
        self.bubbleList = []
        self.structureCounterDicts = {}
        self.voronoi = None
    
    def hideActors(self):
        """
        Hide all actors
        
        """
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self.logger.debug("Removing actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if actorObj.visible:
                        self.logger.debug("  Removing actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                                rw.vtkRen.RemoveActor(actorObj.actor)
                        
                        actorObj.visible = False
            
            else:
                actorObj = val
                if actorObj.visible:
                    self.logger.debug("Removing actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                            rw.vtkRen.RemoveActor(actorObj.actor)
                    
                    actorObj.visible = False
        
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
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
            self.logger.debug("Adding actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
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
            self.logger.debug("Removing actor: '%s'", actorName)
            for rw in self.rendererWindows:
                if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
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
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
    
    def addActors(self):
        """
        Add all actors
        
        """
        for actorName, val in self.actorsDict.iteritems():
            if isinstance(val, dict):
                self.logger.debug("Adding actors for: '%s'", actorName)
                for actorName2, actorObj in val.iteritems():
                    if not actorObj.visible:
                        self.logger.debug("  Adding actor: '%s'", actorName2)
                        for rw in self.rendererWindows:
                            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                                rw.vtkRen.AddActor(actorObj.actor)
                        
                        actorObj.visible = True
            
            else:
                actorObj = val
                if not actorObj.visible:
                    self.logger.debug("Adding actor: '%s'", actorName)
                    for rw in self.rendererWindows:
                        if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                            rw.vtkRen.AddActor(actorObj.actor)
                    
                    actorObj.visible = True
        
        for rw in self.rendererWindows:
            if rw.currentPipelineString == self.mainToolbar.currentPipelineString:
                rw.vtkRenWinInteract.ReInitialize()
        
        self.addScalarBar()
