
"""
Module for rendering

@author: Chris Scott

"""

import os
import sys

import vtk


################################################################################
def setRes(num):
    #res = 15.84 * (0.99999**natoms)
    #if(LowResVar.get()=="LowResOff"):
    if(num==0):
        res = 100
    else:
        #if(ResVar.get()=="LowResOn"):
        #    
        #    res = -1.0361*math.log(num,e) + 14.051
        #    #res = round(res,0)
        #    #res = 176*(num**-0.36)
        #    res = int(res)
        #    
        #elif(ResVar.get()=="HighResOn"):
        #    
        #    res = -2.91*math.log(num,e) + 35
        #    res = round(res,0)
        #    res = 370*(num**-0.36)
        #    res = int(res)
        #    
        #else:
        
        res = -2.91*math.log(num,2.7) + 35
        res = round(res,0)
        res = 170*(num**-0.36)
        res = int(res)    
    
    print "RES = ",res,num    
    return res


################################################################################
class CellOutline:
    def __init__(self, ren):
        
        self.ren = ren
        self.source = vtk.vtkOutlineSource()
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor = vtk.vtkActor()
        self.visible = 0
    
    def add(self, a, b):
        """
        Add the lattice cell.
        
        """
        # first remove if already visible
        if self.visible:
            self.remove()
        
        # now add it
        self.source.SetBounds(a[0], b[0], a[1], b[1], a[2], b[2])
        
        self.mapper.SetInput(self.source.GetOutput())
        
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetColor(0, 0, 0)
        
        self.ren.AddActor(self.actor)
        
        self.visible = 1
    
    def remove(self):
        """
        Remove the cell outline.
        
        """
        self.ren.RemoveActor(self.actor)
        
        self.visible = 0


################################################################################
class Renderer:
    def __init__(self, mainWindow):
        
        self.mainWindow = mainWindow
        self.ren = self.mainWindow.VTKRen
        self.renWinInteract = self.mainWindow.VTKWidget
        
        # is the interactor initialised
        self.init = 0
        
        # setup stuff
        self.camera = self.ren.GetActiveCamera()
        
        # lattice frame
        self.latticeFrame = CellOutline(self.ren)
        
        
    def reinit(self):
        """
        Reinitialise.
        
        """
        if self.init:
            self.renWinInteract.ReInitialize()
        else:
            self.renWinInteract.Initialize()
            self.init = 1
    
    def postRefRender(self):
        """
        Render post read reference file.
        
        """
        dims = self.mainWindow.refState.cellDims
        
        # add lattice frame
        self.latticeFrame.add([0, 0, 0], dims)
        
        # set camera to cell
        self.setCameraToCell()
        
        # reinitialise
        self.reinit()
    
    def setCameraToCell(self):
        """
        Point the camera at the centre of the cell.
        
        """
        dims = self.mainWindow.refState.cellDims
        
        # set camera to lattice
        campos = [0]*3
        if dims[1] > dims[2]:
            campos[0] = -3.0 * dims[1]
        else:
            campos[0] = -3.0 * dims[2]
        campos[1] = 0.5 * dims[1]
        campos[2] = 0.5 * dims[2]
        
        focpnt = [0]*3
        focpnt[0] = 0.5 * dims[0]
        focpnt[1] = 0.5 * dims[1]
        focpnt[2] = 0.5 * dims[2]
        
        self.camera.SetFocalPoint(focpnt)
        self.camera.SetPosition(campos)
    
    def setCameraToCOM(self):
        """
        Point the camera at the centre of mass.
        
        """
        pass
    
    def writeCameraSettings(self):
        """
        Write the camera settings to file.
        So can be loaded back in future
        OPTION TO WRITE TO TMPDIR IF WANT!!!
        
        """
        pass
        
    def addAxes(self):
        """
        Add the axis label
        
        """
        pass
    
    def removeAxes(self):
        """
        Remove the axis label
        
        """
        pass
    
    def removeAllActors(self):
        """
        Remove all actors
        
        """
        filterLists = self.getFilterLists()
        
        for filterList in filterLists:
            filterList.filterer.removeActors()

    def removeActor(self, actor):
        """
        Remove actor
        
        """
        self.ren.RemoveActor(actor)
        
    def removeActorList(self, actorList):
        """
        Remove list of actors
        
        """
        pass
    
    def getFilterLists(self):
        """
        Return filter lists
        
        """
        return self.mainWindow.mainToolbar.filterPage.filterLists
    
    def render(self):
        """
        Render.
        
        """
        print "RENDERING"
        self.removeAllActors()
        
        filterLists = self.getFilterLists()
        count = 0
        for filterList in filterLists:
            print "RENDERING LIST", count
            count += 1
            
            filterList.addActors()


################################################################################
def setupLUT(specieList, specieRGB):
    """
    Setup the colour look up table
    
    """
    NSpecies = len(specieList)
    
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfColors(NSpecies)
    lut.SetNumberOfTableValues(NSpecies)
    lut.SetTableRange(0, NSpecies - 1)
    lut.SetRange(0, NSpecies - 1)
    
    for i in xrange(NSpecies):
        lut.SetTableValue(i, specieRGB[i][0], specieRGB[i][1], specieRGB[i][2], 1.0)
        

################################################################################
def getActorsForFilteredSystem(visibleAtoms, mainWindow):
    """
    Make the actors for the filtered system
    
    """
    actorsList = []
    
    # resolution
    res = setRes(len(visibleAtoms))
    
    # render the atoms
    
    
    
    
    
    
    
    return actorsList

