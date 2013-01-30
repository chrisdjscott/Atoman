
"""
Module for rendering

@author: Chris Scott

"""
import os
import math
import copy

import numpy as np
import vtk
from PIL import Image

from ..visclibs import output_c
from ..visutils import utilities


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
    
#    print "RES = ",res,num    
    return res

################################################################################

class vtkRenderWindowText(vtk.vtkTextActor):
    """
    On screen information text.
    
    @author: Marc Robinson
    
    """
    def __init__(self, inputtext, Size, x, y, r, g, b):
        self.Input =  inputtext
        self.Size =  Size
        self.x =  x
        self.y =  y    
        self.SetDisplayPosition(self.x, self.y)
        self.SetInput(self.Input)
        #textActor.UseBorderAlignOn
        self.GetPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        #textActor.GetPosition2Coordinate().SetValue(0.6, 0.4)
        tprop = self.GetTextProperty()
        tprop.SetFontSize(self.Size)
        tprop.SetFontFamilyToArial()
        tprop.SetJustificationToLeft()
        tprop.SetVerticalJustificationToTop()
        tprop.BoldOn()
        #tprop.ItalicOn()
        #tprop.ShadowOn()
        #tprop.SetShadowOffset(2,2)
        tprop.SetColor(r,g,b)
    
    def change_input(self,inputtext1):
        self.Input = inputtext1
        self.SetInput(self.Input)
    
    def change_pos(self,x,y):
        self.x =  x
        self.y =  y    
        self.SetDisplayPosition(self.x, self.y)


################################################################################
class SlicePlane(object):
    """
    Slice plane.
    
    """
    def __init__(self, ren, renWinInteract, mainWindow):
        self.ren = ren
        self.renWinInteract = renWinInteract
        self.actor = vtk.vtkActor()
        self.source = vtk.vtkPlaneSource()
        self.mapper = vtk.vtkPolyDataMapper()
        
        self.mainWindow = mainWindow
    
        self.visible = False
    
    def show(self, p, n):
        """
        Show the slice plane in given position.
        
        """
        if self.visible:
            self.ren.RemoveActor(self.actor)
            self.visible = False
        
        inputState = self.mainWindow.inputState
        
        # source
        self.source.SetOrigin(-50, -50, 0)
        self.source.SetPoint1(inputState.cellDims[0] + 50, -50, 0)
        self.source.SetPoint2(-50, inputState.cellDims[1] + 50, 0)
        self.source.SetNormal(n)
        self.source.SetCenter(p)
        self.source.SetXResolution(100)
        self.source.SetYResolution(100)
        
        # mapper
        self.mapper.SetInputConnection(self.source.GetOutputPort())
        
        # actor
        self.actor.SetMapper(self.mapper)
        self.actor.GetProperty().SetDiffuseColor(1, 0, 0)
        self.actor.GetProperty().SetSpecular(0.4)
        self.actor.GetProperty().SetSpecularPower(10)
        self.actor.GetProperty().SetOpacity(0.7)
        self.actor.GetProperty().SetLineWidth(2.0)
        self.actor.GetProperty().EdgeVisibilityOn()
        
        # add to ren
        self.ren.AddActor(self.actor)
        self.renWinInteract.ReInitialize()
        
        self.visible = True
    
    def hide(self):
        """
        Remove the actor.
        
        """
        if self.visible:
            self.ren.RemoveActor(self.actor)
            self.renWinInteract.ReInitialize()
            self.visible = False
        

################################################################################
class CellOutline(object):
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
class AxesBasic(vtk.vtkActorCollection):
    """
    @author: Marc Robinson
    
    Modified slightly
    
    """
    def __init__(self, ren, renWinInteract):
        self.Edgesx = vtk.vtkActor()
        self.Edgesy = vtk.vtkActor()
        self.Edgesz = vtk.vtkActor()
        self.conexActor = vtk.vtkActor()
        self.coneyActor = vtk.vtkActor()
        self.coneyActor.RotateZ(90)
        self.conezActor = vtk.vtkActor()
        self.conezActor.RotateY(270)
        self.xlabelActor = vtk.vtkFollower()
        self.xlabelActor.SetScale(3, 3, 3)
        self.ylabelActor = vtk.vtkFollower()
        self.ylabelActor.SetScale(3, 3, 3)
        self.zlabelActor = vtk.vtkFollower()
        self.zlabelActor.SetScale(3, 3, 3)
        self.AddItem(self.Edgesx)
        self.AddItem(self.Edgesy)
        self.AddItem(self.Edgesz)
        self.AddItem(self.conexActor)
        self.AddItem(self.coneyActor)
        self.AddItem(self.conezActor)
        self.AddItem(self.xlabelActor)
        self.AddItem(self.ylabelActor)
        self.AddItem(self.zlabelActor)
        
        self.ren = ren
        self.renWinInteract = renWinInteract
        
        self.visible = 1
    
    def remove(self):
        
        self.InitTraversal()
        tmpactor = self.GetNextItem()
        while tmpactor is not None:
            try:
                self.ren.RemoveActor(tmpactor)
            except:
                pass
            
            tmpactor = self.GetNextItem()
            
        self.renWinInteract.ReInitialize()
        
        self.visible = 0
    
    def add(self):
        
        self.InitTraversal()
        tmpactor = self.GetNextItem()
        while tmpactor is not None:
            try:
                self.ren.AddActor(tmpactor)
            except:
                pass
            
            tmpactor = self.GetNextItem()
            
        self.renWinInteract.ReInitialize()
        
        self.visible = 1
    
    def refresh(self,x0,y0,z0,xl,yl,zl,xtext,ytext,ztext):
        
        self.remove()
        
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.xl = xl
        self.yl = yl
        self.zl = zl
        self.xtext = xtext
        self.ytext = ytext
        self.ztext = ztext
        
        linex = vtk.vtkLineSource()
        linex.SetPoint1(x0,y0,z0)
        linex.SetPoint2(x0+xl,y0,z0)    
        Tubesx = vtk.vtkTubeFilter()
        Tubesx.SetInputConnection(linex.GetOutputPort())
        Tubesx.SetRadius(0.5)
        Tubesx.SetNumberOfSides(5)
        Tubesx.UseDefaultNormalOn()
        Tubesx.SetDefaultNormal(.577, .577, .577)
        TubeMapperx = vtk.vtkPolyDataMapper()
        TubeMapperx.SetInputConnection(Tubesx.GetOutputPort())
        self.Edgesx.SetMapper(TubeMapperx)
        self.Edgesx.GetProperty().SetDiffuseColor(1,0,0)
        self.Edgesx.GetProperty().SetSpecular(.4)
        self.Edgesx.GetProperty().SetSpecularPower(10)
        self.Edgesx.GetProperty().SetLineWidth(2.0)
        
        liney = vtk.vtkLineSource()
        liney.SetPoint1(x0,y0,z0)
        liney.SetPoint2(x0,y0+yl,z0)    
        Tubesy = vtk.vtkTubeFilter()
        Tubesy.SetInputConnection(liney.GetOutputPort())
        Tubesy.SetRadius(0.5)
        Tubesy.SetNumberOfSides(5)
        Tubesy.UseDefaultNormalOn()
        Tubesy.SetDefaultNormal(.577, .577, .577)
        TubeMappery = vtk.vtkPolyDataMapper()
        TubeMappery.SetInputConnection(Tubesy.GetOutputPort())
        
        self.Edgesy.SetMapper(TubeMappery)
        self.Edgesy.GetProperty().SetDiffuseColor(0,1,0)
        self.Edgesy.GetProperty().SetSpecular(.4)
        self.Edgesy.GetProperty().SetSpecularPower(10)
        self.Edgesy.GetProperty().SetLineWidth(2.0)
        #self.AddItem(self.Edgesy)  
        linez = vtk.vtkLineSource()
        linez.SetPoint1(x0,y0,z0)
        linez.SetPoint2(x0,y0,z0+zl)    
        Tubesz = vtk.vtkTubeFilter()
        Tubesz.SetInputConnection(linez.GetOutputPort())
        Tubesz.SetRadius(0.5)
        Tubesz.SetNumberOfSides(5)
        Tubesz.UseDefaultNormalOn()
        Tubesz.SetDefaultNormal(.577, .577, .577)
        TubeMapperz = vtk.vtkPolyDataMapper()
        TubeMapperz.SetInputConnection(Tubesz.GetOutputPort())
        
        self.Edgesz.SetMapper(TubeMapperz)
        self.Edgesz.GetProperty().SetDiffuseColor(0,0,1)
        self.Edgesz.GetProperty().SetSpecular(.4)
        self.Edgesz.GetProperty().SetSpecularPower(10)
        self.Edgesz.GetProperty().SetLineWidth(2.0)
        #self.AddItem(self.Edgesz)
        
        conex = vtk.vtkConeSource()
        conex.SetRadius(1.2)
        conex.SetHeight(2.0)
        conex.SetResolution(50)
        #conex.SetThetaResolution(10)
        conex.SetCenter(x0+xl+1.0,y0,z0)
    
    
        conexMapper = vtk.vtkPolyDataMapper()
        conexMapper.SetInput(conex.GetOutput())
        
        self.conexActor.SetMapper(conexMapper)
        self.conexActor.GetProperty().SetDiffuseColor(1,0,0)
        #self.AddItem(self.conexActor)
        
        coney = vtk.vtkConeSource()
        coney.SetRadius(1.2)
        coney.SetHeight(2.0)
        coney.SetResolution(50)
        coney.SetCenter(x0,y0+yl+1.0,z0)
    
        
        coneyMapper = vtk.vtkPolyDataMapper()
        coneyMapper.SetInput(coney.GetOutput())
        self.coneyActor.SetMapper(coneyMapper)
        self.coneyActor.GetProperty().SetDiffuseColor(0,1,0)
        self.coneyActor.SetOrigin(x0,y0+yl+1.0,z0)
        
        #self.AddItem(self.coneyActor)
        
        conez = vtk.vtkConeSource()
        conez.SetRadius(1.2)
        conez.SetHeight(2.0)
        conez.SetResolution(50)
        conez.SetCenter(x0,y0,z0+zl+1.0)
        conezMapper = vtk.vtkPolyDataMapper()
        conezMapper.SetInput(conez.GetOutput())
        self.conezActor.SetMapper(conezMapper)
        self.conezActor.GetProperty().SetDiffuseColor(0,0,1)
        self.conezActor.SetOrigin(x0,y0,z0+zl+1.0)
        
        
        #self.AddItem(self.conezActor)
            
        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.xtext)
                
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.xlabelActor.SetMapper(labelMapper)
        
        self.xlabelActor.SetPosition(x0+xl+5.0,y0-0.5,z0)
        self.xlabelActor.GetProperty().SetDiffuseColor(1,0,0)
        self.xlabelActor.SetCamera(self.ren.GetActiveCamera())
        
        
        #self.AddItem(self.xlabelActor)
        
        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.ytext)
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.ylabelActor.SetMapper(labelMapper)
        #self.ylabelActor.SetScale(3, 3, 3)
    
        self.ylabelActor.SetPosition(x0,y0+yl+4.0,z0)
        self.ylabelActor.GetProperty().SetDiffuseColor(0,1,0)
        self.ylabelActor.SetCamera(self.ren.GetActiveCamera())
        #self.AddItem(self.ylabelActor)


        caseLabel = vtk.vtkVectorText()
        caseLabel.SetText(self.ztext)
        labelMapper = vtk.vtkPolyDataMapper()
        labelMapper.SetInputConnection(caseLabel.GetOutputPort())
        
        self.zlabelActor.SetMapper(labelMapper)
        #self.zlabelActor.SetScale(3, 3, 3)
        self.zlabelActor.SetPosition(x0,y0,z0+zl+3.0)
        self.zlabelActor.GetProperty().SetDiffuseColor(0,0,1)
        self.zlabelActor.GetProperty().SetEdgeColor(0,0,0)
        self.zlabelActor.GetProperty().EdgeVisibilityOn()
        self.zlabelActor.GetProperty().SetLineWidth(2)
        self.zlabelActor.SetCamera(self.ren.GetActiveCamera())
        #self.AddItem(self.zlabelActor)
        
        self.add()

################################################################################
class Axes(object):
    def __init__(self, ren, renWinInteract):
        
        self.ren = ren
        self.renWinInteract = renWinInteract
        
        self.actor = vtk.vtkAxesActor()
        self.actor.SetTipTypeToCone()
        self.actor.SetShaftTypeToCylinder()
#        self.actor.GetXAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetXAxisCaptionActor2D().SetHeight(0.1)
#        self.actor.GetYAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetYAxisCaptionActor2D().SetHeight(0.1)
#        self.actor.GetZAxisCaptionActor2D().SetWidth(0.1)
#        self.actor.GetZAxisCaptionActor2D().SetHeight(0.1)
        self.actor.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.actor.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        self.actor.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        
        transform = vtk.vtkTransform()
        transform.Translate(-10.0, -10.0, -10.0)
        self.actor.SetUserTransform(transform)
        
        self.actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0)
        self.actor.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        self.actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,1,0)
        self.actor.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        self.actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,1)
        self.actor.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontSize(20)
        
        self.orientationWidget = vtk.vtkOrientationMarkerWidget()
        self.orientationWidget.SetOrientationMarker(self.actor)
#        self.orientationWidget.SetOutlineColor(0.93, 0.57, 0.13)
        self.orientationWidget.SetInteractor(self.renWinInteract)
#        self.orientationWidget.SetViewport(0.0, 0.0, 0.25, 0.25)
        self.orientationWidget.SetEnabled(1)
        self.orientationWidget.InteractiveOff()
        
        self.visible = 0
    
    def add(self, cellDims):
        """
        Add the axes.
        
        """
        self.actor.SetTotalLength(0.2 * cellDims[0], 0.2 * cellDims[1], 0.2 * cellDims[2])
        
        self.ren.AddActor(self.actor)
        self.visible = 1
        
    def remove(self):
        """
        Remove the axes actor.
        
        """
        self.ren.RemoveActor(self.actor)
        
        self.visible = 0


################################################################################
class Renderer(object):
    def __init__(self, mainWindow):
        
        self.mainWindow = mainWindow
        self.ren = self.mainWindow.VTKRen
        self.renWinInteract = self.mainWindow.VTKWidget
        
        self.log = self.mainWindow.console.write
        
        # is the interactor initialised
        self.init = 0
        
        # setup stuff
        self.camera = self.ren.GetActiveCamera()
        
        # lattice frame
        self.latticeFrame = CellOutline(self.ren)
        
        # axes
#        self.axes = Axes(self.ren, self.renWinInteract)
        self.axes = AxesBasic(self.ren, self.renWinInteract)
        
        # slice plane
        self.slicePlane = SlicePlane(self.ren, self.renWinInteract, self.mainWindow)
        
    def reinit(self):
        """
        Reinitialise.
        
        """
        if self.init:
            self.renWinInteract.ReInitialize()
        else:
            self.renWinInteract.Initialize()
            self.init = 1
    
    def getRenWin(self):
        """
        Return the render window
        
        """
        return self.renWinInteract.GetRenderWindow()
    
    def postRefRender(self):
        """
        Render post read reference file.
        
        """
        # add the lattice frame
        self.addLatticeFrame()
        
        # add the axes
        self.addAxes()
        
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
        self.camera.SetViewUp(0, 1, 0)
        
        self.reinit()
    
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
    
    def addLatticeFrame(self):
        """
        Add the lattice frame
        
        """
        dims = self.mainWindow.refState.cellDims
        
        # add lattice frame
        self.latticeFrame.add([0, 0, 0], dims)
    
    def removeLatticeFrame(self):
        """
        Remove the lattice frame
        
        """
        self.latticeFrame.remove()
    
    def toggleLatticeFrame(self):
        """
        Toggle lattice frame visibility
        
        """
        if self.mainWindow.refLoaded == 0:
            return
        
        if self.latticeFrame.visible:
            self.removeLatticeFrame()
        
        else:
            self.addLatticeFrame()
        
        self.reinit()
    
    def toggleAxes(self):
        """
        Toggle axes visibilty
        
        """
        if self.mainWindow.refLoaded == 0:
            return
        
        if self.axes.visible:
            self.removeAxes()
        
        else:
            self.addAxes()
        
        self.reinit()
    
    def addAxes(self):
        """
        Add the axis label
        
        """
        dims = self.mainWindow.refState.cellDims
        
#        self.axes.add(dims)
        self.axes.refresh(-8, -8, -8, 0.2 * dims[0], 0.2 * dims[1], 0.2 * dims[2], "x", "y", "z")
    
    def removeAxes(self):
        """
        Remove the axis label
        
        """
        self.axes.remove()
    
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
        pass
#        print "RENDERING"
#        self.removeAllActors()
#        
#        filterLists = self.getFilterLists()
#        count = 0
#        for filterList in filterLists:
#            print "RENDERING LIST", count
#            count += 1
#            
#            filterList.addActors()
    
    def rotateAndSaveImage(self, renderType, imageFormat, fileprefix, overwrite, degreesPerRotation, povray="povray", overlay=False):
        """
        Rotate image.
        
        """
        NRotations = int(360.0 / degreesPerRotation)
        
        # main loop
        for i in xrange(NRotations):
            # file name
            fileprefixFull = "%s%d" % (fileprefix, i)
            
            # save image
            savedFile = self.saveImage(renderType, imageFormat, fileprefixFull, overwrite, povray=povray, overlay=overlay)
            
            if savedFile is None:
                return 1
            
            # apply rotation
            self.camera.Azimuth(degreesPerRotation)
        
        return 0
    
    def saveImage(self, renderType, imageFormat, fileprefix, overwrite, povray="povray", overlay=False):
        """
        Save image to file.
        
        """
        if renderType == "VTK":
            filename = "%s.%s" % (fileprefix, imageFormat)
            
            renWin = self.getRenWin()
            
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(renWin)
            
            if imageFormat == "jpg":
                writer = vtk.vtkJPEGWriter()
            
            elif imageFormat == "png":
                writer = vtk.vtkPNGWriter()
                
            elif imageFormat == "tif":
                writer = vtk.vtkTIFFWriter()
            
            writer.SetInput(w2if.GetOutput())
            
            if not overwrite:
                count = 0
                while os.path.exists(filename):
                    count += 1
                    filename = "%s(%d).%s" % (fileprefix, count, imageFormat)
            
            writer.SetFileName(filename)
            writer.Write()
        
        elif renderType == "POV":
            
#            pov = vtk.vtkPOVExporter()
#            pov.SetRenderWindow(self.renWinInteract.GetRenderWindow())
#            pov.SetFileName("fruitcake.pov")
#            pov.Write()
#            print "WRITTEN"
#            return None
            
            # size of vtk widget
#            print "WINDOW HEIGHT", self.mainWindow.VTKWidget.height()
#            print "WINDOW WIDTH", self.mainWindow.VTKWidget.width()
            renWinH = 600 #self.mainWindow.VTKWidget.height()
            renWinW = 800 #self.mainWindow.VTKWidget.width()
            
            povfile = os.path.join(self.mainWindow.tmpDirectory, "header.pov")
            fh = open(povfile, "w")
            
            # first write the header (camera info etc.)
            self.writePOVRAYHeader(fh)
            
            # write cell frame if visible
            if self.latticeFrame.visible:
                self.writePOVRAYCellFrame(fh)
            
            # write axes if visible
            
            
            fh.close()
            
            # then join filter list files
            filterLists = self.mainWindow.mainToolbar.filterPage.filterLists
            CWD = os.getcwd()
            try:
                os.chdir(self.mainWindow.tmpDirectory)
                command = "cat header.pov"
                for filterList in filterLists:
                    if filterList.visible:
                        if os.path.exists("atoms%d.pov" % filterList.tab):
                            command += " atoms%d.pov" % filterList.tab
                        
                        if os.path.exists("hulls%d.pov" % filterList.tab):
                            command += " hulls%d.pov" % filterList.tab
                        
                        if os.path.exists("defects%d.pov" % filterList.tab):
                            command += " defects%d.pov" % filterList.tab
                        
                command += " > image.pov"
                output, stderr, status = utilities.runSubProcess(command)
                if status:
                    return None
            
            finally:
                os.chdir(CWD)
            
            # output filename
            filename = "%s.%s" % (fileprefix, imageFormat)
            if not overwrite:
                count = 0
                while os.path.exists(filename):
                    count += 1
                    filename = "%s(%d).%s" % (fileprefix, count, imageFormat)
            
            # POV-Ray settings
            settings = self.mainWindow.mainToolbar.outputPage.imageTab.POVSettings
            
            # create povray ini file
            povIniFile = os.path.join(self.mainWindow.tmpDirectory, "image.ini")
            
            lines = []
            nl = lines.append
            nl("; CDJSVis auto-generated POV-Ray INI file")
            nl("Input_File_Name='%s'" % os.path.join(self.mainWindow.tmpDirectory, "image.pov"))
            nl("Width=%d" % settings.HRes)
            nl("Height=%d" % settings.VRes)
            nl("Display=off")
            nl("Antialias=on")
            nl("Output_File_Name='%s'" % filename)
            
            fh = open(povIniFile, "w")
            fh.write("\n".join(lines))
            fh.close()
            
            # run povray
            command = "%s %s" % (povray, povIniFile)
            output, stderr, status = utilities.runSubProcess(command)
            if status:
                print "STDERR:", stderr
                return None
            
            # remove image files
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "image.pov"))
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "header.pov"))
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "image.ini"))
        
        if not os.path.exists(filename):
            print "WARNING: SOMETHING WENT WRONG WITH SAVEIMAGE"
        
        elif renderType == "POV" and overlay:
            self.overlayImage(filename)
        
        return filename
    
    def overlayImage(self, filename):
        """
        Overlay the image with on screen info.
        
        """
        import time
        overlayTime = time.time()
        
        # local refs
        ren = self.ren
        renWinInteract = self.renWinInteract
        
        # to do this we change cam pos to far away and
        # save temp image with just text in
        
        # so move the camera far away
        camera = ren.GetActiveCamera()
        origCamPos = camera.GetPosition()
        
        newCampPos = [0]*3
        newCampPos[0] = origCamPos[0] * 100000
        newCampPos[1] = origCamPos[1] * 100000
        newCampPos[2] = origCamPos[2] * 100000
        
        camera.SetPosition(newCampPos)
        
        try:
            # save image
            overlayFilePrefix = os.path.join(self.mainWindow.tmpDirectory, "overlay")
            
            overlayFile = self.saveImage("VTK", "jpg", overlayFilePrefix, False)
            
            if not os.path.exists(overlayFile):
                print "WARNING: overlay file does not exist: %s" % overlayFile
                return
            
            try:                
                # open POV-Ray image
                povim = Image.open(filename)
                modified = False
                
                # find text in top left corner
                im = Image.open(overlayFile)
                
                # start point
                xmin = ymin = 0
                xmax = int(im.size[0] * 0.5)
                ymax = int(im.size[1] * 0.8)
                
                # find extremes
                xmin, xmax, ymin, ymax = self.findOverlayExtremes(im, xmin, xmax, ymin, ymax)
                
                # crop
                region = im.crop((xmin, ymin, xmax + 2, ymax + 2))
                
                # add to povray image
                if region.size[0] != 0:
                    region = region.resize((region.size[0], region.size[1]), Image.ANTIALIAS)
                    povim.paste(region, (0, 0))
                    modified = True
                
                # now look for anything at the bottom => scalar bar
                im = Image.open(overlayFile)
                
                # start point
                xmin = 0
                ymin = im.size[1] - 80
                xmax = im.size[0]
                ymax = im.size[1]
                
                # find extremes
                xmin, xmax, ymin, ymax = self.findOverlayExtremes(im, xmin, xmax, ymin, ymax)
                
                # crop
                region = im.crop((xmin, ymin, xmax, ymax))
                
                # add?
                if region.size[0] != 0:
                    newregiondimx = int(povim.size[0]*0.8)
                    dx = (float(povim.size[0]) * 0.8 - float(region.size[0])) / float(region.size[0])
                    newregiondimy = region.size[1] + int(region.size[1] * dx)
                    region = region.resize((newregiondimx, newregiondimy), Image.ANTIALIAS)
                    
                    xpos = int((povim.size[0] - region.size[0]) / 2.0)
                    povim.paste(region, (xpos, int(povim.size[1] - region.size[1])))
                    
                    modified = True
                
                # now look for text in top right corner
                im = Image.open(overlayFile)
                
                # start point
                xmin = int(im.size[0] * 0.5)
                ymin = 0
                xmax = im.size[0]
                ymax = int(im.size[1] * 0.6)
                
                # find extremes
                xmin, xmax, ymin, ymax = self.findOverlayExtremes(im, xmin, xmax, ymin, ymax)
                
                # crop
                region = im.crop((xmin - 2, ymin, xmax, ymax + 2))
                
                if region.size[0] != 0:
                    region = region.resize((region.size[0], region.size[1]), Image.ANTIALIAS)
                    xpos = povim.size[0] - 220
                    povim.paste(region, (xpos, 0))
                    
                    modified = True
                
                # save image
                if modified:
                    povim.save(filename)
            
            finally:
                os.unlink(overlayFile)
        
        finally:
            # return to original cam pos
            camera.SetPosition(origCamPos)
            renWinInteract.ReInitialize()
        
        overlayTime = time.time() - overlayTime
#        print "OVERLAY TIME: %f s" % overlayTime
        
    def findOverlayExtremes(self, im, i0, i1, j0, j1):
        """
        Find extremes of non-white area.
        
        """
        xmax = 0
        ymax = 0
        xmin = 1000
        ymin = 1000    
        for i in xrange(i0, i1):
            for j in xrange(j0, j1):
                r,g,b = im.getpixel((i, j))
                
                if r != 255 and g != 255 and b != 255:
                    if i > xmax:
                        xmax = i
                    
                    if j > ymax:    
                        ymax = j
                    
                    if i < xmin:
                        xmin = i
                    
                    if j < ymin:    
                        ymin = j    
        
        return xmin, xmax, ymin, ymax
    
    def writePOVRAYCellFrame(self, filehandle):
        """
        Write cell frame.
        
        """
        lattice = self.mainWindow.inputState
        
        a = [0]*3
        b = [0]*3 
        b[0] = - lattice.cellDims[0]
        b[1] = lattice.cellDims[1]
        b[2] = lattice.cellDims[2]
        
        filehandle.write("#declare R = 0.15;\n")
        filehandle.write("#declare myObject = union {\n")
        filehandle.write("    sphere { <"+str(a[0])+","+str(a[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(b[0])+","+str(a[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(a[0])+","+str(a[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(b[0])+","+str(a[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(a[0])+","+str(b[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(b[0])+","+str(b[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(a[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    sphere { <"+str(b[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(a[1])+","+str(a[2])+">, <"+str(b[0])+","+str(a[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(a[1])+","+str(b[2])+">, <"+str(b[0])+","+str(a[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(b[1])+","+str(a[2])+">, <"+str(b[0])+","+str(b[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(b[1])+","+str(b[2])+">, <"+str(b[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(a[1])+","+str(a[2])+">, <"+str(a[0])+","+str(b[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(a[1])+","+str(b[2])+">, <"+str(a[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(b[0])+","+str(a[1])+","+str(a[2])+">, <"+str(b[0])+","+str(b[1])+","+str(a[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(b[0])+","+str(a[1])+","+str(b[2])+">, <"+str(b[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(a[1])+","+str(a[2])+">, <"+str(a[0])+","+str(a[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(a[0])+","+str(b[1])+","+str(a[2])+">, <"+str(a[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(b[0])+","+str(a[1])+","+str(a[2])+">, <"+str(b[0])+","+str(a[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    cylinder { <"+str(b[0])+","+str(b[1])+","+str(a[2])+">, <"+str(b[0])+","+str(b[1])+","+str(b[2])+">, R }\n")
        filehandle.write("    texture { pigment { color rgb <0,0,0> }\n")
        filehandle.write("              finish { diffuse 0.9 phong 1 } } }\n")
        filehandle.write("object{myObject}\n")
    
    def writePOVRAYHeader(self, filehandle):
        """
        Write POV-Ray header file.
        
        """
        settings = self.mainWindow.mainToolbar.outputPage.imageTab.POVSettings
        
        focalPoint = self.camera.GetFocalPoint()
        campos = self.camera.GetPosition()
        viewup = self.camera.GetViewUp()
        angle = settings.viewAngle
        if settings.shadowless:
            shadowless = "shadowless "
        else:
            shadowless = ""
        
        string = "camera { perspective location <%f,%f,%f> look_at <%f,%f,%f> angle %f\n" % (- campos[0], campos[1], campos[2],
                                                                                             - focalPoint[0], focalPoint[1], focalPoint[2],
                                                                                             angle)
        string += "sky <%f,%f,%f> }\n" % (- viewup[0], viewup[1], viewup[2])
        string += "light_source { <%f,%f,%f> color rgb <1,1,1> %s}\n" % (- campos[0], campos[1], campos[2], shadowless)
        string += "background { color rgb <1,1,1> }\n"
        
        filehandle.write(string)
        
#        lines = []
#        nl = lines.append
#        
#        nl("camera { perspective")
#        nl("         location <%f,%f,%f>" % (- campos[0], campos[1], campos[2]))
#        nl("         sky <%f,%f,%f>" % (- viewup[0], viewup[1], viewup[2]))
##        nl("         right <-1,0,0>")
#        nl("         angle 30.0")
#        nl("         look_at <%f,%f,%f>" % (- focalPoint[0], focalPoint[1], focalPoint[2]))
#        nl("}")
#        
#        nl("light_source { <%f,%f,%f>" % (- campos[0], campos[1], campos[2]))
#        nl("               color <1,1,1>*1.0")
##        nl("               parallel")
#        nl("               point_at <%f,%f,%f>" % (- focalPoint[0], focalPoint[1], focalPoint[2]))
#        nl("}")
#        
#        nl("background { color rgb <1,1,1> }")
#        
#        filehandle.write("\n".join(lines))
        
        
        
################################################################################

def povrayAtom(pos, radius, rgb):
    """
    Return string for rendering atom in POV-Ray.
    
    """
    line = "sphere { <%f,%f,%f>, %f pigment { color rgb <%f,%f,%f> } finish { ambient %f phong %f } }\n" % (-pos[0], pos[1], pos[2], radius, rgb[0], rgb[1], rgb[2], 0.25, 0.9)
    
    return line     

################################################################################
def setupLUT(specieList, specieRGB, colouringOptions):
    """
    Setup the colour look up table
    
    """
    lut = vtk.vtkLookupTable()
    
    if colouringOptions.colourBy == "Specie" or colouringOptions.colourBy == "Solid colour":
        NSpecies = len(specieList)
        
        lut.SetNumberOfColors(NSpecies)
        lut.SetNumberOfTableValues(NSpecies)
        lut.SetTableRange(0, NSpecies - 1)
        lut.SetRange(0, NSpecies - 1)
        
        for i in xrange(NSpecies):
            if colouringOptions.colourBy == "Specie":
                lut.SetTableValue(i, specieRGB[i][0], specieRGB[i][1], specieRGB[i][2], 1.0)
            
            elif colouringOptions.colourBy == "Solid colour":
                lut.SetTableValue(i, colouringOptions.solidColourRGB[0], colouringOptions.solidColourRGB[1], colouringOptions.solidColourRGB[2])
    
    elif colouringOptions.colourBy == "Height":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.minVal, colouringOptions.maxVal)    
        lut.SetRampToLinear()
        lut.Build()
    
    elif colouringOptions.colourBy == "Atom property":
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.propertyMinSpin.value(), colouringOptions.propertyMaxSpin.value())    
        lut.SetRampToLinear()
        lut.Build()
    
    else:
        lut.SetNumberOfColors(1024)
        lut.SetHueRange(0.667,0.0)
        lut.SetRange(colouringOptions.scalarMinSpin.value(), colouringOptions.scalarMaxSpin.value())    
        lut.SetRampToLinear()
        lut.Build()
    
    return lut

################################################################################
def getActorsForFilteredSystem(visibleAtoms, mainWindow, actorsCollection, colouringOptions, povFileName, scalarsArray, NVisibleForRes=None):
    """
    Make the actors for the filtered system
    
    """
    NVisible = len(visibleAtoms)
    
    if NVisibleForRes is None:
        NVisibleForRes = NVisible
    
    # povray file
    povFilePath = os.path.join(mainWindow.tmpDirectory, povFileName)
    fpov = open(povFilePath, "w")
    
    # resolution
    res = setRes(NVisibleForRes)
    
    lattice = mainWindow.inputState
    
    # make LUT
    lut = setupLUT(lattice.specieList, lattice.specieRGB, colouringOptions)
    
    NSpecies = len(lattice.specieList)
    specieCount = np.zeros(NSpecies, np.int32)
    
    atomPointsList = []
    atomScalarsList = []
    for i in xrange(NSpecies):
        atomPointsList.append(vtk.vtkPoints())
        atomScalarsList.append(vtk.vtkFloatArray())
        
    # loop over atoms, setting points and scalars
    pos = lattice.pos
    spec = lattice.specie
    for i in xrange(NVisible):
        index = visibleAtoms[i]
        specInd = spec[index]
        
        # specie counter
        specieCount[specInd] += 1
        
        # position
        atomPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        
        # scalar
        if colouringOptions.colourBy == "Specie" or colouringOptions.colourBy == "Solid colour":
            scalar = specInd
        
        elif colouringOptions.colourBy == "Height":
            scalar = pos[3*index+colouringOptions.heightAxis]
        
        elif colouringOptions.colourBy == "Atom property":
            if colouringOptions.atomPropertyType == "Kinetic energy":
                scalar = lattice.KE[index]
            elif colouringOptions.atomPropertyType == "Potential energy":
                scalar = lattice.PE[index]
            else:
                scalar = lattice.charge[index]
        
        else:
            scalar = scalarsArray[i]
        
        # store scalar value
        atomScalarsList[specInd].InsertNextValue(scalar)
        
        # colour for povray file
        rgb = np.empty(3, np.float64)
        lut.GetColor(scalar, rgb)
        
        # povray atom
        fpov.write(povrayAtom(pos[3*index:3*index+3], lattice.specieCovalentRadius[specInd], rgb))
        
    # now loop over species, making actors
    for i in xrange(NSpecies):
        
        atomsPolyData = vtk.vtkPolyData()
        atomsPolyData.SetPoints(atomPointsList[i])
        atomsPolyData.GetPointData().SetScalars(atomScalarsList[i])
        
        atomsGlyphSource = vtk.vtkSphereSource()
        atomsGlyphSource.SetRadius(lattice.specieCovalentRadius[i])
        atomsGlyphSource.SetPhiResolution(res)
        atomsGlyphSource.SetThetaResolution(res)
        
        atomsGlyph = vtk.vtkGlyph3D()
        atomsGlyph.SetSource(atomsGlyphSource.GetOutput())
        atomsGlyph.SetInput(atomsPolyData)
        atomsGlyph.SetScaleFactor(1.0)
        atomsGlyph.SetScaleModeToDataScalingOff()
        
        atomsMapper = vtk.vtkPolyDataMapper()
        atomsMapper.SetInput(atomsGlyph.GetOutput())
        atomsMapper.SetLookupTable(lut)
        if colouringOptions.colourBy == "Specie":
            atomsMapper.SetScalarRange(0, NSpecies - 1)
        
        elif colouringOptions.colourBy == "Height":
            atomsMapper.SetScalarRange(colouringOptions.minVal, colouringOptions.maxVal)
        
        elif colouringOptions.colourBy == "Atom property":
            atomsMapper.SetScalarRange(colouringOptions.propertyMinSpin.value(), colouringOptions.propertyMaxSpin.value())
        
        else:
            atomsMapper.SetScalarRange(colouringOptions.scalarMinSpin.value(), colouringOptions.scalarMaxSpin.value())
        
        atomsActor = vtk.vtkActor()
        atomsActor.SetMapper(atomsMapper)
        
        actorsCollection.AddItem(atomsActor)
        
    fpov.close()
    
    # scalar bar
    scalarBar = None
    if colouringOptions.colourBy != "Specie" and colouringOptions.colourBy != "Solid colour":
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(lut)
        
        if colouringOptions.colourBy == "Height":
            title = colouringOptions.scalarBarText
        elif colouringOptions.colourBy == "Atom property":
            title = str(colouringOptions.scalarBarTextEdit3.text())
        else:
            title = str(colouringOptions.scalarBarTextEdit2.text())
        
        scalarBar.SetTitle(title)
        scalarBar.SetOrientationToHorizontal()
        
        lprop = scalarBar.GetTitleTextProperty()
        lprop.SetColor((0, 0, 0))
        lprop.ItalicOff()
        lprop.BoldOn()
        lprop.SetFontSize(20)
        lprop.SetFontFamilyToArial()
        
        lprop = scalarBar.GetLabelTextProperty()
        lprop.SetColor((0, 0, 0))
        lprop.ItalicOff()
        lprop.BoldOn()
        lprop.SetFontSize(10)
        lprop.SetFontFamilyToArial()
        
        scalarBar.SetWidth(0.85)
        scalarBar.GetPositionCoordinate().SetValue(0.1, 0.01)
        scalarBar.SetHeight(0.12)
    
    return scalarBar, specieCount


################################################################################
def writePovrayDefects(filename, vacancies, interstitials, antisites, onAntisites, 
                       settings, mainWindow):
    """
    Write defects to povray file.
    
    """
    povfile = os.path.join(mainWindow.tmpDirectory, filename)
    
    inputLattice = mainWindow.inputState
    refLattice = mainWindow.refState
    
    output_c.writePOVRAYDefects(povfile, vacancies, interstitials, antisites, onAntisites, inputLattice.specie, inputLattice.pos,
                                refLattice.specie, refLattice.pos, inputLattice.specieRGB, inputLattice.specieCovalentRadius,
                                refLattice.specieRGB, refLattice.specieCovalentRadius)


################################################################################
def writePovrayAtoms(filename, visibleAtoms, mainWindow):
    """
    Write pov-ray atoms to file.
    
    """
    povfile = os.path.join(mainWindow.tmpDirectory, filename)
    
    lattice = mainWindow.inputState
    
    # call C routine to write atoms to file
    output_c.writePOVRAYAtoms(povfile, lattice.specie, lattice.pos, visibleAtoms, 
                              lattice.specieRGB, lattice.specieCovalentRadius)


################################################################################
def writePovrayHull(facets, clusterPos, mainWindow, filename, settings):
    """
    Write hull to POV-Ray file.
    
    """
    if len(clusterPos) / 3 < 3:
        pass
    
    else:
        if os.path.exists(filename):
            fh = open(filename, "a")
        
        else:
            fh = open(filename, "w")
        
        # how many vertices
        vertices = set()
        vertexMapper = {}
        NVertices = 0
        for facet in facets:
            for j in xrange(3):
                if facet[j] not in vertices:
                    vertices.add(facet[j])
                    vertexMapper[facet[j]] = NVertices
                    NVertices += 1
        
        # construct mesh
        lines = []
        nl = lines.append
        
        nl("mesh2 {")
        nl("  vertex_vectors {")
        nl("    %d," % NVertices)
        
        count = 0
        for key, value in sorted(vertexMapper.iteritems(), key=lambda (k, v): (v, k)):
            if count == NVertices - 1:
                string = ""
            
            else:
                string = ","
            
            nl("    <%f,%f,%f>%s" % (- clusterPos[3*key], clusterPos[3*key+1], clusterPos[3*key+2], string))
            
            count += 1
        
        nl("  }")
        nl("  face_indices {")
        nl("    %d," % len(facets))
        
        count = 0
        for facet in facets:
            if count == len(facets) - 1:
                string = ""
            
            else:
                string = ","
            
            nl("    <%d,%d,%d>%s" % (vertexMapper[facet[0]], vertexMapper[facet[1]], vertexMapper[facet[2]], string))
            
            count += 1
        
        nl("  }")
        nl("  pigment { color rgbt <%f,%f,%f,%f> }" % (settings.hullCol[0], settings.hullCol[1], 
                                                       settings.hullCol[2], 1.0 - settings.hullOpacity))
        nl("  finish { diffuse 0.4 ambient 0.25 phong 0.9 }")
        nl("}")
        nl("")
        
        fh.write("\n".join(lines))

    
################################################################################
def getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, splitInterstitials, mainWindow, actorsCollection, colouringOptions):
    
    NInt = len(interstitials)
    NVac = len(vacancies)
    NAnt = len(antisites)
    NSplit = len(splitInterstitials) / 3
    NDef = NInt + NVac + NAnt + len(splitInterstitials)
    
    # resolution
    res = setRes(NDef)
    
    inputLattice = mainWindow.inputState
    refLattice = mainWindow.refState
    
    # specie counters
    NSpeciesInput = len(inputLattice.specieList)
    NSpeciesRef = len(refLattice.specieList)
    vacSpecCount = np.zeros(NSpeciesRef, np.int32)
    intSpecCount = np.zeros(NSpeciesInput, np.int32)
    antSpecCount = np.zeros((NSpeciesRef, NSpeciesInput), np.int32)
    splitSpecCount = np.zeros((NSpeciesInput, NSpeciesInput), np.int32)
    
    #----------------------------------------#
    # interstitials first
    #----------------------------------------#
    NSpecies = len(inputLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(inputLattice.specieList, inputLattice.specieRGB, colouringOptions)
    
    # loop over interstitials, settings points
    pos = inputLattice.pos
    spec = inputLattice.specie
    for i in xrange(NInt):
        index = interstitials[i]
        specInd = spec[index]
        
        intSpecCount[specInd] += 1
        
        intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd].InsertNextValue(specInd)
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        intsPolyData = vtk.vtkPolyData()
        intsPolyData.SetPoints(intPointsList[i])
        intsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        intsGlyphSource = vtk.vtkSphereSource()
        intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i])
        intsGlyphSource.SetPhiResolution(res)
        intsGlyphSource.SetThetaResolution(res)
        
        intsGlyph = vtk.vtkGlyph3D()
        intsGlyph.SetSource(intsGlyphSource.GetOutput())
        intsGlyph.SetInput(intsPolyData)
        intsGlyph.SetScaleFactor(1.0)
        intsGlyph.SetScaleModeToDataScalingOff()
        
        intsMapper = vtk.vtkPolyDataMapper()
        intsMapper.SetInput(intsGlyph.GetOutput())
        intsMapper.SetLookupTable(lut)
        intsMapper.SetScalarRange(0, NSpecies - 1)
        
        intsActor = vtk.vtkActor()
        intsActor.SetMapper(intsMapper)
        
        actorsCollection.AddItem(intsActor)
    
    #----------------------------------------#
    # split interstitial atoms next
    #----------------------------------------#
    NSpecies = len(inputLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(inputLattice.specieList, inputLattice.specieRGB, colouringOptions)
    
    # loop over interstitials, settings points
    pos = inputLattice.pos
    spec = inputLattice.specie
    for i in xrange(NSplit):
        # first 
        index = splitInterstitials[3*i+1]
        specInd1 = spec[index]
        
        intPointsList[specInd1].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd1].InsertNextValue(specInd1)
        
        # second
        index = splitInterstitials[3*i+2]
        specInd2 = spec[index]
        
        intPointsList[specInd2].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd2].InsertNextValue(specInd2)
        
        # counter
        splitSpecCount[specInd1][specInd2] += 1
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        intsPolyData = vtk.vtkPolyData()
        intsPolyData.SetPoints(intPointsList[i])
        intsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        intsGlyphSource = vtk.vtkSphereSource()
        intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i])
        intsGlyphSource.SetPhiResolution(res)
        intsGlyphSource.SetThetaResolution(res)
        
        intsGlyph = vtk.vtkGlyph3D()
        intsGlyph.SetSource(intsGlyphSource.GetOutput())
        intsGlyph.SetInput(intsPolyData)
        intsGlyph.SetScaleFactor(1.0)
        intsGlyph.SetScaleModeToDataScalingOff()
        
        intsMapper = vtk.vtkPolyDataMapper()
        intsMapper.SetInput(intsGlyph.GetOutput())
        intsMapper.SetLookupTable(lut)
        intsMapper.SetScalarRange(0, NSpecies - 1)
        
        intsActor = vtk.vtkActor()
        intsActor.SetMapper(intsMapper)
        
        actorsCollection.AddItem(intsActor)
    
    #----------------------------------------#
    # split interstitial bonds next
    #----------------------------------------#
    NSpecies = len(refLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB, colouringOptions)
    
    # loop over interstitials, settings points
    pos = refLattice.pos
    spec = refLattice.specie
    for i in xrange(NSplit):
        index = splitInterstitials[3*i]
        specInd = spec[index]
        
        intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd].InsertNextValue(specInd)
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        vacsPolyData = vtk.vtkPolyData()
        vacsPolyData.SetPoints(intPointsList[i])
        vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        vacsGlyphSource = vtk.vtkCubeSource()
        vacsGlyphSource.SetXLength(1.5 * refLattice.specieCovalentRadius[i])
        vacsGlyphSource.SetYLength(1.5 * refLattice.specieCovalentRadius[i])
        vacsGlyphSource.SetZLength(1.5 * refLattice.specieCovalentRadius[i])
        
        vacsGlyph = vtk.vtkGlyph3D()
        vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
        vacsGlyph.SetInput(vacsPolyData)
        vacsGlyph.SetScaleFactor(1.0)
        vacsGlyph.SetScaleModeToDataScalingOff()
        
        vacsMapper = vtk.vtkPolyDataMapper()
        vacsMapper.SetInput(vacsGlyph.GetOutput())
        vacsMapper.SetLookupTable(lut)
        vacsMapper.SetScalarRange(0, NSpecies - 1)
        
        vacsActor = vtk.vtkActor()
        vacsActor.SetMapper(vacsMapper)
        vacsActor.GetProperty().SetSpecular(0.4)
        vacsActor.GetProperty().SetSpecularPower(10)
        vacsActor.GetProperty().SetOpacity(0.8)
        
        actorsCollection.AddItem(vacsActor)
    
    
    
    #----------------------------------------#
    # antisites occupying atom
    #----------------------------------------#
    NSpecies = len(inputLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB, colouringOptions)
    
    # loop over interstitials, settings points
    pos = refLattice.pos
    spec = inputLattice.specie
    for i in xrange(NAnt):
        index = onAntisites[i]
        specInd = spec[index]
        intScalarsList[specInd].InsertNextValue(specInd)
        
        index = antisites[i]
        intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        intsPolyData = vtk.vtkPolyData()
        intsPolyData.SetPoints(intPointsList[i])
        intsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        intsGlyphSource = vtk.vtkSphereSource()
        intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i])
        intsGlyphSource.SetPhiResolution(res)
        intsGlyphSource.SetThetaResolution(res)
        
        intsGlyph = vtk.vtkGlyph3D()
        intsGlyph.SetSource(intsGlyphSource.GetOutput())
        intsGlyph.SetInput(intsPolyData)
        intsGlyph.SetScaleFactor(1.0)
        intsGlyph.SetScaleModeToDataScalingOff()
        
        intsMapper = vtk.vtkPolyDataMapper()
        intsMapper.SetInput(intsGlyph.GetOutput())
        intsMapper.SetLookupTable(lut)
        intsMapper.SetScalarRange(0, NSpecies - 1)
        
        intsActor = vtk.vtkActor()
        intsActor.SetMapper(intsMapper)
        
        actorsCollection.AddItem(intsActor)
    
    #----------------------------------------#
    # vacancies
    #----------------------------------------#
    NSpecies = len(refLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB, colouringOptions)
    
    # loop over interstitials, settings points
    pos = refLattice.pos
    spec = refLattice.specie
    for i in xrange(NVac):
        index = vacancies[i]
        specInd = spec[index]
        
        vacSpecCount[specInd] += 1
        
        intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd].InsertNextValue(specInd)
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        vacsPolyData = vtk.vtkPolyData()
        vacsPolyData.SetPoints(intPointsList[i])
        vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        vacsGlyphSource = vtk.vtkCubeSource()
        vacsGlyphSource.SetXLength(1.5 * refLattice.specieCovalentRadius[i])
        vacsGlyphSource.SetYLength(1.5 * refLattice.specieCovalentRadius[i])
        vacsGlyphSource.SetZLength(1.5 * refLattice.specieCovalentRadius[i])
        
        vacsGlyph = vtk.vtkGlyph3D()
        vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
        vacsGlyph.SetInput(vacsPolyData)
        vacsGlyph.SetScaleFactor(1.0)
        vacsGlyph.SetScaleModeToDataScalingOff()
        
        vacsMapper = vtk.vtkPolyDataMapper()
        vacsMapper.SetInput(vacsGlyph.GetOutput())
        vacsMapper.SetLookupTable(lut)
        vacsMapper.SetScalarRange(0, NSpecies - 1)
        
        vacsActor = vtk.vtkActor()
        vacsActor.SetMapper(vacsMapper)
        vacsActor.GetProperty().SetSpecular(0.4)
        vacsActor.GetProperty().SetSpecularPower(10)
        vacsActor.GetProperty().SetOpacity(0.8)
        
        actorsCollection.AddItem(vacsActor)
    
    #----------------------------------------#
    # antisites
    #----------------------------------------#
    NSpecies = len(refLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB, colouringOptions)
    
    # loop over antisites, settings points
    pos = refLattice.pos
    spec = refLattice.specie
    for i in xrange(NAnt):
        index = antisites[i]
        specInd = spec[index]
        
        antSpecCount[specInd][inputLattice.specie[onAntisites[i]]] += 1
        
        intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        intScalarsList[specInd].InsertNextValue(specInd)
    
    # now loop over species making actors
    for i in xrange(NSpecies):
        
        vacsPolyData = vtk.vtkPolyData()
        vacsPolyData.SetPoints(intPointsList[i])
        vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
        
        cubeGlyphSource = vtk.vtkCubeSource()
        cubeGlyphSource.SetXLength(2.0 * refLattice.specieCovalentRadius[i])
        cubeGlyphSource.SetYLength(2.0 * refLattice.specieCovalentRadius[i])
        cubeGlyphSource.SetZLength(2.0 * refLattice.specieCovalentRadius[i])
        edges = vtk.vtkExtractEdges()
        edges.SetInputConnection(cubeGlyphSource.GetOutputPort())
        vacsGlyphSource = vtk.vtkTubeFilter()
        vacsGlyphSource.SetInputConnection(edges.GetOutputPort())
        vacsGlyphSource.SetRadius(0.1)
        vacsGlyphSource.SetNumberOfSides(5)
        vacsGlyphSource.UseDefaultNormalOn()
        vacsGlyphSource.SetDefaultNormal(.577, .577, .577)
        
        vacsGlyph = vtk.vtkGlyph3D()
        vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
        vacsGlyph.SetInput(vacsPolyData)
        vacsGlyph.SetScaleFactor(1.0)
        vacsGlyph.SetScaleModeToDataScalingOff()
        
        vacsMapper = vtk.vtkPolyDataMapper()
        vacsMapper.SetInput(vacsGlyph.GetOutput())
        vacsMapper.SetLookupTable(lut)
        vacsMapper.SetScalarRange(0, NSpecies - 1)
        
        vacsActor = vtk.vtkActor()
        vacsActor.SetMapper(vacsMapper)
        
        actorsCollection.AddItem(vacsActor)
        
        return (vacSpecCount, intSpecCount, antSpecCount, splitSpecCount)


################################################################################
def makeTriangle(indexes):
    """
    Make a triangle given indexes in points array
    
    """
    inda = indexes[0]
    indb = indexes[1]
    indc = indexes[2]
    
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0,inda)
    triangle.GetPointIds().SetId(1,indb)
    triangle.GetPointIds().SetId(2,indc)
    
    return triangle


################################################################################
def getActorsForHullFacets(facets, pos, mainWindow, actorsCollection, settings):
    """
    Render convex hull facets
    
    """
    # probably want to pass some settings through too eg colour, opacity etc
    
    
    points = vtk.vtkPoints()
    for i in xrange(len(pos) / 3):
        points.InsertNextPoint(pos[3*i], pos[3*i+1], pos[3*i+2])
    
    # create triangles
    triangles = vtk.vtkCellArray()
    for i in xrange(len(facets)):
        triangle = makeTriangle(facets[i])
        triangles.InsertNextCell(triangle)
    
    # polydata object
    trianglePolyData = vtk.vtkPolyData()
    trianglePolyData.SetPoints(points)
    trianglePolyData.SetPolys(triangles)
    
    # mapper
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInput(trianglePolyData)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(settings.hullOpacity)
    actor.GetProperty().SetColor(settings.hullCol[0], settings.hullCol[1], settings.hullCol[2])
    
    actorsCollection.AddItem(actor)


