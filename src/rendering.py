
"""
Module for rendering

@author: Chris Scott

"""

import os
import sys
import math

import vtk

from visclibs import output_c
import utilities


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
class Axes:
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
class Renderer:
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
    
    def saveImage(self, renderType, imageFormat, fileprefix, overwrite, povray="povray"):
        """
        Save image to file
        
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
                        command += " filter%d.pov" % (filterList.tab,)
                command += " > image.pov"
                output, stderr, status = utilities.runSubProcess(command)
                if status:
                    return None
            
            finally:
                os.chdir(CWD)
            
            # now run pov-ray
            filename = "%s.%s" % (fileprefix, imageFormat)
            if not overwrite:
                count = 0
                while os.path.exists(filename):
                    count += 1
                    filename = "%s(%d).%s" % (fileprefix, count, imageFormat)
            
            command = "%s -I%s -D +A +W%d +H%d +O'%s'" % (povray, os.path.join(self.mainWindow.tmpDirectory, "image.pov"), 
                                                          800, 600, filename)
            output, stderr, status = utilities.runSubProcess(command)
            if status:
                print "STDERR:", stderr
                return None
        
        return filename
    
    def writePOVRAYCellFrame(self, filehandle):
        """
        Write cell frame.
        
        """
        lattice = self.mainWindow.inputState
        
        a = [0]*3
        b = lattice.cellDims
        b[0] = - 1 * lattice.cellDims[0]
        
        filehandle.write("#declare R = 0.1;\n")
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
        focalPoint = self.camera.GetFocalPoint()
        campos = self.camera.GetPosition()
        viewup = self.camera.GetViewUp()
        
        string = "camera { perspective location <%f,%f,%f> look_at <%f,%f,%f> angle %f\n" % (campos[0] * -1, campos[1], campos[2],
                                                                                             focalPoint[0] * -1, focalPoint[1], focalPoint[2],
                                                                                             self.camera.GetViewAngle())
        string += "sky <%f,%f,%f> }\n" % (viewup[0] * -1, viewup[1], viewup[2])
        string += "light_source { <%f,%f,%f> color rgb <1,1,1> }\n" % (campos[0] * -1, campos[1], campos[2])
        string += "background { color rgb <1,1,1> }\n"
        
        filehandle.write(string)
        
        
        

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
    
    return lut

################################################################################
def getActorsForFilteredSystem(visibleAtoms, mainWindow, actorsCollection):
    """
    Make the actors for the filtered system
    
    """
    NVisible = len(visibleAtoms)
    
    # resolution
    res = setRes(NVisible)
    
    lattice = mainWindow.inputState
    
    # make LUT
    lut = setupLUT(lattice.specieList, lattice.specieRGB)
    
    NSpecies = len(lattice.specieList)
    
    atomPointsList = []
    atomScalarsList = []
    for i in xrange(NSpecies):
        atomPointsList.append(vtk.vtkPoints())
        atomScalarsList.append(vtk.vtkFloatArray())
        
    # loop over atoms, setting points
    pos = lattice.pos
    spec = lattice.specie
    for i in xrange(NVisible):
        index = visibleAtoms[i]
        specInd = spec[index]
        
        atomPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        atomScalarsList[specInd].InsertNextValue(specInd)
        
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
        atomsMapper.SetScalarRange(0, NSpecies - 1)
        
        atomsActor = vtk.vtkActor()
        atomsActor.SetMapper(atomsMapper)
        
        actorsCollection.AddItem(atomsActor)


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
def getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, mainWindow, actorsCollection):
    
    NInt = len(interstitials)
    NVac = len(vacancies)
    NAnt = len(antisites)
    NDef = NInt + NVac + NAnt
    
    # resolution
    res = setRes(NDef)
    
    inputLattice = mainWindow.inputState
    refLattice = mainWindow.refState
    
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
    lut = setupLUT(inputLattice.specieList, inputLattice.specieRGB)
    
    # loop over interstitials, settings points
    pos = inputLattice.pos
    spec = inputLattice.specie
    for i in xrange(NInt):
        index = interstitials[i]
        specInd = spec[index]
        
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
    # antisites occupying atom
    #----------------------------------------#
    NSpecies = len(inputLattice.specieList)
    intPointsList = []
    intScalarsList = []
    for i in xrange(NSpecies):
        intPointsList.append(vtk.vtkPoints())
        intScalarsList.append(vtk.vtkFloatArray())
    
    # make LUT
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB)
    
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
        intsGlyphSource.SetRadius(refLattice.specieCovalentRadius[i])
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
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB)
    
    # loop over interstitials, settings points
    pos = refLattice.pos
    spec = refLattice.specie
    for i in xrange(NVac):
        index = vacancies[i]
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
        vacsActor.GetProperty().SetOpacity(1.0)
        
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
    lut = setupLUT(refLattice.specieList, refLattice.specieRGB)
    
    # loop over interstitials, settings points
    pos = refLattice.pos
    spec = refLattice.specie
    for i in xrange(NAnt):
        index = antisites[i]
        specInd = spec[index]
        
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
def getActorsForHullFacets(facets, pos, mainWindow, actorsCollection):
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
    actor.GetProperty().SetOpacity(0.5)
    actor.GetProperty().SetColor(0,0,1)
    
    actorsCollection.AddItem(actor)


