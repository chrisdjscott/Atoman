
"""
Module for rendering

@author: Chris Scott

"""
import os
import sys
import shutil
import glob
import logging
import time
import threading
import Queue
import uuid

import numpy as np
import vtk
from vtk.util import numpy_support
from PIL import Image
from PySide import QtGui, QtCore

from ..visutils import utilities
from ..state import _output as output_c
from . import axes
from . import cell
from .utils import setRes, setupLUT, getScalar, setMapperScalarRange, makeScalarBar, getScalarsType
from . import utils
from . import _rendering
from ..visutils.threading_vis import GenericRunnable
from ..algebra import vectors as vectorslib


################################################################################
class Renderer(object):
    def __init__(self, parent):
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        self.ren = self.parent.vtkRen
        self.renWinInteract = self.parent.vtkRenWinInteract
        self.renWin = self.parent.vtkRenWin
        
        self.log = self.parent.mainWindow.console.write
        self.logger = logging.getLogger(__name__+".Renderer")
        
        # is the interactor initialised
        self.init = False
        
        # setup stuff
        self.camera = self.ren.GetActiveCamera()
        
        # lattice frame
        self.latticeFrame = cell.CellOutline(self.ren)
        
        # axes
        self.useNewAxes = True
        
        # old axes
        self.axes = axes.AxesBasic(self.ren, self.reinit)
        self.axes.remove()
        
        # new axes
        self.axesNew = vtk.vtkAxesActor()
        self.axesNew.SetShaftTypeToCylinder()
        self.axesNew.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0)
        self.axesNew.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self.axesNew.GetXAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self.axesNew.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,1,0)
        self.axesNew.GetYAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self.axesNew.GetYAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        self.axesNew.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetColor(0,0,1)
        self.axesNew.GetZAxisCaptionActor2D().GetCaptionTextProperty().SetFontFamilyToArial()
        self.axesNew.GetZAxisCaptionActor2D().GetCaptionTextProperty().ShadowOff()
        
        self.axesMarker = vtk.vtkOrientationMarkerWidget()
        self.axesMarker.SetInteractor(self.renWinInteract)
        self.axesMarker.SetOrientationMarker(self.axesNew)
        self.axesMarker.SetViewport(0, 0, 0.25, 0.25)
        self.axesMarker.SetEnabled(0)
        self.axesEnabled = False
        
#         self.distanceWidget = vtk.vtkDistanceWidget()
#         self.distanceWidget.SetInteractor(self.renWinInteract)
#         self.distanceWidget.CreateDefaultRepresentation()
#         self.distanceWidget.EnabledOn()
#         self.distanceRepr = self.distanceWidget.GetDistanceRepresentation()
            
    def reinit(self):
        """
        Reinitialise.
        
        """
        if self.init:
            self.renWinInteract.ReInitialize()
        else:
            self.renWinInteract.Initialize()
            self.init = True
    
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
    
    def getRefState(self):
        """
        Get the current ref state
        
        """
        return self.parent.getCurrentRefState()
    
    def getInputState(self):
        """
        Get the current input state
        
        """
        return self.parent.getCurrentInputState()
    
    def setCameraToCell(self):
        """
        Point the camera at the centre of the cell.
        
        """
        ref = self.getRefState()
        if ref is None:
            return
        
        dims = ref.cellDims
        
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
        ref = self.getRefState()
        
        dims = ref.cellDims
        
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
        if self.parent.getCurrentPipelinePage().refState is None:
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
        if self.parent.getCurrentPipelinePage().refState is None:
            return
        
        if self.useNewAxes:
            if self.axesEnabled:
                self.removeAxes()
            else:
                self.addAxes()
        
        else:
            if self.axes.visible:
                self.removeAxes()
            else:
                self.addAxes()
        
        self.reinit()
    
    def addAxes(self):
        """
        Add the axis label
        
        """
        if self.useNewAxes:
            if not self.axesEnabled:
                self.axesMarker.SetEnabled(1)
                self.axesEnabled = True
        
        else:
            ref = self.getRefState()
            dims = ref.cellDims
        
            self.axes.refresh(-8, -8, -8, 0.2 * dims[0], 0.2 * dims[1], 0.2 * dims[2], "x", "y", "z")
    
    def removeAxes(self):
        """
        Remove the axis label
        
        """
        if self.useNewAxes:
            if self.axesEnabled:
                self.axesMarker.SetEnabled(0)
                self.axesEnabled = False
        
        else:
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
        return self.parent.getFilterLists()
    
    def getCurrentPipelinePage(self):
        """
        Get current pipeline page
        
        """
        return self.parent.getCurrentPipelinePage()
    
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
    
    def rotateAndSaveImage(self, renderType, imageFormat, fileprefix, overwrite, degreesPerRotation, povray="povray"):
        """
        Rotate image.
        
        """
        NRotations = int(360.0 / degreesPerRotation) + 1
        
        # save camera
        campos = self.camera.GetPosition()
        camfoc = self.camera.GetFocalPoint()
        camvup = self.camera.GetViewUp()
        
        # progress dialog
        progDialog = QtGui.QProgressDialog("Running rotator...", "Cancel", 0, NRotations)
        progDialog.setWindowModality(QtCore.Qt.WindowModal)
        progDialog.setWindowTitle("Progress")
        progDialog.setValue(0)
        
        progDialog.show()
        QtGui.QApplication.processEvents()
        
        # main loop
        try:
            status = 0
            for i in xrange(NRotations):
                # file name
                fileprefixFull = "%s%d" % (fileprefix, i)
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    status = 2
                    break
                
                # save image
                savedFile = self.saveImage(renderType, imageFormat, fileprefixFull, overwrite, povray=povray)
                
                if savedFile is None:
                    status = 1
                    break
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    status = 2
                    break
                
                # progress
                progDialog.setValue(i)
                
                QtGui.QApplication.processEvents()
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    status = 2
                    break
                
                # apply rotation
                self.camera.Azimuth(degreesPerRotation)
                
                # exit if cancelled
                if progDialog.wasCanceled():
                    status = 2
                    break
                
                self.reinit()
        
        finally:
            # close progress dialog
            progDialog.close()
        
        # restore camera
        self.camera.SetFocalPoint(camfoc)
        self.camera.SetPosition(campos)
        self.camera.SetViewUp(camvup)
        
        self.reinit()
        
        return status
    
    def saveImage(self, renderType, imageFormat, fileprefix, overwrite, povray="povray"):
        """
        Save image to file.
        
        """
        logger = self.logger
        
        if renderType == "VTK":
            filename = "%s.%s" % (fileprefix, imageFormat)
            
            renWin = self.renWin
            
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(renWin)
            
            if imageFormat == "jpg":
                writer = vtk.vtkJPEGWriter()
            
            elif imageFormat == "png":
                writer = vtk.vtkPNGWriter()
                
            elif imageFormat == "tif":
                writer = vtk.vtkTIFFWriter()
            
            writer.SetInputConnection(w2if.GetOutputPort())
            
            if not overwrite:
                count = 0
                while os.path.exists(filename):
                    count += 1
                    filename = "%s(%d).%s" % (fileprefix, count, imageFormat)
            
            writer.SetFileName(filename)
            writer.Write()
        
        elif renderType == "POV":
#             pov = vtk.vtkPOVExporter()
#             pov.SetRenderWindow(self.renWinInteract.GetRenderWindow())
#             pov.SetFileName("fruitcake.pov")
#             pov.Write()
#             print "WRITTEN"
#             return None
            
            # check pov files are ready
            filterLists = self.getFilterLists()
            for filterList in filterLists:
                if filterList.visible and not filterList.filterer.povrayAtomsWritten:
                    self.mainWindow.displayError("Error: POV-Ray atoms not written to file yet; please try again in a few seconds")
                    return
            
            renIndex = self.parent.rendererIndex
            pipelineIndex = self.parent.currentPipelineIndex
            
            self.logger.debug("Ren %d; PIPE %d", renIndex, pipelineIndex)
            
            # header file
            povfile = os.path.join(self.mainWindow.tmpDirectory, "renderer%d_header.pov" % renIndex)
            fh = open(povfile, "w")
            
            # first write the header (camera info etc.)
            self.writePOVRAYHeader(fh)
            
            # write cell frame if visible
            if self.latticeFrame.visible:
                self.writePOVRAYCellFrame(fh)
            
            # write axes if visible
            
            
            fh.close()
            
            # POV-Ray settings
            settings = self.mainWindow.preferences.povrayForm
            overlay = settings.overlayImage
            
            # then join filter list files
            pp = self.getCurrentPipelinePage()
            filterLists = self.parent.getFilterLists()
            CWD = os.getcwd()
            try:
                os.chdir(self.mainWindow.tmpDirectory)
                command = "cat 'renderer%d_header.pov'" % renIndex
                for filterList in filterLists:
                    if filterList.visible:
                        pipeline_pov_files = glob.glob("pipeline%d_*%d_%s.pov" % (pipelineIndex, filterList.tab, str(pp.currentRunID)))
                        
                        for fn in pipeline_pov_files:
                            command += " '%s'" % fn
                
                fullPovFile = "renderer%d_image.pov" % renIndex
                
                command += " > '%s'" % fullPovFile
                output, stderr, status = utilities.runSubProcess(command)
                if status:
                    return None
                
                # create povray ini file
                povIniFile = "renderer%d_image.ini" % renIndex
                
                tmpPovOutputFile = "renderer%d_image.%s" % (renIndex, imageFormat)
                
                lines = []
                nl = lines.append
                nl("; CDJSVis auto-generated POV-Ray INI file")
                nl("Input_File_Name='%s'" % fullPovFile)
                nl("Width=%d" % settings.HRes)
                nl("Height=%d" % settings.VRes)
                nl("Display=off")
                nl("Antialias=on")
                nl("Output_File_Name='%s'" % tmpPovOutputFile)
                
                fh = open(povIniFile, "w")
                fh.write("\n".join(lines))
                fh.close()
                
                # run povray
                command = "%s '%s'" % (povray, povIniFile)
                resultQ = Queue.Queue()
                
                # thread
                thread = threading.Thread(target=utilities.runSubprocessInThread, args=(command, resultQ))
                thread.start()
                
                # check queue for result
                while thread.isAlive():
                    thread.join(1)
                    QtGui.QApplication.processEvents()
                
                # result
                try:
                    output, stderr, status = resultQ.get(timeout=1)
                except Queue.Empty:
                    logger.error("Could not get result from POV-Ray thread!")
                    return None
                    
                if status:
                    logging.error("POV-Ray failed: out: %s", output)
                    logging.error("POV-Ray failed: err: %s", stderr)
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
            
            # rename tmp image file to where it should be
            try:
                shutil.move(os.path.join(self.mainWindow.tmpDirectory, tmpPovOutputFile), filename)
            except:
                print "ERROR COPYING POV FILE", sys.exc_info()
                        
            # remove image files
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "renderer%d_image.pov" % renIndex))
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "renderer%d_header.pov" % renIndex))
            os.unlink(os.path.join(self.mainWindow.tmpDirectory, "renderer%d_image.ini" % renIndex))
        
        if not os.path.exists(filename):
            self.logger.error("Something went wrong with save image")
            return None
        
        elif renderType == "POV" and overlay:
            self.overlayImage(filename)
            
            # overlay image in separate thread
#             thread = threading.Thread(target=self.overlayImage, args=(filename,))
#             thread.start()
#             while thread.isAlive():
#                 thread.join(0.1)
#                 QtGui.QApplication.processEvents()
        
        return filename
    
    def overlayImage(self, filename):
        """
        Overlay the image with on screen info.
        
        """
        overlayTime = time.time()
        
        # local refs
        ren = self.ren
        renWinInteract = self.renWinInteract
        renIndex = self.parent.rendererIndex
        
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
            overlayFilePrefix = os.path.join(self.mainWindow.tmpDirectory, "renderer%d_overlay" % renIndex)
            
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
        if self.parent.blackBackground:
            R = G = B = 0
        else:
            R = G = B = 255
        
        xmax = 0
        ymax = 0
        xmin = 1000
        ymin = 1000    
        for i in xrange(i0, i1):
            for j in xrange(j0, j1):
                r,g,b = im.getpixel((i, j))
                
                if r != R and g != G and b != B:
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
        lattice = self.getInputState()
        settings = self.mainWindow.preferences.povrayForm
        
        a = [0]*3
        b = [0]*3 
        b[0] = - lattice.cellDims[0]
        b[1] = lattice.cellDims[1]
        b[2] = lattice.cellDims[2]
        
        if self.parent.blackBackground:
            R = G = B = 1
        else:
            R = G = B = 0
        
        filehandle.write("#declare R = %f;\n" % settings.cellFrameRadius)
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
        filehandle.write("    texture { pigment { color rgb <%f,%f,%f> }\n" % (R, G, B))
        filehandle.write("              finish { diffuse 0.9 phong 1 } } }\n")
        filehandle.write("object{myObject}\n")
    
    def writePOVRAYHeader(self, filehandle):
        """
        Write POV-Ray header file.
        
        """
        settings = self.mainWindow.preferences.povrayForm
        
        focalPoint = self.camera.GetFocalPoint()
        campos = self.camera.GetPosition()
        viewup = self.camera.GetViewUp()
        angle = settings.viewAngle
        if settings.shadowless:
            shadowless = "shadowless "
        else:
            shadowless = ""
        
        if self.parent.blackBackground:
            R = G = B = 0
        else:
            R = G = B = 1
        
        string = "camera { perspective location <%f,%f,%f> look_at <%f,%f,%f> angle %f\n" % (- campos[0], campos[1], campos[2],
                                                                                             - focalPoint[0], focalPoint[1], focalPoint[2],
                                                                                             angle)
        string += "sky <%f,%f,%f> }\n" % (- viewup[0], viewup[1], viewup[2])
        string += "light_source { <%f,%f,%f> color rgb <1,1,1> %s}\n" % (- campos[0], campos[1], campos[2], shadowless)
        string += "background { color rgb <%f,%f,%f> }\n" % (R, G, B)
        
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

def povrayBond(pos, vector):
    """
    Return string for rendering bond in povray.
    
    """
    pass

################################################################################

class PovrayColouringOptions(object):
    """
    Dummy class for passing to thread (contains only what is required)
    
    """
    def __init__(self, colouringOptions):
        self.colourBy = colouringOptions.colourBy
        self.heightAxis = colouringOptions.heightAxis

################################################################################

class PovrayDisplayOptions(object):
    """
    Dummy class for passing to thread (contains only what is required)
    
    """
    def __init__(self, displayOptions):
        self.atomScaleFactor = displayOptions.atomScaleFactor

################################################################################

class PovRayAtomsWriter(QtCore.QObject):
    """
    Write POV-Ray atoms to file
    
    """
    finished = QtCore.Signal(int, float, str)
    allDone = QtCore.Signal()
    
    def __init__(self, filename, visibleAtoms, lattice, scalarsDict, colouringOptions, displayOptions, lut, uniqueId):
        super(PovRayAtomsWriter, self).__init__()
        
        self.filename = filename
        self.visibleAtoms = visibleAtoms
        self.lattice = lattice
        self.scalarsDict = scalarsDict
        self.colouringOptions = PovrayColouringOptions(colouringOptions)
        self.lut = lut
        self.displayOptions = PovrayDisplayOptions(displayOptions)
        self.uniqueId = uniqueId
    
    def run(self):
        """
        Write atoms to file
        
        """
        povtime = time.time()
        
        # local refs
        visibleAtoms = self.visibleAtoms
        lattice = self.lattice
        scalarsDict = self.scalarsDict
        colouringOptions = self.colouringOptions
        displayOptions = self.displayOptions
        lut = self.lut
        specie = lattice.specie
        pos = lattice.pos
        charge = lattice.charge
        
        # scalars type
        scalarsType = getScalarsType(colouringOptions)
        
        # open pov file
        fpov = open(self.filename, "w")
        
        # loop over atoms
        for i, index in enumerate(visibleAtoms):
            # specie index
            specInd = specie[index]
            
            # scalar val
            if scalarsType == 0:
                scalar = specInd
            elif scalarsType == 1:
                scalar = pos[3*index+colouringOptions.heightAxis]
            elif scalarsType == 4:
                scalar = charge[index]
            else:
                if colouringOptions.colourBy.startswith("Lattice: "):
                    scalar = lattice.scalarsDict[colouringOptions.colourBy[9:]][i]
                else:
                    scalar = scalarsDict[colouringOptions.colourBy][i]
            
            # colour for povray file
            rgb = np.empty(3, np.float64)
            lut.GetColor(scalar, rgb)
             
            # povray atom
            fpov.write(povrayAtom(pos[3*index:3*index+3], lattice.specieCovalentRadius[specInd] * displayOptions.atomScaleFactor, rgb))
        
        fpov.close()
        povtime = time.time() - povtime
        
        # emit finished signal
        self.finished.emit(0, povtime, str(self.uniqueId))
        self.allDone.emit()

################################################################################

class RGBCallBackClass2(object):
    def __init__(self, lut):
        self.lut = lut
    
    def getRGB(self, scalar):
        rgb = np.empty(3, np.float64)
        self.lut.GetColor(scalar, rgb)
        return rgb

def writePovrayAtoms(filename, visibleAtoms, lattice, scalarsDict, colouringOptions, lut):
    """
    Write pov-ray atoms to file.
    
    """
    # scalars type
    scalarsType = getScalarsType(colouringOptions)
    if scalarsType == 5:
        if colouringOptions.colourBy.startswith("Lattice: "):
            scalarsArray = lattice.scalarsDict[colouringOptions.colourBy[9:]]
        else:
            scalarsArray = scalarsDict[colouringOptions.colourBy]
    
    else:
        scalarsArray = np.array([], dtype=np.float64)
    
    # rgb callback
    rgbcalc = utils.RGBCallBackClass(lut)
    
    # call C routine to write atoms to file
    output_c.writePOVRAYAtoms(filename, visibleAtoms, lattice.specie, lattice.pos, lattice.specieCovalentRadius, 
                              lattice.charge, scalarsArray, scalarsType, colouringOptions.heightAxis, rgbcalc.cfunc)

################################################################################
def getActorsForFilteredSystem(visibleAtoms, mainWindow, actorsDict, colouringOptions, povFileName, scalarsDict, displayOptions, 
                               pipelinePage, povFinishedSlot, vectorsDict, vectorsOptions, NVisibleForRes=None, sequencer=False):
    """
    Make the actors for the filtered system
    
    """
    logger = logging.getLogger(__name__)
    logger.debug("Getting actors for filtered system")
    logger.debug("  Colour by: '%s'", colouringOptions.colourBy)
    
    getActorsTime = time.time()
    
    NVisible = len(visibleAtoms)
    
    if NVisibleForRes is None:
        NVisibleForRes = NVisible
    
    # resolution
    res = setRes(NVisibleForRes, displayOptions)
    
    # lattice
    lattice = pipelinePage.inputState
    
    # make LUT
    lut = setupLUT(lattice.specieList, lattice.specieRGB, colouringOptions)
    
    # number of species
    NSpecies = len(lattice.specieList)
    
    # call method to make POV-Ray file (with callback to get rgb?)
    # use class similar to Allocator...
    
    #NOTE: if this is slow we should move this into a thread
    #      set var when done and only render after var is set...
    #      probably run thread from filterer
    
    # povray file and writer
    povtime = time.time()
    povFilePath = os.path.join(mainWindow.tmpDirectory, povFileName)
    uniqueId = uuid.uuid4()
    povAtomWriter = PovRayAtomsWriter(povFilePath, visibleAtoms, lattice, scalarsDict, colouringOptions, displayOptions, lut, uniqueId)
    
    # write pov atoms now if we're running sequencer, otherwise in separate thread
    logger.debug("Preparing to write POV-Ray atoms (sequencer=%s) (%s)", sequencer, uniqueId)
    if sequencer:
        povAtomWriter.run()
    
    else:
        # connect finished slot
        povAtomWriter.finished.connect(povFinishedSlot)
        povAtomWriter.allDone.connect(povAtomWriter.deleteLater)
    
        # create runner
        runnable = GenericRunnable(povAtomWriter)
        runnable.setAutoDelete(False)
        
        # add to threadpool
        QtCore.QThreadPool.globalInstance().start(runnable)
    
    povtime = time.time() - povtime
    
    if sequencer:
        povFinishedSlot(0, povtime, uniqueId)
    
    # make radius array
    radiusArray = _rendering.makeVisibleRadiusArray(visibleAtoms, lattice.specie, lattice.specieCovalentRadius)
    radiusArrayVTK = numpy_support.numpy_to_vtk(radiusArray, deep=1)
    radiusArrayVTK.SetName("radius")
    
    # scalar type
    scalarType = getScalarsType(colouringOptions)
    
    # scalars array
    if scalarType == 5:
        if colouringOptions.colourBy.startswith("Lattice: "):
            scalarsArray = lattice.scalarsDict[colouringOptions.colourBy[9:]]
        else:
            scalarsArray = scalarsDict[colouringOptions.colourBy]
    
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
        
        scalarsArray = _rendering.makeVisibleScalarArray(visibleAtoms, scalarsFull)
    
    scalarsArrayVTK = numpy_support.numpy_to_vtk(scalarsArray, deep=1)
    scalarsArrayVTK.SetName("colours")
    
    # make points array
    atomPoints = _rendering.makeVisiblePointsArray(visibleAtoms, lattice.pos)
    atomPointsVTK = vtk.vtkPoints()
    atomPointsVTK.SetData(numpy_support.numpy_to_vtk(atomPoints, deep=1))
    
    # poly data
    atomsPolyData = vtk.vtkPolyData()
    atomsPolyData.SetPoints(atomPointsVTK)
    atomsPolyData.GetPointData().AddArray(scalarsArrayVTK)
    atomsPolyData.GetPointData().SetScalars(radiusArrayVTK)
    
    # glyph source
    atomsGlyphSource = vtk.vtkSphereSource()
    atomsGlyphSource.SetPhiResolution(res)
    atomsGlyphSource.SetThetaResolution(res)
    atomsGlyphSource.SetRadius(1.0)
    
    # glyph
    atomsGlyph = vtk.vtkGlyph3D()
    if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        atomsGlyph.SetSource(atomsGlyphSource.GetOutput())
        atomsGlyph.SetInput(atomsPolyData)
    else:
        atomsGlyph.SetSourceConnection(atomsGlyphSource.GetOutputPort())
        atomsGlyph.SetInputData(atomsPolyData)
    atomsGlyph.SetScaleFactor(displayOptions.atomScaleFactor)
    atomsGlyph.SetScaleModeToScaleByScalar()
    atomsGlyph.ClampingOff()
      
    # mapper
    atomsMapper = vtk.vtkPolyDataMapper()
    atomsMapper.SetInputConnection(atomsGlyph.GetOutputPort())
    atomsMapper.SetLookupTable(lut)
    atomsMapper.SetScalarModeToUsePointFieldData()
    atomsMapper.SelectColorArray("colours")
    setMapperScalarRange(atomsMapper, colouringOptions, NSpecies)
    
    # glyph mapper
#     glyphMapper = vtk.vtkGlyph3DMapper()
#     if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
#         glyphMapper.SetInputConnection(atomsPolyData.GetProducerPort())
#     else:
#         glyphMapper.SetInputData(atomsPolyData)
#     glyphMapper.SetSourceConnection(atomsGlyphSource.GetOutputPort())
#     glyphMapper.SetScaleFactor(displayOptions.atomScaleFactor)
#     glyphMapper.SetScaleModeToScaleByMagnitude()
#     glyphMapper.ClampingOff()
#     glyphMapper.SetLookupTable(lut)
#     glyphMapper.SetScalarModeToUsePointFieldData()
#     glyphMapper.SelectColorArray("colours")
#     setMapperScalarRange(glyphMapper, colouringOptions, NSpecies)
#     atomsMapper = glyphMapper
    
    # actor
    atomsActor = vtk.vtkActor()
    atomsActor.SetMapper(atomsMapper)
    atomsActor.GetProperty().SetSpecular(0.4)
    atomsActor.GetProperty().SetSpecularPower(50)
    actorsDict["Atoms"] = utils.ActorObject(atomsActor)
    
    # check for vectors
    vectorsName = vectorsOptions.selectedVectorsName
    if vectorsName is not None and vectorsName not in vectorsDict:
        logger.warning("Skipping adding vectors because could not find array: '%s'", vectorsName)
    
    elif vectorsName is not None:
        logger.debug("Adding arrows for vector data: '%s'", vectorsName)
        
        # vectors
        npvects = vectorsDict[vectorsName]
        if vectorsOptions.vectorNormalise:
            npvects = vectorslib.normalise(npvects)
        vects = numpy_support.numpy_to_vtk(npvects, deep=1)
        vects.SetName("vectors")
        
        # polydata
        arrowPolyData = vtk.vtkPolyData()
        arrowPolyData.SetPoints(atomPointsVTK)
        arrowPolyData.GetPointData().SetScalars(scalarsArrayVTK)
        arrowPolyData.GetPointData().SetVectors(vects)
    
        # arrow source
        arrowSource = vtk.vtkArrowSource()
        arrowSource.SetShaftResolution(vectorsOptions.vectorResolution)
        arrowSource.SetTipResolution(vectorsOptions.vectorResolution)
        arrowSource.Update()
        
        # glyph
#         arrowGlyph = vtk.vtkGlyph3D()
#         arrowGlyph.OrientOn()
#         if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
#             arrowGlyph.SetSource(arrowSource.GetOutput())
#             arrowGlyph.SetInput(arrowPolyData)
#         else:
#             arrowGlyph.SetSourceConnection(arrowSource.GetOutputPort())
#             arrowGlyph.SetInputData(arrowPolyData)
#         arrowGlyph.SetScaleModeToScaleByVector()
#         arrowGlyph.SetVectorModeToUseVector()
#         arrowGlyph.SetColorModeToColorByScalar()
#          
#         # mapper
#         arrowMapper = vtk.vtkPolyDataMapper()
#         arrowMapper.SetInputConnection(arrowGlyph.GetOutputPort())
#         arrowMapper.SetLookupTable(lut)
#         setMapperScalarRange(arrowMapper, colouringOptions, NSpecies)
        
        # glyph mapper
        arrowGlyph = vtk.vtkGlyph3DMapper()
        arrowGlyph.OrientOn()
        if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
            arrowGlyph.SetInputConnection(arrowPolyData.GetProducerPort())
        else:
            arrowGlyph.SetInputData(arrowPolyData)
        arrowGlyph.SetSourceConnection(arrowSource.GetOutputPort())
        arrowGlyph.SetScaleModeToScaleByMagnitude()
        arrowGlyph.SetScaleArray("vectors")
        arrowGlyph.SetScalarModeToUsePointFieldData()
        arrowGlyph.SelectColorArray("colours")
        arrowGlyph.SetScaleFactor(vectorsOptions.vectorScaleFactor)
        arrowMapper = arrowGlyph
        arrowMapper.SetLookupTable(lut)
        setMapperScalarRange(arrowMapper, colouringOptions, NSpecies)
    
        # actor
        arrowActor = vtk.vtkActor()
        arrowActor.SetMapper(arrowMapper)
        actorsDict["Vectors"] = utils.ActorObject(arrowActor)
    
    # scalar bar
    scalarBar_white = None
    scalarBar_black = None
    if colouringOptions.colourBy != "Specie" and colouringOptions.colourBy != "Solid colour":
        scalarBar_white = makeScalarBar(lut, colouringOptions, (0, 0, 0))
        scalarBar_black = makeScalarBar(lut, colouringOptions, (1, 1, 1))
    
    # visible specie count
    visSpecCount = _rendering.countVisibleBySpecie(visibleAtoms, NSpecies, lattice.specie)
    
    getActorsTime = time.time() - getActorsTime
    logger.debug("Get actors time: %f s", getActorsTime)
#     logger.debug("  Set points time (new): %f s", povtime + setPointsTime2)
#     logger.debug("    Build spec arrays time: %f s", setPointsTime2)
#     logger.debug("    Write pov atoms time: %f s", povtime)
#     for i, t1 in enumerate(t1s):
#         logger.debug("  Make actors time (%d): %f s", i, t1)
    
    return scalarBar_white, scalarBar_black, visSpecCount

################################################################################
def writePovrayDefects(filename, vacancies, interstitials, antisites, onAntisites, 
                       settings, mainWindow, displayOptions, splitInterstitials, pipelinePage):
    """
    Write defects to povray file.
    
    """
    povfile = os.path.join(mainWindow.tmpDirectory, filename)
    
    inputLattice = pipelinePage.inputState
    refLattice = pipelinePage.refState
    
    output_c.writePOVRAYDefects(povfile, vacancies, interstitials, antisites, onAntisites, inputLattice.specie, inputLattice.pos,
                                refLattice.specie, refLattice.pos, inputLattice.specieRGB, inputLattice.specieCovalentRadius * displayOptions.atomScaleFactor,
                                refLattice.specieRGB, refLattice.specieCovalentRadius * displayOptions.atomScaleFactor, splitInterstitials)


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
def getActorsForFilteredDefects(interstitials, vacancies, antisites, onAntisites, splitInterstitials, actorsDict, 
                                colouringOptions, filterSettings, displayOptions, pipelinePage):
    
    actorsDictLocal = {}
    
    NInt = len(interstitials)
    NVac = len(vacancies)
    NAnt = len(antisites)
    NSplit = len(splitInterstitials) / 3
    NDef = NInt + NVac + NAnt + len(splitInterstitials)
    
    # resolution
    res = setRes(NDef, displayOptions)
    
    inputLattice = pipelinePage.inputState
    refLattice = pipelinePage.refState
    
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
    if NInt:
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
            
            # scalar
            scalar = getScalar(colouringOptions, inputLattice, index)
            
            intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd].InsertNextValue(scalar)
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            intsPolyData = vtk.vtkPolyData()
            intsPolyData.SetPoints(intPointsList[i])
            intsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            intsGlyphSource = vtk.vtkSphereSource()
            intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            intsGlyphSource.SetPhiResolution(res)
            intsGlyphSource.SetThetaResolution(res)
            
            intsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                intsGlyph.SetSource(intsGlyphSource.GetOutput())
                intsGlyph.SetInput(intsPolyData)
            else:
                intsGlyph.SetSourceConnection(intsGlyphSource.GetOutputPort())
                intsGlyph.SetInputData(intsPolyData)
            intsGlyph.SetScaleFactor(1.0)
            intsGlyph.SetScaleModeToDataScalingOff()
            
            intsMapper = vtk.vtkPolyDataMapper()
            intsMapper.SetInputConnection(intsGlyph.GetOutputPort())
            intsMapper.SetLookupTable(lut)
            setMapperScalarRange(intsMapper, colouringOptions, NSpecies)
            
            intsActor = vtk.vtkActor()
            intsActor.SetMapper(intsMapper)
            
            actorsDictLocal["Interstitials ({0})".format(inputLattice.specieList[i])] = utils.ActorObject(intsActor)
    
    #----------------------------------------#
    # split interstitial atoms next
    #----------------------------------------#
    if NSplit:
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
            
            scalar = getScalar(colouringOptions, inputLattice, index)
            
            intPointsList[specInd1].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd1].InsertNextValue(scalar)
            
            # second
            index = splitInterstitials[3*i+2]
            specInd2 = spec[index]
            
            scalar = getScalar(colouringOptions, inputLattice, index)
            
            intPointsList[specInd2].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd2].InsertNextValue(scalar)
            
            # counter
            splitSpecCount[specInd1][specInd2] += 1
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            intsPolyData = vtk.vtkPolyData()
            intsPolyData.SetPoints(intPointsList[i])
            intsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            intsGlyphSource = vtk.vtkSphereSource()
            intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            intsGlyphSource.SetPhiResolution(res)
            intsGlyphSource.SetThetaResolution(res)
            
            intsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                intsGlyph.SetSource(intsGlyphSource.GetOutput())
                intsGlyph.SetInput(intsPolyData)
            else:
                intsGlyph.SetSourceConnection(intsGlyphSource.GetOutputPort())
                intsGlyph.SetInputData(intsPolyData)
            intsGlyph.SetScaleFactor(1.0)
            intsGlyph.SetScaleModeToDataScalingOff()
            
            intsMapper = vtk.vtkPolyDataMapper()
            intsMapper.SetInputConnection(intsGlyph.GetOutputPort())
            intsMapper.SetLookupTable(lut)
            setMapperScalarRange(intsMapper, colouringOptions, NSpecies)
            
            intsActor = vtk.vtkActor()
            intsActor.SetMapper(intsMapper)
            
            actorsDictLocal["Split ints ({0})".format(inputLattice.specieList[i])] = utils.ActorObject(intsActor)
        
        #----------------------------------------#
        # split interstitial vacs next
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
            
            scalar = getScalar(colouringOptions, refLattice, index)
            
            intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd].InsertNextValue(scalar)
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            vacsPolyData = vtk.vtkPolyData()
            vacsPolyData.SetPoints(intPointsList[i])
            vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            vacsGlyphSource = vtk.vtkCubeSource()
            scaleVacs = 2.0 * filterSettings.vacScaleSize
            vacsGlyphSource.SetXLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            vacsGlyphSource.SetYLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            vacsGlyphSource.SetZLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            
            vacsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
                vacsGlyph.SetInput(vacsPolyData)
            else:
                vacsGlyph.SetSourceConnection(vacsGlyphSource.GetOutputPort())
                vacsGlyph.SetInputData(vacsPolyData)
            vacsGlyph.SetScaleFactor(1.0)
            vacsGlyph.SetScaleModeToDataScalingOff()
            
            vacsMapper = vtk.vtkPolyDataMapper()
            vacsMapper.SetInputConnection(vacsGlyph.GetOutputPort())
            vacsMapper.SetLookupTable(lut)
            setMapperScalarRange(vacsMapper, colouringOptions, NSpecies)
            
            vacsActor = vtk.vtkActor()
            vacsActor.SetMapper(vacsMapper)
            vacsActor.GetProperty().SetSpecular(filterSettings.vacSpecular)
            vacsActor.GetProperty().SetSpecularPower(filterSettings.vacSpecularPower)
            vacsActor.GetProperty().SetOpacity(filterSettings.vacOpacity)
            
            actorsDictLocal["Split vacs ({0})".format(refLattice.specieList[i])] = utils.ActorObject(vacsActor)
    
    #----------------------------------------#
    # antisites occupying atom
    #----------------------------------------#
    if NAnt:
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
        for i in xrange(NAnt):
            index = onAntisites[i]
            specInd = spec[index]
            
            scalar = getScalar(colouringOptions, inputLattice, index)
            
            intScalarsList[specInd].InsertNextValue(scalar)
            intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            intsPolyData = vtk.vtkPolyData()
            intsPolyData.SetPoints(intPointsList[i])
            intsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            intsGlyphSource = vtk.vtkSphereSource()
            intsGlyphSource.SetRadius(inputLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            intsGlyphSource.SetPhiResolution(res)
            intsGlyphSource.SetThetaResolution(res)
            
            intsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                intsGlyph.SetSource(intsGlyphSource.GetOutput())
                intsGlyph.SetInput(intsPolyData)
            else:
                intsGlyph.SetSourceConnection(intsGlyphSource.GetOutputPort())
                intsGlyph.SetInputData(intsPolyData)
            intsGlyph.SetScaleFactor(1.0)
            intsGlyph.SetScaleModeToDataScalingOff()
            
            intsMapper = vtk.vtkPolyDataMapper()
            intsMapper.SetInputConnection(intsGlyph.GetOutputPort())
            intsMapper.SetLookupTable(lut)
            setMapperScalarRange(intsMapper, colouringOptions, NSpecies)
            
            intsActor = vtk.vtkActor()
            intsActor.SetMapper(intsMapper)
            
            actorsDictLocal["Antisites occupying ({0})".format(inputLattice.specieList[i])] = utils.ActorObject(intsActor)
    
    #----------------------------------------#
    # vacancies
    #----------------------------------------#
    if NVac:
        NSpecies = len(refLattice.specieList)
        intPointsList = []
        intScalarsList = []
        for i in xrange(NSpecies):
            intPointsList.append(vtk.vtkPoints())
            intScalarsList.append(vtk.vtkFloatArray())
        
        # make LUT
        lut = setupLUT(refLattice.specieList, refLattice.specieRGB, colouringOptions)
        
        # loop over vacancies, settings points
        pos = refLattice.pos
        spec = refLattice.specie
        for i in xrange(NVac):
            index = vacancies[i]
            specInd = spec[index]
            
            vacSpecCount[specInd] += 1
            
            scalar = getScalar(colouringOptions, refLattice, index)
            
            intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd].InsertNextValue(scalar)
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            vacsPolyData = vtk.vtkPolyData()
            vacsPolyData.SetPoints(intPointsList[i])
            vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            vacsGlyphSource = vtk.vtkCubeSource()
            scaleVacs = 2.0 * filterSettings.vacScaleSize
            vacsGlyphSource.SetXLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            vacsGlyphSource.SetYLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            vacsGlyphSource.SetZLength(scaleVacs * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            
            vacsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
                vacsGlyph.SetInput(vacsPolyData)
            else:
                vacsGlyph.SetSourceConnection(vacsGlyphSource.GetOutputPort())
                vacsGlyph.SetInputData(vacsPolyData)
            vacsGlyph.SetScaleFactor(1.0)
            vacsGlyph.SetScaleModeToDataScalingOff()
            
            vacsMapper = vtk.vtkPolyDataMapper()
            vacsMapper.SetInputConnection(vacsGlyph.GetOutputPort())
            vacsMapper.SetLookupTable(lut)
            setMapperScalarRange(vacsMapper, colouringOptions, NSpecies)
            
            vacsActor = vtk.vtkActor()
            vacsActor.SetMapper(vacsMapper)
            vacsActor.GetProperty().SetSpecular(filterSettings.vacSpecular)
            vacsActor.GetProperty().SetSpecularPower(filterSettings.vacSpecularPower)
            vacsActor.GetProperty().SetOpacity(filterSettings.vacOpacity)
            
            actorsDictLocal["Vacancies ({0})".format(refLattice.specieList[i])] = utils.ActorObject(vacsActor)
    
    #----------------------------------------#
    # antisites
    #----------------------------------------#
    if NAnt:
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
            
            scalar = getScalar(colouringOptions, refLattice, index)
            
            intPointsList[specInd].InsertNextPoint(pos[3*index], pos[3*index+1], pos[3*index+2])
            intScalarsList[specInd].InsertNextValue(scalar)
        
        # now loop over species making actors
        for i in xrange(NSpecies):
            
            vacsPolyData = vtk.vtkPolyData()
            vacsPolyData.SetPoints(intPointsList[i])
            vacsPolyData.GetPointData().SetScalars(intScalarsList[i])
            
            cubeGlyphSource = vtk.vtkCubeSource()
            cubeGlyphSource.SetXLength(2.0 * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            cubeGlyphSource.SetYLength(2.0 * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            cubeGlyphSource.SetZLength(2.0 * refLattice.specieCovalentRadius[i] * displayOptions.atomScaleFactor)
            edges = vtk.vtkExtractEdges()
            edges.SetInputConnection(cubeGlyphSource.GetOutputPort())
            vacsGlyphSource = vtk.vtkTubeFilter()
            vacsGlyphSource.SetInputConnection(edges.GetOutputPort())
            vacsGlyphSource.SetRadius(0.1)
            vacsGlyphSource.SetNumberOfSides(5)
            vacsGlyphSource.UseDefaultNormalOn()
            vacsGlyphSource.SetDefaultNormal(.577, .577, .577)
            
            vacsGlyph = vtk.vtkGlyph3D()
            if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
                vacsGlyph.SetSource(vacsGlyphSource.GetOutput())
                vacsGlyph.SetInput(vacsPolyData)
            else:
                vacsGlyph.SetSourceConnection(vacsGlyphSource.GetOutputPort())
                vacsGlyph.SetInputData(vacsPolyData)
            vacsGlyph.SetScaleFactor(1.0)
            vacsGlyph.SetScaleModeToDataScalingOff()
            
            vacsMapper = vtk.vtkPolyDataMapper()
            vacsMapper.SetInputConnection(vacsGlyph.GetOutputPort())
            vacsMapper.SetLookupTable(lut)
            setMapperScalarRange(vacsMapper, colouringOptions, NSpecies)
            
            vacsActor = vtk.vtkActor()
            vacsActor.SetMapper(vacsMapper)
            
            actorsDictLocal["Antisites frames ({0})".format(refLattice.specieList[i])] = utils.ActorObject(vacsActor)
    
    actorsDict["Defects"] = actorsDictLocal
    
    # scalar bar
    scalarBar_white = None
    scalarBar_black = None
    if colouringOptions.colourBy != "Specie" and colouringOptions.colourBy != "Solid colour":
        scalarBar_white = makeScalarBar(lut, colouringOptions, (0, 0, 0))
        scalarBar_black = makeScalarBar(lut, colouringOptions, (1, 1, 1))
    
    return vacSpecCount, intSpecCount, antSpecCount, splitSpecCount, scalarBar_white, scalarBar_black


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
def getActorsForHullFacets(facets, pos, mainWindow, actorsDict, settings, caller):
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
    if vtk.vtkVersion.GetVTKMajorVersion() <= 5:
        mapper.SetInput(trianglePolyData)
    else:
        mapper.SetInputData(trianglePolyData)
    
    # actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(settings.hullOpacity)
    actor.GetProperty().SetColor(settings.hullCol[0], settings.hullCol[1], settings.hullCol[2])
    
    d = {}
    
    actorsDict["Hulls - ({0})".format(caller)] = utils.ActorObject(actor)
