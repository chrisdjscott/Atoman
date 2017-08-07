
"""
Module for rendering

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import sys
import shutil
import logging
import time
import threading
import six.moves.queue

import vtk
from PIL import Image
from PySide2 import QtCore, QtWidgets


from ..visutils import utilities
from . import cell
from . import axes
from six.moves import range


class Renderer(object):
    def __init__(self, parent):
        
        self.parent = parent
        self.mainWindow = self.parent.mainWindow
        self.ren = self.parent.vtkRen
        self.renWinInteract = self.parent.vtkRenWinInteract
        self.renWin = self.parent.vtkRenWin
        
        self.logger = logging.getLogger(__name__ + ".Renderer")
        
        # is the interactor initialised
        self.init = False
        
        # setup stuff
        self.camera = self.ren.GetActiveCamera()
        
        # lattice frame
        self.latticeFrame = cell.CellOutline(self.ren)
        
        # axes
        self.axes = axes.Axes(self.renWinInteract)
        
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
        self.axes.add()
        
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
        campos = [0] * 3
        if dims[1] > dims[2]:
            campos[0] = -3.0 * dims[1]
        else:
            campos[0] = -3.0 * dims[2]
        campos[1] = 0.5 * dims[1]
        campos[2] = 0.5 * dims[2]
        
        focpnt = [0] * 3
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
        self.axes.toggle()
        self.reinit()
    
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
        progDialog = QtWidgets.QProgressDialog("Running rotator...", "Cancel", 0, NRotations)
        progDialog.setWindowModality(QtCore.Qt.WindowModal)
        progDialog.setWindowTitle("Progress")
        progDialog.setValue(0)
        
        progDialog.show()
        QtWidgets.QApplication.processEvents()
        
        # main loop
        try:
            status = 0
            for i in range(NRotations):
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
                
                QtWidgets.QApplication.processEvents()
                
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
            self.logger.debug("Rendering using POV-Ray")

            # which renderer and pipeline
            renIndex = self.parent.rendererIndex
            pipelineIndex = self.parent.currentPipelineIndex
            
            self.logger.debug("Renderer %d; Pipeline %d", renIndex, pipelineIndex)
            
            # header file
            povfile = os.path.join(self.mainWindow.tmpDirectory, "renderer%d.pov" % renIndex)
            self.logger.debug("Povray file: '%s'", povfile)
            with open(povfile, "w") as fh:
                # first write the header (camera info etc.)
                self.writePOVRAYHeader(fh)
                
                # write cell frame if visible
                if self.latticeFrame.visible:
                    self.writePOVRAYCellFrame(fh)
                
                # TODO: write axes if visible
            
            # write povray files for active renderers
            self.logger.debug("Writing renderer povray data")
            filterLists = self.parent.getFilterLists()
            for flist in filterLists:
                if flist.visible:
                    for rend in flist.renderer.renderers():
                        rend.writePovray(povfile)
            
            # POV-Ray settings
            settings = self.mainWindow.preferences.povrayForm
            overlay = settings.overlayImage
            
            # then join filter list files
            _cwd = os.getcwd()
            os.chdir(self.mainWindow.tmpDirectory)
            try:
                # create povray ini file
                povIniFile = "renderer%d_image.ini" % renIndex
                tmpPovOutputFile = "renderer%d_image.%s" % (renIndex, imageFormat)
                with open(povIniFile, "w") as fh:
                    fh.write("; Atoman auto-generated POV-Ray INI file\n")
                    fh.write("Input_File_Name='%s'\n" % os.path.basename(povfile))
                    fh.write("Width=%d\n" % settings.HRes)
                    fh.write("Height=%d\n" % settings.VRes)
                    fh.write("Display=off\n")
                    fh.write("Antialias=on\n")
                    fh.write("Output_File_Name='%s'\n" % tmpPovOutputFile)
                
                # run povray
                command = "%s '%s'" % (povray, povIniFile)
                resultQ = six.moves.queue.Queue()
                
                # run in thread
                thread = threading.Thread(target=utilities.runSubprocessInThread, args=(command, resultQ))
                thread.start()
                while thread.isAlive():
                    thread.join(1)
                    QtWidgets.QApplication.processEvents()
                
                # result
                try:
                    output, stderr, status = resultQ.get(timeout=1)
                except six.moves.queue.Empty:
                    logger.error("Could not get result from POV-Ray thread!")
                    return None
                    
                if status:
                    logging.error("POV-Ray failed: out: %s", output)
                    logging.error("POV-Ray failed: err: %s", stderr)
                    return None
                
            finally:
                os.chdir(_cwd)
            
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
                print("ERROR COPYING POV FILE", sys.exc_info())
                        
            # remove image files
            # os.unlink(povfile)
            # os.unlink(os.path.join(self.mainWindow.tmpDirectory, povIniFile))
        
        if not os.path.exists(filename):
            self.logger.error("Something went wrong with save image")
            return None
        
        elif renderType == "POV" and overlay:
            self.overlayImage(filename)
        
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
        newCamPos = [0] * 3
        newCamPos[0] = origCamPos[0] * 100000
        newCamPos[1] = origCamPos[1] * 100000
        newCamPos[2] = origCamPos[2] * 100000
        camera.SetPosition(newCamPos)
        
        # also hide axes if visible
        if self.axes.isEnabled():
            axesHidden = True
            self.toggleAxes()
        else:
            axesHidden = False
        
        try:
            # save image
            overlayFilePrefix = os.path.join(self.mainWindow.tmpDirectory, "renderer%d_overlay" % renIndex)
            overlayFile = self.saveImage("VTK", "jpg", overlayFilePrefix, False)
            if not os.path.exists(overlayFile):
                print("WARNING: overlay file does not exist: %s" % overlayFile)
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
                    newregiondimx = int(povim.size[0] * 0.8)
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
            
            # show axes again
            if axesHidden:
                self.toggleAxes()
        
        overlayTime = time.time() - overlayTime
        self.logger.debug("Overlay time: %f s", overlayTime)
        
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
        for i in range(i0, i1):
            for j in range(j0, j1):
                r, g, b = im.getpixel((i, j))
                
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
    
    def writePOVRAYCellFrame(self, fh):
        """
        Write cell frame.
        
        """
        lattice = self.getInputState()
        settings = self.mainWindow.preferences.povrayForm
        
        a = [0] * 3
        b = [0] * 3
        b[0] = - lattice.cellDims[0]
        b[1] = lattice.cellDims[1]
        b[2] = lattice.cellDims[2]
        
        if self.parent.blackBackground:
            rval = gval = bval = 1
        else:
            rval = gval = bval = 0
        
        fh.write("#declare R = %f;\n" % settings.cellFrameRadius)
        fh.write("#declare cellObject = union {\n")
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (a[0], a[1], a[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (b[0], a[1], a[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (a[0], a[1], b[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (b[0], a[1], b[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (a[0], b[1], a[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (b[0], b[1], a[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (a[0], b[1], b[2]))
        fh.write("  sphere { <%f,%f,%f>, R }\n" % (b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{1},{2}>, <{3},{1},{2}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{1},{5}>, <{3},{1},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{4},{2}>, <{3},{4},{2}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{4},{5}>, <{3},{4},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{1},{2}>, <{0},{4},{2}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{1},{5}>, <{0},{4},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{3},{1},{2}>, <{3},{4},{2}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{3},{1},{5}>, <{3},{4},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{1},{2}>, <{0},{1},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{0},{4},{2}>, <{0},{4},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{3},{1},{2}>, <{3},{1},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  cylinder {{ <{3},{4},{2}>, <{3},{4},{5}>, R }}\n".format(a[0], a[1], a[2], b[0], b[1], b[2]))
        fh.write("  texture { pigment { color rgb <%f,%f,%f> }\n" % (rval, gval, bval))
        fh.write("            finish { diffuse 0.9 phong 1 } } }\n")
        fh.write("object{cellObject}\n")
    
    def writePOVRAYHeader(self, fh):
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
            rval = gval = bval = 0
        else:
            rval = gval = bval = 1
        
        fh.write("camera { perspective location <%f,%f,%f>\n" % (- campos[0], campos[1], campos[2]))
        fh.write("         look_at <%f,%f,%f>\n" % (- focalPoint[0], focalPoint[1], focalPoint[2]))
        fh.write("         angle %f\n" % angle)
        fh.write("         sky <%f,%f,%f> }\n" % (- viewup[0], viewup[1], viewup[2]))
        fh.write("light_source { <%f,%f,%f> color rgb <1,1,1> %s }\n" % (- campos[0], campos[1], campos[2], shadowless))
        fh.write("background { color rgb <%f,%f,%f> }\n" % (rval, gval, bval))
