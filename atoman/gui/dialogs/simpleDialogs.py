# -*- coding: utf-8 -*-

"""
Additional dialogs.

@author: Chris Scott

"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import copy
import logging

from PySide2 import QtGui, QtCore, QtWidgets

import numpy as np
import math as math

from ...system.atoms import elements
from ...visutils.utilities import resourcePath, iconPath
from ...visutils import utilities


################################################################################

class CameraSettingsDialog(QtWidgets.QDialog):
    """
    Camera settings dialog

    """
    def __init__(self, parent, renderer):
        super(CameraSettingsDialog, self).__init__(parent)

#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.renderer = renderer

        self.setModal(True)

        self.setWindowTitle("Camera settings")
        self.setWindowIcon(QtGui.QIcon(iconPath("cam.png")))

        self.contentLayout = QtWidgets.QFormLayout(self)
#         self.contentLayout.setAlignment(QtCore.Qt.AlignHCenter)

        # ini vals
        self.campos = list(renderer.camera.GetPosition())
        self.camfoc = list(renderer.camera.GetFocalPoint())
        self.camvup = list(renderer.camera.GetViewUp())

        self.camposbkup = copy.deepcopy(self.campos)
        self.camfocbkup = copy.deepcopy(self.camfoc)
        self.camvupbkup = copy.deepcopy(self.camvup)

        # cam pos
        self.camPosXSpin = QtWidgets.QDoubleSpinBox()
        self.camPosXSpin.setMinimum(-99999.0)
        self.camPosXSpin.setMaximum(99999.0)
        self.camPosXSpin.setValue(self.campos[0])
        self.camPosXSpin.valueChanged[float].connect(self.camxposChanged)

        self.camPosYSpin = QtWidgets.QDoubleSpinBox()
        self.camPosYSpin.setMinimum(-99999.0)
        self.camPosYSpin.setMaximum(99999.0)
        self.camPosYSpin.setValue(self.campos[1])
        self.camPosYSpin.valueChanged[float].connect(self.camyposChanged)

        self.camPosZSpin = QtWidgets.QDoubleSpinBox()
        self.camPosZSpin.setMinimum(-99999.0)
        self.camPosZSpin.setMaximum(99999.0)
        self.camPosZSpin.setValue(self.campos[2])
        self.camPosZSpin.valueChanged[float].connect(self.camzposChanged)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.camPosXSpin)
        row.addWidget(self.camPosYSpin)
        row.addWidget(self.camPosZSpin)
        self.contentLayout.addRow("Position", row)

        # cam focal point
        self.camFocXSpin = QtWidgets.QDoubleSpinBox()
        self.camFocXSpin.setMinimum(-99999.0)
        self.camFocXSpin.setMaximum(99999.0)
        self.camFocXSpin.setValue(self.camfoc[0])
        self.camFocXSpin.valueChanged[float].connect(self.camxfocChanged)

        self.camFocYSpin = QtWidgets.QDoubleSpinBox()
        self.camFocYSpin.setMinimum(-99999.0)
        self.camFocYSpin.setMaximum(99999.0)
        self.camFocYSpin.setValue(self.camfoc[1])
        self.camFocYSpin.valueChanged[float].connect(self.camyfocChanged)

        self.camFocZSpin = QtWidgets.QDoubleSpinBox()
        self.camFocZSpin.setMinimum(-99999.0)
        self.camFocZSpin.setMaximum(99999.0)
        self.camFocZSpin.setValue(self.camfoc[2])
        self.camFocZSpin.valueChanged[float].connect(self.camzfocChanged)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.camFocXSpin)
        row.addWidget(self.camFocYSpin)
        row.addWidget(self.camFocZSpin)
        self.contentLayout.addRow("Focal point", row)

        # cam view up
        self.camVupXSpin = QtWidgets.QDoubleSpinBox()
        self.camVupXSpin.setMinimum(-99999.0)
        self.camVupXSpin.setMaximum(99999.0)
        self.camVupXSpin.setValue(self.camvup[0])
        self.camVupXSpin.valueChanged[float].connect(self.camxvupChanged)

        self.camVupYSpin = QtWidgets.QDoubleSpinBox()
        self.camVupYSpin.setMinimum(-99999.0)
        self.camVupYSpin.setMaximum(99999.0)
        self.camVupYSpin.setValue(self.camvup[1])
        self.camVupYSpin.valueChanged[float].connect(self.camyvupChanged)

        self.camVupZSpin = QtWidgets.QDoubleSpinBox()
        self.camVupZSpin.setMinimum(-99999.0)
        self.camVupZSpin.setMaximum(99999.0)
        self.camVupZSpin.setValue(self.camvup[2])
        self.camVupZSpin.valueChanged[float].connect(self.camzvupChanged)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.camVupXSpin)
        row.addWidget(self.camVupYSpin)
        row.addWidget(self.camVupZSpin)
        self.contentLayout.addRow("View up", row)

        # button box
        self.buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close | QtWidgets.QDialogButtonBox.Reset)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.buttonBox.clicked.connect(self.buttonBoxClicked)
        self.contentLayout.addWidget(self.buttonBox)

    def buttonBoxClicked(self, button):
        """A button was clicked."""
        if self.buttonBox.button(QtWidgets.QDialogButtonBox.Reset) == button:
            self.resetChanges()

    def resetChanges(self):
        """
        Reset changes

        """
        self.campos = self.camposbkup
        self.camfoc = self.camfocbkup
        self.camvup = self.camvupbkup

        self.renderer.camera.SetPosition(self.campos)
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.camera.SetViewUp(self.camvup)

        self.renderer.reinit()

    def camxposChanged(self, val):
        """
        Cam x pos changed

        """
        self.campos[0] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()

    def camyposChanged(self, val):
        """
        Cam y pos changed

        """
        self.campos[1] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()

    def camzposChanged(self, val):
        """
        Cam z pos changed

        """
        self.campos[2] = val
        self.renderer.camera.SetPosition(self.campos)
        self.renderer.reinit()

    def camxfocChanged(self, val):
        """
        Cam x foc changed

        """
        self.camfoc[0] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()

    def camyfocChanged(self, val):
        """
        Cam y foc changed

        """
        self.camfoc[1] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()

    def camzfocChanged(self, val):
        """
        Cam z foc changed

        """
        self.camfoc[2] = val
        self.renderer.camera.SetFocalPoint(self.camfoc)
        self.renderer.reinit()

    def camxvupChanged(self, val):
        """
        Cam x foc changed

        """
        self.camvup[0] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()

    def camyvupChanged(self, val):
        """
        Cam y foc changed

        """
        self.camvup[1] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()

    def camzvupChanged(self, val):
        """
        Cam z foc changed

        """
        self.camvup[2] = val
        self.renderer.camera.SetViewUp(self.camvup)
        self.renderer.reinit()

################################################################################

class ImageViewer(QtWidgets.QDialog):
    """
    Image viewer.

    @author: Marc Robinson
    Rewritten by Chris Scott

    """
    def __init__(self, mainWindow, parent=None):
        super(ImageViewer, self).__init__(parent)

#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.parent = parent
        self.mainWindow = mainWindow

        self.setWindowTitle("Image Viewer:")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/applications-graphics.png")))

        # main layout
        dialogLayout = QtWidgets.QHBoxLayout()

        # initial dir
        startDir = os.getcwd()

        # dir model
        self.model = QtWidgets.QFileSystemModel()
        self.model.setFilter(QtCore.QDir.NoDot | QtCore.QDir.NoDotDot | QtCore.QDir.AllDirs | QtCore.QDir.Files)
        self.model.setNameFilters(["*.jpg", "*.tif","*.png","*.bmp"])
        self.model.setNameFilterDisables(0)
        self.model.setRootPath(startDir)

        # dir view
        self.view = QtWidgets.QTreeView(parent=self)
        self.view.setModel(self.model)
        self.view.clicked[QtCore.QModelIndex].connect(self.clicked)
        self.view.hideColumn(1)
        self.view.setRootIndex(self.model.index(startDir))
        self.view.setMinimumWidth(300)
        self.view.setColumnWidth(0, 150)
        self.view.setColumnWidth(2, 50)

        # add to main layout
        dialogLayout.addWidget(self.view)

        # image label
        self.imageLabel = QtWidgets.QLabel()

        column = QtWidgets.QWidget()
        columnLayout = QtWidgets.QVBoxLayout(column)
        columnLayout.setSpacing(0)
        columnLayout.setContentsMargins(0, 0, 0, 0)

        columnLayout.addWidget(self.imageLabel)

        # delete button
        deleteImageButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath("oxygen/edit-delete.png")), "Delete image")
        deleteImageButton.clicked.connect(self.deleteImage)
        deleteImageButton.setStatusTip("Delete image")
        deleteImageButton.setAutoDefault(False)
        columnLayout.addWidget(deleteImageButton)

        # add to layout
        dialogLayout.addWidget(column)

        # set layout
        self.setLayout(dialogLayout)

    def clicked(self, index):
        """
        File clicked.

        """
        self.showImage(self.model.filePath(index))

    def showImage(self, filename):
        """
        Show image.

        """
        try:
            image = QtGui.QImage(filename)
            self.imageLabel.setPixmap(QtGui.QPixmap.fromImage(image))
            self.setWindowTitle("Image Viewer: %s" % filename)

        except:
            print("ERROR: could not display image in Image Viewer")

    def deleteImage(self):
        """
        Delete image.

        """
        reply = QtWidgets.QMessageBox.question(self, "Message",
                                           "Delete file: %s?" % self.model.filePath(self.view.currentIndex()),
                                           QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            success = self.model.remove(self.view.currentIndex())

            if success:
                self.clearImage()

    def clearImage(self):
        """
        Clear the image label.

        """
        self.imageLabel.clear()
        self.setWindowTitle("Image Viewer:")

    def changeDir(self, dirname):
        """
        Change directory

        """
        self.view.setRootIndex(self.model.index(dirname))
        self.clearImage()

    def keyReleaseEvent(self, event):
        """
        Handle up/down key press

        """
        if event.key() == QtCore.Qt.Key_Up or event.key() == QtCore.Qt.Key_Down:
            self.model.filePath(self.view.currentIndex())
            self.showImage(self.model.filePath(self.view.currentIndex()))

################################################################################

class AboutMeDialog(QtWidgets.QMessageBox):
    """
    About me dialog.

    """
    def __init__(self, parent=None):
        super(AboutMeDialog, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        from ..visutils.version import getVersion
        import datetime
        import paramiko
        import matplotlib
        import platform
        import PySide
        import vtk
        import scipy
        version = getVersion()

        self.setWindowTitle("Atoman %s" % version)

        # message box layout (grid layout)
        l = self.layout()

        self.setText("""<p><b>Atoman</b> %s</p>
                          <p>Copyright &copy; %d Loughborough University</p>
                          <p>Written by Chris Scott</p>
                          <p>This application can be used to visualise atomistic simulations.</p>
                          <p>GUI based on <a href="http://sourceforge.net/projects/avas/">AVAS</a>
                             by Marc Robinson.</p>""" % (
                          version, datetime.date.today().year))

        packageList = QtWidgets.QListWidget()

        packageList.addItem("Python %s" % platform.python_version())
        packageList.addItem("Qt %s" % QtCore.__version__)
        packageList.addItem("PySide %s" % PySide.__version__)
        packageList.addItem("VTK %s" % vtk.vtkVersion.GetVTKVersion())
        packageList.addItem("NumPy %s" % np.__version__)
        packageList.addItem("SciPy %s" % scipy.__version__)
        packageList.addItem("Matplotlib %s" % matplotlib.__version__)
        packageList.addItem("Paramiko %s" % paramiko.__version__)


        # Hide the default button
        button = l.itemAtPosition( l.rowCount() - 1, 1 ).widget()
        l.removeWidget(button)

        # add list widget to layout
        l.addWidget(packageList, l.rowCount(), 1, 1, l.columnCount(), QtCore.Qt.AlignHCenter)

        # add widget back in
        l.addWidget(button, l.rowCount(), 1, 1, 1, QtCore.Qt.AlignRight)

        self.setStandardButtons(QtWidgets.QMessageBox.Ok)
        self.setIcon(QtWidgets.QMessageBox.Information)

################################################################################

class ConfirmCloseDialog(QtWidgets.QDialog):
    """
    Confirm close dialog.

    """
    def __init__(self, parent=None):
        super(ConfirmCloseDialog, self).__init__(parent)

        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.setModal(1)
        self.setWindowTitle("Exit application?")

        layout = QtWidgets.QVBoxLayout(self)

        # label
        label = QtWidgets.QLabel("<b>Are you sure you want to exit?</b>")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(label)
        layout.addLayout(row)

        # clear settings
        self.clearSettingsCheck = QtWidgets.QCheckBox("Clear settings")
        row = QtWidgets.QHBoxLayout()
        row.setAlignment(QtCore.Qt.AlignRight)
        row.addWidget(self.clearSettingsCheck)
        layout.addLayout(row)

        # buttons
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Yes | QtWidgets.QDialogButtonBox.No)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)

        layout.addWidget(buttonBox)

################################################################################

class RotateViewPointDialog(QtWidgets.QDialog):
    """
    Rotate view point dialog

    """
    def __init__(self, rw, parent=None):
        super(RotateViewPointDialog, self).__init__(parent)

#         self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowStaysOnTopHint)

        self.setWindowTitle("Rotate view point")
        self.setWindowIcon(QtGui.QIcon(iconPath("oxygen/transform-rotate.png")))
        self.setModal(0)

        self.rw = rw
        self.parent = parent

        # Dialog layout
        layout = QtWidgets.QGridLayout(self)


        # Rotation group
        RotGroup = QtWidgets.QGroupBox("Custom Rotation")
        RotGroup.setAlignment(QtCore.Qt.AlignHCenter)
        RotGroupLayout = QtWidgets.QGridLayout(RotGroup)

        # rotate up button
        UpButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('other/rotup.png')),"Up")
        UpButton.setStatusTip("Rotate Up by selected angle")
        UpButton.setToolTip("<p>Rotate Up by selected angle</p>")
        UpButton.clicked.connect(self.RotateUp)
        RotGroupLayout.addWidget(UpButton, 0, 1 )

        # angle selection
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)

        label = QtWidgets.QLabel("Angle:")

        self.angleSpin = QtWidgets.QDoubleSpinBox()
        self.angleSpin.setSingleStep(0.1)
        self.angleSpin.setMinimum(0.0)
        self.angleSpin.setMaximum(180.0)
        self.angleSpin.setValue(90)
        self.angleSpin.setToolTip("Rotation angle")

        rowLayout.addWidget(label)
        rowLayout.addWidget(self.angleSpin)
        RotGroupLayout.addWidget(row, 1, 1)

        # rotate left button
        LeftButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('other/rotleft.png')), "Left")
        LeftButton.setStatusTip("Rotate Left by selected angle")
        LeftButton.setToolTip("Rotate Left by selected angle")
        LeftButton.clicked.connect(self.RotateLeft)
        RotGroupLayout.addWidget(LeftButton, 1, 0)

        # rotate right button
        RightButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('other/rotright.png')), 'Right')
        RightButton.setStatusTip("Rotate right by selected angle")
        RightButton.setToolTip("Rotate right by selected angle")
        RightButton.clicked.connect(self.RotateRight)
        RotGroupLayout.addWidget(RightButton, 1, 2)

        # rotate down button
        DownButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('other/rotdown.png')),"Down")
        DownButton.setStatusTip("Rotate Down by selected angle")
        DownButton.setToolTip("Rotate Down by selected angle")
        DownButton.clicked.connect(self.RotateDown)
        RotGroupLayout.addWidget(DownButton, 2, 1)

        # Reset button
        ResetButton = QtWidgets.QPushButton("Reset View")
        ResetButton.setStatusTip("Reset view to the visualiser default")
        ResetButton.setToolTip("Reset view to the visualiser default")
        ResetButton.clicked.connect(self.setCameraToCell)
        RotGroupLayout.addWidget(ResetButton, 4, 1)

        # Clockwise Rotation
        CWButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('oxygen/view-refresh.png')),"ClockWise")
        CWButton.setStatusTip("Rotate lattice clockwise by selected angle")
        CWButton.setToolTip("Rotate lattice clockwise by selected angle")
        CWButton.clicked.connect(self.RotateClockWise)
        RotGroupLayout.addWidget(CWButton, 3, 0)

        # AntiClockwise Rotation
        ACWButton = QtWidgets.QPushButton(QtGui.QIcon(iconPath('other/anticlockwise.png')),"Anti-ClockWise")
        ACWButton.setStatusTip("Rotate lattice anti-clockwise by selected angle")
        ACWButton.setToolTip("Rotate lattice anti-clockwise by selected angle")
        ACWButton.clicked.connect(self.RotateAntiClockWise)
        RotGroupLayout.addWidget(ACWButton, 3, 2)


        # Add RotGroup to window
        layout.addWidget(RotGroup,1,0)


        # Custom rotate to crystal plane
        CrystalGroup = QtWidgets.QGroupBox("Rotate to top-down view of given crystal plane (EXPERIMENTAL)")
        CrystalGroup.setAlignment(QtCore.Qt.AlignHCenter)
        CrystalGroupLayout = QtWidgets.QGridLayout(CrystalGroup)

        label2 = QtWidgets.QLabel("Select crystal plane")
        CrystalGroupLayout.addWidget(label2, 0, 1)

        # inputs for crystal plane Miller index
        self.CrystalXSpin = QtWidgets.QSpinBox()
        self.CrystalXSpin.setSingleStep(1)
        self.CrystalXSpin.setMinimum(-10)
        self.CrystalXSpin.setMaximum(10)
        self.CrystalXSpin.setValue(1)
        self.CrystalXSpin.setToolTip("X param of crystal plane Miller Index")
        CrystalGroupLayout.addWidget(self.CrystalXSpin, 1, 0)

        self.CrystalYSpin = QtWidgets.QSpinBox()
        self.CrystalYSpin.setSingleStep(1)
        self.CrystalYSpin.setMinimum(-10)
        self.CrystalYSpin.setMaximum(10)
        self.CrystalYSpin.setValue(1)
        self.CrystalYSpin.setToolTip("Y param of crystal plane Miller Index")
        CrystalGroupLayout.addWidget(self.CrystalYSpin, 1, 1)

        self.CrystalZSpin = QtWidgets.QSpinBox()
        self.CrystalZSpin.setSingleStep(1)
        self.CrystalZSpin.setMinimum(-10)
        self.CrystalZSpin.setMaximum(10)
        self.CrystalZSpin.setValue(1)
        self.CrystalZSpin.setToolTip("Z param of crystal plane Miller Index")
        CrystalGroupLayout.addWidget(self.CrystalZSpin, 1, 2)

        # Button to do the rotation
        CrystalRotButton = QtWidgets.QPushButton("Rotate")
        CrystalRotButton.setStatusTip("Rotate to a top-down view of the given plane above.")
        CrystalRotButton.setToolTip("Rotate to a top-down view of the given plane above.")
        CrystalRotButton.clicked.connect(self.RotateViewGeneral)
        CrystalGroupLayout.addWidget(CrystalRotButton, 2, 1)

        # Add CrystalGroup to window
        layout.addWidget(CrystalGroup,2,0)


        # Close Button
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        rowLayout.setAlignment(QtCore.Qt.AlignHCenter)

        CloseButton = QtWidgets.QPushButton("Close")
        CloseButton.clicked.connect(self.reject)
        CloseButton.setDefault(True)
        rowLayout.addWidget(CloseButton)
        layout.addWidget(row,3,0)


    def RotateClockWise(self):
        """
        Rotate viewpoint clockwise about the vector coming out of the center of the screen.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = -self.angleSpin.value()

        # apply rotation
        logger.debug("Appling clockwise rotation by %f degrees", angle)
        logger.debug("Calling: Roll %f", angle)
        renderer.camera.Roll(float(angle))
        renderer.reinit()

    def RotateAntiClockWise(self):
        """
        Rotate viewpoint anticlockwise about the vector coming out of the center of the screen.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = self.angleSpin.value()

        # apply rotation
        logger.debug("Appling anticlockwise rotation by %f degrees", angle)
        logger.debug("Calling: Roll %f", angle)
        renderer.camera.Roll(float(angle))
        renderer.reinit()

    def RotateRight(self):
        """
        Rotate viewpoint Right by the given angle.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = -self.angleSpin.value()

        # apply rotation
        logger.debug("Appling right rotation by %f degrees", angle)
        logger.debug("Calling: azimuth %f", angle)
        renderer.camera.Azimuth(float(angle))
        renderer.reinit()

    def RotateLeft(self):
        """
        Rotate viewpoint Left by the given angle.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = self.angleSpin.value()

        # apply rotation
        logger.debug("Appling right rotation by %f degrees", angle)
        logger.debug("Calling: azimuth %f", angle)
        renderer.camera.Azimuth(float(angle))
        renderer.reinit()

    def RotateUp(self):
        """
        Rotate viewpoint Up by the given angle.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = -self.angleSpin.value()

        # apply rotation
        logger.debug("Appling right rotation by %f degrees", angle)
        logger.debug("Calling: elevation %f", angle)
        if( ((angle > 89) and (angle < 91)) or ((angle > -91) and (angle < -89))  ):
            # This is done in two steps so new viewup can be calculated correctly
            # otherwise ViewUp and DirectionOfProjection vectors become parallel
            renderer.camera.Elevation(float(angle/2.0))
            renderer.camera.OrthogonalizeViewUp()
            renderer.camera.Elevation(float(angle/2.0))
            renderer.camera.OrthogonalizeViewUp()
        else:
            renderer.camera.Elevation(float(angle))
            renderer.camera.OrthogonalizeViewUp()

        renderer.reinit()

    def RotateDown(self):
        """
        Rotate viewpoint Down by the given angle.

        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer

        angle = self.angleSpin.value()

        # apply rotation
        logger.debug("Appling right rotation by %f degrees", angle)
        logger.debug("Calling: elevation %f", angle)
        if( ((angle > 89) and (angle < 91)) or ((angle > -91) and (angle < -89))  ):
            # This is done in two steps so new viewup can be calculated correctly
            # otherwise ViewUp and DirectionOfProjection vectors become parallel
            renderer.camera.Elevation(float(angle/2.0))
            renderer.camera.OrthogonalizeViewUp()
            renderer.camera.Elevation(float(angle/2.0))
            renderer.camera.OrthogonalizeViewUp()
        else:
            renderer.camera.Elevation(float(angle))
            renderer.camera.OrthogonalizeViewUp()

        renderer.reinit()

    def setCameraToCell(self):
        """
        Reset the camera to point at the cell

        """
        renderer = self.rw.renderer
        renderer.setCameraToCell()

    def RotateViewGeneral(self):
        # crystal plane (also normal vector)
        x = self.CrystalXSpin.value()
        y = self.CrystalYSpin.value()
        z = self.CrystalZSpin.value()
        # return without doing anything if zero plane is given
        if( (x == 0) and (y == 0) and (z == 0) ):
            return

        mag_norm = math.sqrt(x*x + y*y +z*z)

        mag_proj = math.sqrt(x*x + y*y)

        # Angle between normal and projection to xy plane
        if(z == 0):
            ang_norm_xy = 0
        elif( (x == 0) and (y == 0) ):
            ang_norm_xy = 90.0
        elif( (z < 0) ):
            ang_dp = x*x + y*y
            ang_norm_xy = 360 - math.degrees( math.acos(ang_dp/(mag_proj*mag_norm)) )
        else:
            ang_dp = x*x + y*y
            ang_norm_xy = math.degrees( math.acos(ang_dp/(mag_proj*mag_norm)) )

        # Angle between xy projection and x axis
        if( (x == 0) and (y == 0) ):
            ang_norm_xz = 0.0
        elif( (y < 0) ):
            mag_proj = math.sqrt(x*x + y*y)
            ang_norm_xz = 360 - math.degrees( math.acos(x/mag_proj) )
        else:
            mag_proj = math.sqrt(x*x + y*y)
            ang_norm_xz = math.degrees( math.acos(x/mag_proj) )


        # reset view
        self.setCameraToCell()
        # get renderer
        renderer = self.rw.renderer
        # Rotate to view yz plane down the x axis
        renderer.camera.Azimuth(float(90.0))
        renderer.camera.Elevation(float(-45.0))
        renderer.camera.OrthogonalizeViewUp()
        renderer.camera.Elevation(float(-45.0))
        renderer.camera.OrthogonalizeViewUp()
        renderer.camera.Azimuth(float(90.0))

        # rotate to the crystal plane
        renderer.camera.Azimuth(float(ang_norm_xz))

        if( ((ang_norm_xy > 89) and (ang_norm_xy < 91)) or ((ang_norm_xy > -91) and (ang_norm_xy < -89))  ):
            # This is done in two steps so new viewup can be calculated correctly
            # otherwise ViewUp and DirectionOfProjection vectors become parallel
            renderer.camera.Elevation(float(ang_norm_xy/2.0))
            renderer.camera.OrthogonalizeViewUp()
            renderer.camera.Elevation(float(ang_norm_xy/2.0))
            renderer.camera.OrthogonalizeViewUp()
        else:
            renderer.camera.Elevation(float(ang_norm_xy))
            renderer.camera.OrthogonalizeViewUp()

        renderer.reinit()


################################################################################

class ReplicateCellDialog(QtWidgets.QDialog):
    """
    Ask user which directions they want to replicate the cell in

    """
    def __init__(self, pbc, parent=None):
        super(ReplicateCellDialog, self).__init__(parent)

        self.setWindowTitle("Replicate cell options")

        # layout
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)

        self.setMinimumWidth(230)
        #self.setMinimumHeight(200)

        # x
        self.replicateInXSpin = QtWidgets.QSpinBox()
        self.replicateInXSpin.setMinimum(0)
        self.replicateInXSpin.setMaximum(10)
        self.replicateInXSpin.setValue(0)
        self.replicateInXSpin.setToolTip("Number of times to replicate the cell in the x direction")
        if not pbc[0]:
            self.replicateInXSpin.setEnabled(False)
        layout.addRow("Replicate in x", self.replicateInXSpin)

        # y
        self.replicateInYSpin = QtWidgets.QSpinBox()
        self.replicateInYSpin.setMinimum(0)
        self.replicateInYSpin.setMaximum(10)
        self.replicateInYSpin.setValue(0)
        self.replicateInYSpin.setToolTip("Number of times to replicate the cell in the y direction")
        if not pbc[1]:
            self.replicateInYSpin.setEnabled(False)
        layout.addRow("Replicate in y", self.replicateInYSpin)

        # z
        self.replicateInZSpin = QtWidgets.QSpinBox()
        self.replicateInZSpin.setMinimum(0)
        self.replicateInZSpin.setMaximum(10)
        self.replicateInZSpin.setValue(0)
        self.replicateInZSpin.setToolTip("Number of times to replicate the cell in the z direction")
        if not pbc[2]:
            self.replicateInYSpin.setEnabled(False)
        layout.addRow("Replicate in z", self.replicateInZSpin)

        # button box
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)

################################################################################

class ShiftCellDialog(QtWidgets.QDialog):
    """
    Ask user which directions they want to replicate the cell in

    """
    def __init__(self, pbc, cellDims, parent=None):
        super(ShiftCellDialog, self).__init__(parent)

        self.setWindowTitle("Shift cell options")

        # layout
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)

        # x
        self.shiftXSpin = QtWidgets.QDoubleSpinBox()
        self.shiftXSpin.setMinimum(-cellDims[0])
        self.shiftXSpin.setMaximum(cellDims[0])
        self.shiftXSpin.setSingleStep(1)
        self.shiftXSpin.setValue(0)
        self.shiftXSpin.setToolTip("Distance to shift the cell in the x direction")
        if not pbc[0]:
            self.shiftXSpin.setEnabled(False)
        layout.addRow("Shift in x", self.shiftXSpin)

        # y
        self.shiftYSpin = QtWidgets.QDoubleSpinBox()
        self.shiftYSpin.setMinimum(-cellDims[1])
        self.shiftYSpin.setMaximum(cellDims[1])
        self.shiftYSpin.setSingleStep(1)
        self.shiftYSpin.setValue(0)
        self.shiftYSpin.setToolTip("Distance to shift the cell in the y direction")
        if not pbc[1]:
            self.shiftYSpin.setEnabled(False)
        layout.addRow("Shift in y", self.shiftYSpin)

        # z
        self.shiftZSpin = QtWidgets.QDoubleSpinBox()
        self.shiftZSpin.setMinimum(-cellDims[2])
        self.shiftZSpin.setMaximum(cellDims[2])
        self.shiftZSpin.setSingleStep(1)
        self.shiftZSpin.setValue(0)
        self.shiftZSpin.setToolTip("Distance to shift the cell in the z direction")
        if not pbc[2]:
            self.shiftZSpin.setEnabled(False)
        layout.addRow("Shift in z", self.shiftZSpin)

        # button box
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)


################################################################################

class ShiftAtomDialog(QtWidgets.QDialog):
    """
    Ask user which atom should be shifted and the distance of the shift in each direction.

    """
    def __init__(self, inputID, pbc, cellDims, NumAtoms, parent=None):
        super(ShiftAtomDialog, self).__init__(parent)

        self.setWindowTitle("Shift atom options")

        # layout
        layout = QtWidgets.QFormLayout()
        self.setLayout(layout)

        self.setMinimumWidth(220)
        self.setMinimumHeight(200)

        # Atom ID
        # only allow numbers, commas and hyphens
        rx = QtCore.QRegExp("[0-9]+(?:[-,]?[0-9]+)*")
        validator = QtGui.QRegExpValidator(rx, self)

        self.lineEdit = QtWidgets.QLineEdit()
        self.lineEdit.setValidator(validator)
        if (inputID>=0):
            self.lineEdit.setText(str(inputID))
        self.lineEdit.setToolTip("Comma separated list of atom IDs or ranges of atom IDs (hyphenated) that are visible (eg. '22,30-33' will show atom IDs 22, 30, 31, 32 and 33)")
        #self.lineEdit.editingFinished.connect(self._settings.updateSetting("filterString", str(self.lineEdit.text())))
        layout.addRow("Atom IDs", self.lineEdit)




        # x
        self.shiftXSpin = QtWidgets.QDoubleSpinBox()
        self.shiftXSpin.setMinimum(-cellDims[0])
        self.shiftXSpin.setMaximum(cellDims[0])
        self.shiftXSpin.setSingleStep(1)
        self.shiftXSpin.setValue(0)
        self.shiftXSpin.setToolTip("Distance to shift the atom in the x direction")
        #if not pbc[0]:
        #    self.shiftXSpin.setEnabled(False)
        layout.addRow("Shift in x", self.shiftXSpin)

        # y
        self.shiftYSpin = QtWidgets.QDoubleSpinBox()
        self.shiftYSpin.setMinimum(-cellDims[1])
        self.shiftYSpin.setMaximum(cellDims[1])
        self.shiftYSpin.setSingleStep(1)
        self.shiftYSpin.setValue(0)
        self.shiftYSpin.setToolTip("Distance to shift the atom in the y direction")
        #if not pbc[1]:
        #    self.shiftYSpin.setEnabled(False)
        layout.addRow("Shift in y", self.shiftYSpin)

        # z
        self.shiftZSpin = QtWidgets.QDoubleSpinBox()
        self.shiftZSpin.setMinimum(-cellDims[2])
        self.shiftZSpin.setMaximum(cellDims[2])
        self.shiftZSpin.setSingleStep(1)
        self.shiftZSpin.setValue(0)
        self.shiftZSpin.setToolTip("Distance to shift the atom in the z direction")
        #if not pbc[2]:
        #    self.shiftZSpin.setEnabled(False)
        layout.addRow("Shift in z", self.shiftZSpin)

        # button box
        buttonBox = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttonBox.accepted.connect(self.accept)
        buttonBox.rejected.connect(self.reject)
        layout.addRow(buttonBox)
