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
        
        layout = QtWidgets.QVBoxLayout(self)
        
        # direction
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        
        label = QtWidgets.QLabel("Direction:")
        
        self.directionCombo = QtWidgets.QComboBox()
        self.directionCombo.addItems(["Right", "Left", "Up", "Down"])
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.directionCombo)
        
        layout.addWidget(row)
        
        # angle
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        
        label = QtWidgets.QLabel("Angle:")
        
        self.angleSpin = QtWidgets.QDoubleSpinBox()
        self.angleSpin.setSingleStep(0.1)
        self.angleSpin.setMinimum(0.0)
        self.angleSpin.setMaximum(360.0)
        self.angleSpin.setValue(90)
        
        rowLayout.addWidget(label)
        rowLayout.addWidget(self.angleSpin)
        
        layout.addWidget(row)
        
        # apply button
        row = QtWidgets.QWidget(self)
        rowLayout = QtWidgets.QHBoxLayout(row)
        
        applyButton = QtWidgets.QPushButton("Apply")
        applyButton.setStatusTip("Apply rotation")
        applyButton.setToolTip("Apply rotation")
        applyButton.clicked.connect(self.applyRotation)
        
        rowLayout.addWidget(applyButton)
        
        layout.addWidget(row)
    
    def applyRotation(self):
        """
        Apply the rotation
        
        """
        logger = logging.getLogger(__name__+".RotateViewPoint")
        renderer = self.rw.renderer
        
        angle = self.angleSpin.value()
        direction = str(self.directionCombo.currentText())
        logger.debug("Appling rotation: %s by %f degrees", direction, angle)
        
        if direction == "Right" or direction == "Left":
            if direction == "Right":
                angle = - angle
            
            # apply rotation
            renderer.camera.Azimuth(angle)
            renderer.camera.OrthogonalizeViewUp()
            logger.debug("Calling: azimuth %f", angle)
        
        else:
            if direction == "Up":
                angle = - angle
            
            # apply rotation
            renderer.camera.Elevation(angle)
            renderer.camera.OrthogonalizeViewUp()
            logger.debug("Calling: elevation %f", angle)
        
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
