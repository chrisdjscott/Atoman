
"""
The output tab for the main toolbar

@author: Chris Scott

"""

import os
import sys

from PyQt4 import QtGui, QtCore

from utilities import iconPath
from genericForm import GenericForm
import resources




################################################################################
class OutputTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(OutputTab, self).__init__(parent)
        
        self.mainToolbar = parent
        self.mainWindow = mainWindow
        self.width = width
        
        # layout
        outputTabLayout = QtGui.QVBoxLayout(self)
        outputTabLayout.setContentsMargins(0, 0, 0, 0)
        outputTabLayout.setSpacing(0)
        outputTabLayout.setAlignment(QtCore.Qt.AlignTop)
        
        # add tab bar
        self.outputTypeTabBar = QtGui.QTabWidget(self)
        self.outputTypeTabBar.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        self.connect(self.outputTypeTabBar, QtCore.SIGNAL('currentChanged(int)'), self.outputTypeTabBarChanged)
        
#        outputTypeLayout = QtGui.QHBoxLayout(self.outputTypeTabBar)
#        outputTypeLayout.setContentsMargins(0, 0, 0, 0)
#        outputTypeLayout.setSpacing(0)
#        outputTypeLayout.setAlignment(QtCore.Qt.AlignCenter)
        
        # add tabs to tab bar
        singleImageWidget = QtGui.QWidget()
        singleImageLayout = QtGui.QVBoxLayout(singleImageWidget)
        singleImageLayout.setContentsMargins(0, 0, 0, 0)
        
        self.singleImageTab = SingleImageTab(self, self.mainWindow, self.width)
        singleImageLayout.addWidget(self.singleImageTab)
        
        self.outputTypeTabBar.addTab(singleImageWidget, "Image")
        
        # add tab bar to layout
        outputTabLayout.addWidget(self.singleImageTab)
        
        
    def outputTypeTabBarChanged(self):
        pass


################################################################################
class SingleImageTab(QtGui.QWidget):
    def __init__(self, parent, mainWindow, width):
        super(SingleImageTab, self).__init__(parent)
        
        self.parent = parent
        self.mainWindow = mainWindow
        self.width = width
        
#        self.setFixedWidth(self.width)
        
        
        
        

