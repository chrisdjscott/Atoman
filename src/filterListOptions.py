
"""
Options for filter lists.

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui, QtCore

import utilities
from utilities import iconPath
import genericForm

try:
    import resources
except ImportError:
    print "ERROR: could not import resources: ensure setup.py ran correctly"
    sys.exit(36)




################################################################################

class ColouringOptionsWindow(QtGui.QDialog):
    """
    Window for displaying colouring options for filter list
    
    """
    def __init__(self, parent=None):
        super(ColouringOptionsWindow, self).__init__(parent)
        
        self.parent = parent
        self.setModal(0)
#        self.setSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        
        self.setWindowTitle("Filter list colouring options")
        self.setWindowIcon(QtGui.QIcon(iconPath("painticon.png")))
        self.resize(500,300)
        
        # defaults
        self.colourBy = "Specie"
        
        windowLayout = QtGui.QVBoxLayout(self)
#        windowLayout.setAlignment(QtCore.Qt.AlignTop)
#        windowLayout.setContentsMargins(0, 0, 0, 0)
#        windowLayout.setSpacing(0)
        
        # combo box
        self.colouringCombo = QtGui.QComboBox()
        self.colouringCombo.addItem("Specie")
        self.colouringCombo.addItem("Height")
        self.colouringCombo.currentIndexChanged.connect(self.colourByChanged)
        
        windowLayout.addWidget(self.colouringCombo)
        
        # stacked widget
        self.stackedWidget = QtGui.QStackedWidget(self)
        
        # specie widget
        class SpecieOptions(QtGui.QWidget):
            """
            Specie options.
            
            """
            def __init__(self, parent=None):
                super(SpecieOptions, self).__init__(parent)
                
                self.parent = parent
        
        self.specieOptions = SpecieOptions(self)
        self.stackedWidget.addWidget(self.specieOptions)
        
        # height widget
        class HeightOptions(QtGui.QWidget):
            """
            Height options.
            
            """
            def __init__(self, parent=None):
                super(HeightOptions, self).__init__(parent)
                
                self.parent = parent
                
                self.axis = 1
                
                # layout
                layout = QtGui.QVBoxLayout(self)
                layout.setSpacing(0)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setAlignment(QtCore.Qt.AlignTop)
                
                # axis
                axisCombo = QtGui.QComboBox()
                axisCombo.addItem("Height in x")
                axisCombo.addItem("Height in y")
                axisCombo.addItem("Height in z")
                axisCombo.currentIndexChanged.connect(self.axisChanged)
                layout.addWidget(axisCombo)
                
                # min/max
                
            def axisChanged(self, index):
                """
                Changed axis.
                
                """
                self.axis = index
                
        self.heightOptions = HeightOptions(self)
        self.stackedWidget.addWidget(self.heightOptions)
        
        windowLayout.addWidget(self.stackedWidget)

    
    def colourByChanged(self, index):
        """
        Colour by changed.
        
        """
        self.colourBy = str(self.colouringCombo.currentText())
        
        self.stackedWidget.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """
        Close event.
        
        """
        self.parent.colouringOptionsOpen = False
        self.hide()

