#!/usr/bin/env python

"""
Initialise application.

@author: Chris Scott

"""
import sys

from PyQt4 import QtGui

from CDJSVis import mainWindow
from CDJSVis.visutils.utilities import iconPath

################################################################################

def main():
    app = QtGui.QApplication(sys.argv)
    
    mw = mainWindow.MainWindow()
    mw.setWindowIcon(QtGui.QIcon(iconPath("CDJSVis.ico")))
    
    sys.exit(app.exec_())

################################################################################

if __name__ == '__main__':
    main()
