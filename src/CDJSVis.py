#!/usr/bin/env python

"""
Initialise application

author: Chris Scott
last edited: February 2012
"""

import sys

from PyQt4 import QtGui

import gui
from utilities import iconPath


################################################################################
def main():
    app = QtGui.QApplication(sys.argv)
    
    mw = gui.MainWindow()
    mw.setWindowIcon(QtGui.QIcon(iconPath("applications.ico")))
    
    sys.exit(app.exec_())


################################################################################
if __name__ == '__main__':
    main()
