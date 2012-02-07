#!/usr/bin/env python

"""
Initialise application

author: Chris Scott
last edited: February 2012
"""

import sys

try:
    from PyQt4 import QtGui
except:
    print __name__+ ": ERROR: could not import PyQt4"

try:
    import gui
except:
    print __name__+ ": ERROR: could not import gui"



################################################################################
def main():
    app = QtGui.QApplication(sys.argv)
    
    mw = gui.MainWindow()
    
    sys.exit(app.exec_())


################################################################################
if __name__ == '__main__':
    main()
