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
    sys.exit(__name__, "ERROR: PyQt4 not found")

try:
    import gui
except:
    sys.exit(__name__, "ERROR: gui not found")



################################################################################
def main():
    app = QtGui.QApplication(sys.argv)
    
    mw = gui.MainWindow()
    
    sys.exit(app.exec_())


################################################################################
if __name__ == '__main__':
    main()
