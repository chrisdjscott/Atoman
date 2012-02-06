
"""
Utility methods

author: Chris Scott
last edited: February 2012
"""

import os
import sys


################################################################################
def iconPath(icon):
    """
    Return full path to given icon.
    
    """
    return os.path.join(sys.path[0], "icons", icon)


