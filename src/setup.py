
"""
Setup script for CDJSVis.

Currently compiles resource file.

Eventually should compile C libraries too.

@author: Chris Scott

"""
import os
import sys

import utilities


################################################################################
def main():
    
    pyrcc4 = utilities.checkForExe("pyrcc4")
    
    # on mac it is appended with python version
    if not pyrcc4:
        pyrcc4 = utilities.checkForExe("pyrcc4-%d.%d" % (sys.version_info[0], sys.version_info[1]))
    
    if not pyrcc4:
        sys.exit("ERROR: COULD NOT LOCATE PYRCC4")
    
    command = "%s resources.qrc > resources.py" % (pyrcc4,)
    print command
    os.system(command)


################################################################################
if __name__ == "__main__":
    main()
