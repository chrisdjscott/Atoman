
"""
Setup script for CDJSVis.

Currently compiles resource file.

Eventually should compile C libraries too.

@author: Chris Scott

"""
import os
import sys

from .visutils import utilities


################################################################################
def main():
    
    if len(sys.argv) == 2 and sys.argv[1] == "clean":
        command = "rm -f *.pyc"
        print command
        os.system(command)
        
        os.chdir("visclibs")
        command = "make clean"
        print command
        os.system(command)
        
    else:
        pyrcc4 = utilities.checkForExe("pyrcc4")
        
        # on mac it is appended with python version
        if not pyrcc4:
            pyrcc4 = utilities.checkForExe("pyrcc4-%d.%d" % (sys.version_info[0], sys.version_info[1]))
        
        if not pyrcc4:
            sys.exit("ERROR: COULD NOT LOCATE PYRCC4")
        
        command = "%s resources.qrc > resources.py" % (pyrcc4,)
        print command
        os.system(command)
        
        # run Makefile
        os.chdir("visclibs")
        command = "make"
        print command
        os.system(command)


################################################################################
if __name__ == "__main__":
    main()
