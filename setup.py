
"""
Setup script for CDJSVis.

Currently compiles resource file.

Eventually should compile C libraries too.

@author: Chris Scott

"""
import os
import sys

from CDJSVis.visutils import utilities


################################################################################
def main():
    os.chdir("CDJSVis")
    
    if len(sys.argv) == 2 and sys.argv[1] == "clean":
        # walk
        for dirpath, dirnames, filenames in os.walk(os.getcwd()):
            os.chdir(dirpath)
            if "Makefile" in filenames:
                command = "make clean"
                print command
                os.system(command)
            
            else:
                command = "rm -f *.pyc *.pyo"
                print command
                os.system(command)
                
                if "resources.py" in filenames:
                    command = "rm -f resources.py"
                    print command
                    os.system(command)
                
                if "LBOMDInterface.so" in filenames:
                    os.unlink("LBOMDInterface.so")
    
    else:
        pyrcc4 = utilities.checkForExe("pyside-rcc")
        
        # on mac it is appended with python version
        if not pyrcc4:
            pyrcc4 = utilities.checkForExe("pyside-rcc-%d.%d" % (sys.version_info[0], sys.version_info[1]))
        
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
        os.chdir("..")
        
        os.chdir("md")
        
        libName = "LBOMDInterface.so"
        if os.path.islink(libName):
            head, currentLib = os.path.split(os.readlink(libName))
        else:
            currentLib = None
        
        # if we are on a mac
        lib = None
        if os.uname()[0] == "Darwin":
            lib = "LBOMDInterface.maci7.so"
        
        # if we are on a linux box
        elif os.uname()[0] == "Linux":
            
            # if on hydra
            if os.uname()[1][:5] == "hydra":
                lib = "LBOMDInterface.hydra.so"
            
            # hera
            elif os.uname()[1][:4] == "hera":
                lib = "LBOMDInterface.hera.so"
            
            # otherwise use my linux (Ubuntu) compiled library
            else:
                lib = "LBOMDInterface.linux.so"
        
        # link to lib
        if lib is None or lib == currentLib:
            pass
        else:
            if currentLib is not None:
                os.unlink(libName)
            os.symlink(os.path.join("lib", lib), libName)
        
        os.chdir("../..")
        
        if len(sys.argv) == 2 and sys.argv[1] == "test":
            print ""
            print "="*80
            print "RUNNING ALL TESTS"
            print "="*80
            print ""
            
            os.system("nosetests -v")
        
        elif len(sys.argv) == 2 and sys.argv[1] == "unittest":
            print ""
            print "="*80
            print "RUNNING UNIT TESTS ONLY"
            print "="*80
            print ""
            
            os.chdir("CDJSVis")
            os.system("nosetests -v")
            os.chdir("..")
        
        elif len(sys.argv) == 2 and sys.argv[1] == "slowtest":
            print ""
            print "="*80
            print "RUNNING SLOW TESTS ONLY"
            print "="*80
            print ""
            
            os.chdir("slow_tests")
            os.system("nosetests -v")
            os.chdir("..")

        
    
    
################################################################################
if __name__ == "__main__":
    main()
