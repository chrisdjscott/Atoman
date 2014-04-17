
"""
Setup script for CDJSVis.

Currently compiles resource file.

Eventually should compile C libraries too.

@author: Chris Scott

"""
import os
import sys
import shutil

from CDJSVis.visutils import utilities

################################################################################

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "clean":
        os.chdir("CDJSVis")
        
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
        # compile resource file
        pyrcc4 = utilities.checkForExe("pyside-rcc")
        
        # on mac it is appended with python version
        if not pyrcc4:
            pyrcc4 = utilities.checkForExe("pyside-rcc-%d.%d" % (sys.version_info[0], sys.version_info[1]))
        
        if not pyrcc4:
            sys.exit("ERROR: COULD NOT LOCATE PYRCC4")
        
        # have to compile resources first so that it can be imported by sphinx...
        # ...viscous circle...
        try:
            from CDJSVis import resources
        except ImportError:
            os.chdir("CDJSVis")
        
            command = "%s resources.qrc > resources.py" % (pyrcc4,)
            print command
            os.system(command)
        
            os.chdir("..")
        
        # compile C libs
        os.chdir(os.path.join("CDJSVis", "visclibs"))
        command = "make"
        print command
        os.system(command)
        os.chdir("../..")
        
        # build sphinx doc
        os.chdir("doc")
        
#        if "doc" in sys.argv:
#            # first generate latest module info
#            os.system("./modules_gen_auto.sh")
        
        # then run make
        os.system("make html")
        
        os.chdir("..")
        
        # copy doc to CDJSVis
        if os.path.isdir(os.path.join("CDJSVis", "doc")):
            shutil.rmtree(os.path.join("CDJSVis", "doc"))
        
        shutil.copytree(os.path.join("doc", "_build", "html"), os.path.join("CDJSVis", "doc"))
        
        # edit resources file
        os.chdir("CDJSVis")
        
        fn = "resources.qrc"
        f = open(fn)
        lines = f.readlines()
        f.close()

        count = 0
        for line in lines:
            if line.startswith("</qresource>"):
                break
    
            count += 1

        lines = lines[:count]

        lines.append("\n")

        assert os.path.isdir("doc")
        assert os.path.exists(os.path.join("doc", "index.html"))

        if os.path.exists(os.path.join("doc", ".buildinfo")):
            os.unlink(os.path.join("doc", ".buildinfo"))

        count = 0
        for root, dirs, files in os.walk("doc"):
            for addfn in files:
                lines.append("    <file>%s</file>\n" % os.path.join(root, addfn))
                count += 1

        lines.append("</qresource>\n")
        lines.append("</RCC>\n")

        f = open("resources_mod.qrc", "w")
        f.write("".join(lines))
        f.close()
        
        # compile resource file
        command = "%s resources_mod.qrc > resources.py" % (pyrcc4,)
        print command
        os.system(command)
        
        # delete doc/ dir and modified qrc file, no longer required
        os.unlink("resources_mod.qrc")
        shutil.rmtree("doc")
        
        os.chdir("..")
        
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
