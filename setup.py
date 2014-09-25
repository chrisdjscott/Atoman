# -*- coding: utf-8 -*-

"""
Setup script for CDJSVis

@author: Chris Scott

"""
import os
import glob
import sys
import subprocess
import shutil
import platform

from CDJSVis.visutils import utilities
from CDJSVis.visutils import version

VERSION = version.getVersion()

# if on Mac we have to force gcc (for openmp...)
if platform.system() == "Darwin":
    os.environ["CC"] = "gcc"

try:
    from sphinx.setup_command import BuildDoc
    HAVE_SPHINX = True
except:
    HAVE_SPHINX = False

if HAVE_SPHINX:
    class CDJSVisBuildDoc(BuildDoc):
        """Compile resources and run in-place build before Sphinx doc-build"""
        def run(self):
            # in place build
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building CDJSVis failed")
            
            # build doc
            BuildDoc.run(self)
            
            # copy doc to CDJSVis
            if os.path.exists(os.path.join("doc", "_build", "html", "index.html")):
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
                command = "%s resources_mod.qrc > resources.py" % PYRCC
                print command
                os.system(command)
                
                # delete doc/ dir and modified qrc file, no longer required
                os.unlink("resources_mod.qrc")
                shutil.rmtree("doc")
                
                os.chdir("..")

# look for rcc exe
PYRCC = utilities.checkForExe("pyside-rcc")
# on mac it is appended with python version
if not PYRCC:
    PYRCC = utilities.checkForExe("pyside-rcc-%d.%d" % (sys.version_info[0], sys.version_info[1]))
# fail
if not PYRCC:
    raise RuntimeError("Cannot locate pyside-rcc executable")

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration(None, parent_package, top_path, version=VERSION)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.add_subpackage("CDJSVis")
    config.add_scripts(["CDJSVis.py"])
    
    return config

def do_clean():
    cwd = os.getcwd()
    os.chdir("CDJSVis")
    for root, dirs, files in os.walk(os.getcwd()):
        so_files = glob.glob(os.path.join(root, "*.so"))
        for so_file in so_files:
            print "rm CDJSVis/%s" % os.path.relpath(so_file)
            os.unlink(so_file)
        
        if "resources.py" in files:
            os.unlink(os.path.join(root, "resources.py"))
        
#         cmd = "rm -f %s %s" % (os.path.join(root, "*.pyc"), os.path.join(root, "*.pyo"))
#         print cmd
    
    os.chdir(cwd)
    
    if os.path.isdir("doc/_build"):
        print "rm -rf doc/_build"
        shutil.rmtree(os.path.join("doc", "_build"))

def setup_package():
    # clean?
    if "clean" in sys.argv:
        do_clean()
     
    # documentation (see scipy...)
    if HAVE_SPHINX:
        cmdclass = {'build_sphinx': CDJSVisBuildDoc}
    else:
        cmdclass = {}
    
    # recompile resources file if cannot import it
    try:
        from CDJSVis import resources
    except ImportError:
        os.chdir("CDJSVis")
        command = "%s resources.qrc > resources.py" % PYRCC
        print command
        os.system(command)
        os.chdir("..")
    
    # metadata
    metadata = dict(
        name = "CDJSVis",
        maintainer = "Chris Scott",
        maintainer_email = "chris@chrisdjscott.co.uk",
        description = "CDJSVis Atomistic Visualisation and Analysis Library",
         long_description = "CDJSVis Atomistic Visualisation and Analysis Library",
        url = "http://chrisdjscott.com",
        author = "Chris Scott",
        author_email = "chris@chrisdjscott.co.uk",
#         download_url = "",
         license = "BSD",
#         classifiers = "",
        platforms = ["Linux", "Mac OS-X"],
#         test_suite = "",
        cmdclass = cmdclass,
    )
    
    if len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or sys.argv[1] in ('--help-commands', 'egg_info', 
                                                                           '--version', 'clean', 'nosetests',
                                                                           'test')):
        try:
            from setuptools import setup
        except ImportError:
            from distutils.core import setup
        
        metadata['version'] = VERSION
        metadata['test_suite'] = "nose.collector"
    
    else:
        from numpy.distutils.core import setup
    
        metadata["configuration"] = configuration
    
    # run setup
    setup(**metadata)
    
    if "clean" in sys.argv:
        if os.path.isdir("build"):
            print "rm -rf build/"
            shutil.rmtree("build")
        if os.path.isdir("CDJSVis.egg-info"):
            print "rm -rf CDJSVis.egg-info/"
            shutil.rmtree("CDJSVis.egg-info")

if __name__ == "__main__":
    setup_package()
