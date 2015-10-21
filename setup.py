# -*- coding: utf-8 -*-

"""
Setup script for atoman

@author: Chris Scott

"""
import os
import glob
import sys
import subprocess
import shutil
import platform

from atoman.visutils import version

VERSION = version.getVersion()

# write version to freeze file
open(os.path.join("atoman", "visutils", "version_freeze.py"), "w").write("__version__ = '%s'\n" % VERSION)

# if on Mac we have to force gcc (for openmp...)
if platform.system() == "Darwin":
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"

try:
    from sphinx.setup_command import BuildDoc
    HAVE_SPHINX = True
except:
    HAVE_SPHINX = False

if HAVE_SPHINX:
    class AtomanBuildDoc(BuildDoc):
        """Compile resources and run in-place build before Sphinx doc-build"""
        def run(self):
            # in place build
            ret = subprocess.call([sys.executable, sys.argv[0], 'build_ext', '-i'])
            if ret != 0:
                raise RuntimeError("Building atoman failed")
            
            # build doc
            BuildDoc.run(self)
            
            # copy doc to atoman
            sphinxHtmlDir = os.path.join("build", "sphinx", "html")
            if os.path.exists(os.path.join(sphinxHtmlDir, "index.html")):
                if os.path.isdir(os.path.join("atoman", "doc")):
                    shutil.rmtree(os.path.join("atoman", "doc"))
                shutil.copytree(sphinxHtmlDir, os.path.join("atoman", "doc"))
            
            else:
                raise RuntimeError("Could not locate Sphinx documentation HTML files")

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration(None, parent_package, top_path, version=VERSION)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       quiet=True)
    
    config.add_subpackage("atoman")
    
    return config

def do_clean():
    cwd = os.getcwd()
    os.chdir("atoman")
    for root, dirs, files in os.walk(os.getcwd()):
        so_files = glob.glob(os.path.join(root, "*.so"))
        for so_file in so_files:
            print "rm atoman/%s" % os.path.relpath(so_file)
            os.unlink(so_file)
        
        if "resources.py" in files:
            os.unlink(os.path.join(root, "resources.py"))
       
        pyc_files = glob.glob(os.path.join(root, "*.pyc"))
        for pyc_file in pyc_files:
            os.unlink(pyc_file)
    
    os.chdir(cwd)
    
    if os.path.isdir("doc/_build"):
        print "rm -rf doc/_build"
        shutil.rmtree(os.path.join("doc", "_build"))
    
    if os.path.isdir("atoman/doc"):
        print "rm -rf atoman/doc"
        shutil.rmtree(os.path.join("atoman", "doc"))
    
    if os.path.isdir("dist"):
        print "rm -rf dist/"
        shutil.rmtree("dist")

def setup_package():
    # clean?
    if "clean" in sys.argv:
        do_clean()
     
    # documentation (see scipy...)
    if HAVE_SPHINX:
        cmdclass = {'build_sphinx': AtomanBuildDoc}
    else:
        cmdclass = {}
    
    # metadata
    metadata = dict(
        name = "atoman",
        maintainer = "Chris Scott",
        maintainer_email = "chris@chrisdjscott.co.uk",
        description = "Atomistic simulation analysis and visualisation library",
         long_description = "Atomistic simulation analysis and visualisation library",
        url = "http://vis.chrisdjscott.com.uk",
        author = "Chris Scott",
        author_email = "chris@chrisdjscott.co.uk",
#         download_url = "",
         license = "LGPL",
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
        if os.path.isdir("atoman.egg-info"):
            print "rm -rf atoman.egg-info/"
            shutil.rmtree("atoman.egg-info")

if __name__ == "__main__":
    setup_package()
