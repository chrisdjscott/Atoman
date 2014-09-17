# -*- coding: utf-8 -*-

import os
import glob
import sys


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from CDJSVis.visutils import version
    
    config = Configuration(None, parent_package, top_path, version=version.getVersion())
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
        if "Makefile" in files:
            cmd = "make clean"
            os.chdir(root)
            print "(cd %s; %s)" % (root, cmd)
            print cmd
            os.system(cmd)
        
        so_files = glob.glob(os.path.join(root, "*.so"))
        for so_file in so_files:
            print "rm %s" % os.path.join(root, so_file)
            os.unlink(so_file)
        
        if "resources.py" in files:
            os.unlink(os.path.join(root, "resources.py"))
    
    os.chdir(cwd)

def setup_package():
    # clean?
    if "clean" in sys.argv:
        do_clean()
    
    # srcpath stuff?
    
    
    # always recompile resources file
    
    
    # redo version file?
    
    
    # documentation (see scipy...)
    
    
    # metadata
    metadata = dict(
        name = "CDJSVis",
        maintainer = "Chris Scott",
        maintainer_email = "chris@chrisdjscott.co.uk",
        description = "CDJSVis Atomistic Visualisation and Analysis Library",
#         long_description = "",
        url = "http://chrisdjscott.com",
        author = "Chris Scott",
        author_email = "chris@chrisdjscott.co.uk",
#         download_url = "",
#         license = "BSD",
#         classifiers = "",
        platforms = ["Linux", "Mac OS-X"],
#         test_suite = "",
#         cmdclass = "",
    )
    
    from numpy.distutils.core import setup
    
    metadata["configuration"] = configuration
    
    setup(**metadata)

if __name__ == "__main__":
    setup_package()
