# -*- coding: utf-8 -*-



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

def setup_package():
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
