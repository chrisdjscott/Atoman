from __future__ import print_function


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    config = Configuration("atoman", parent_package, top_path)
    config.add_subpackage("algebra")
    config.add_subpackage("filtering")
    config.add_subpackage("gui")
    config.add_subpackage("lattice_gen")
    config.add_subpackage("plotting")
    config.add_subpackage("rendering")
    config.add_subpackage("system")
    config.add_subpackage("visutils")
    config.add_subpackage("visclibs")
    config.add_data_dir("icons")
    
    return config

if __name__ == "__main__":
    print("This is the wrong setup.py to run")
