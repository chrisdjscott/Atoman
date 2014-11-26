
import os

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    
    # path to header files
    cwd = os.path.dirname(os.path.abspath(__file__))
    incdir = os.path.abspath(os.path.join(cwd, "..", "visclibs"))
    
    # config
    config = Configuration("lattice_gen", parent_package, top_path)
    
    config.add_extension("_lattice_gen_bcc", 
                         ["_lattice_gen_bcc.c", "../visclibs/array_utils.c"],
                         include_dirs=[incdir])
    
    config.add_extension("_lattice_gen_fcc", 
                         ["_lattice_gen_fcc.c", "../visclibs/array_utils.c"],
                         include_dirs=[incdir])
    
    return config

if __name__ == "__main__":
    print "This is the wrong setup.py to run"
