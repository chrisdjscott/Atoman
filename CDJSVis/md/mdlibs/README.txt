So you should place interface files (.so files containing python modules)
in the lib directory with a unique (sensible) name, eg. LBOMDInterface.hydra.so 
or LBOMD.tomasmac.so.

Then you must add a rule to the __init__.py file to select your module 
depending on uname etc. See how I have done it.
