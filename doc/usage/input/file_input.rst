File input
==========

File input is handled by the systems dialog. Multiple files can be loaded at the same time by shift/cmd clicking them. Once loaded they will be added to the "Loaded systems" list.  Systems can be removed from the list by selecting them (multiple selection is possible) and clicking the minus sign. Note that systems that are currently selected on an analysis pipeline, as either a ref or input, cannot be removed.

Basically you should always leave this as 'AUTO DETECT'. The available formats are listed below.

AUTO DETECT
-----------

This method should usually always be selected. It will read the first few lines of a file and decide which reader should be used (datReader, xyzReader, refReader, ...).  If this doesn't work then probably there is something wrong with the file or it is in a format not recognised yet (let me know).

LBOMD DAT
---------

Read LBOMD lattice format files.  They should be in the format that LBOMD requires for 'lattice.dat':

  #) First line should be the number of atoms
  #) Second line should be the cell dimensions (x, y, z)
  #) Then one line per atom containing (separated by whitespace): 
      
     * symbol
     * x position
     * y position
     * z position
     * charge 

.. _LBOMD_REF:

LBOMD REF
---------

Read LBOMD animation-reference format files. They should be in the following format

  #) First line should be the number of atoms
  #) Second line should be the cell dimensions (x, y, z)
  #) Then one line per atom containing (separated by whitespace): 
      
     * symbol
     * atom index (starting from 1 to NATOMS)
     * x position
     * y position
     * z position
     * kinetic energy at time 0
     * potential energy at time 0
     * x force at time 0
     * y force at time 0
     * z force at time 0
     * charge

LBOMD XYZ
---------

Read LBOMD XYZ format files.  XYZ files must be linked to an :ref:`LBOMD_REF` file (i.e. you must read one of those files first).  It does not make sense to have an XYZ file without an animation-reference file because the atom symbols are only stored in the reference. The number of atoms must be the same in the reference you are using to link with the XYZs.  When you load an :ref:`LBOMD_REF` file it will automatically be linked to any subsequently loaded XYZ files.

Different formats of XYZ files are supported (more can be added...)

#)  Positions and energies

    *   First line is number of atoms
    *   Second line is simulation time in fs
    *   Then one line per atom containing (separated by whitespace)
    
        *   atom index
        *   x position
        *   y position
        *   z position
        *   kinetic energy
        *   potential energy

#)  Positions, energies and charges

    *   First line is number of atoms
    *   Second line is simulation time in fs
    *   Then one line per atom containing (separated by whitespace)
    
        *   atom index
        *   x position
        *   y position
        *   z position
        *   kinetic energy
        *   potential energy
        *   charge
