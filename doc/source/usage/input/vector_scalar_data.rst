
Vector/scalar data
==================

Vector and/or scalar data can be loaded onto a system by right clicking the 
system in the list and selecting the appropriate option from the context menu.

File format
-----------

* Scalar data files should have one line for each atom in the system, with one
  value per line (that atoms scalar).
* Vector data files should have one line for each atom in the system, with
  three values (that atoms vector) per line.

Use with sequencer
------------------

If the scalar/vector data files are named appropriately the sequencer will
automatically detect and read them in as it works through the inputs. 

For example, you read a lattice called "simu3.dat" and load a vector file
called "simu_vects3.dat". Then you a sequencer from 0 to 9 on this system.
It will read the files:

1. "simu0.dat" and "simu_vects0.dat"
2. "simu1.dat" and "simu_vects1.dat"

\...

3. "simu9.dat" and "simu_vects9.dat"

If the vector data files do not exist the vector options are ignored for that
system. The number format for the data files must be the same as for the
lattice files.
