==================================
Adaptive common neighbour analysis
==================================

The ACNA calculator/filter performs the adaptive common neighbour analysis of Stutowski [1]_.

This classifies an atom as either:

0. Disordered
1. FCC
2. HCP
3. BCC
4. Icosahedral 

On the settings form you must set the parameter *Max bond distance* to be something sensible for your system.  
This parameter is used to spatially decompose the system in order to speed up the algorithm and should be chosen
so that the required number of neighbours (14 for BCC, 12 for the others) will be found within this distance of
a given atom. If in doubt, set it to something large (the code will just run slower).

More information to follow...

.. [1] A. Stukowski. *Modelling Simul. Mater. Sci. Eng.* **20** (2012) 045021; `doi: 10.1088/0965-0393/20/4/045021 <http://dx.doi.org/10.1088/0965-0393/20/4/045021>`_.
