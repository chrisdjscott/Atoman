=======
Cluster
=======

The cluster filter finds clusters within the visible atoms and allows you to view clusters of 
certain sizes. This filter is most useful when combined with other filter(s).

* *Neighbour radius* specifies the radius to use when recursively building clusters of atoms; 
  i.e. if two atoms are within this distance of each other they will be in the same cluster

* Only display atoms that belong to clusters containing at least *Minimum cluster size* atoms

* Only display atoms that belong to clusters containing less than *Maximum cluster size* 
  atoms.  Set this to be "-1" if you don't want an upper limit on cluster size

There are some additional options that are available to the cluster filter:

* *Draw convex hulls* will compute and render a convex hull around the set of points in the 
  cluster.  There are also options to specify the *Hull colour* and *Hull opacity* and you
  can also "Hide atoms" from the rendered image
  
* *Calculate volumes* allows you to calculate the volumes of the clusters using one of two 
  methods:
    
  * *Use volume of convex hull* computes the volume of the convex hull of the set of points
  * *Sum Voronoi volumes* sums the Voronoi volumes of the atoms in the cluster.  The 
    Voronoi volumes are computed using the *Voronoi settings* on the 
    :ref:`voronoi_options_label` page.
   
  I would strongly advise that you use *Sum Voronoi volumes* as this is likely to give a more
  accurate result.
