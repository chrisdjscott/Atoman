=========
Changelog
=========

**170201**

* Update dependencies in README.md
* Fix bug when writing partial lattice (species were not correct)

**161205**

* Add option to set the number of labels on the scalar bar
* Add option to set the format of the labels on the scalar bar
* Increase number of decimal places in spin boxes on colouring options form

**161119**

* Fix bug when scalar bar is shown and changing the background colour

**160704**

* Fix bug when determining if atoms belong within a convex hull
* Fix use of numpy.where
* Add overrides for all mouse events to ensure picking works properly
* Version printed properly on about screen from pyinstaller app
* Fix bug when drawBonds is selected but no bonds are checked

**160409**

* Add spaghetti rendering to point defects filter

**160314**

* Add option to clusters filter to display all atoms in/out-side the convex hulls 

**160301**

* Fix bug where checking drift compensation did nothing
* Interstitial count displayed on screen now includes split interstitials

**160227**

* Rewrite of filtering and rendering (easier to call from scripts)
* Add support for Python 3 (with VTK 7.0)
* Add more tests
* Add versioneer for reporting the version (section added to setup.cfg.example)

**151120**

* Add entry point to replace cdjsvis.py script (run as 'python -m atoman')
* Rearrange structure of subpackages
* Remove dependency on GSL
* Add lots more tests
* Set numpy deprecated api version
* Fix bug in defects filter regarding filtering by species
* Many other changes

**151002**

* Rewrite filters to be objects, with settings objects
* Rewrite filter settings forms to use new settings objects
* Add documentation and tests for many filters
* Update other parts of the documentation
* Extra options for Slip filter

**150828**

* Fix bug that could have caused a segfault in some C libraries
* Add option to generate H.264 mp4 movies (default)

**150811**

* Add basic slip filter
* Fix bug that might prevent scalar bar being displayed
* Orthogonalise view up vector after rotating lattice

**150616**

* Fix bug when loading FAILSAFE files: make sure all scalar/vector data is stored as float (for now)

**150609**

* Fix bug preventing split interstitials from being displayed

**150527**

* Fix bug preventing displaying file dialog when loading scalar/vector data.

**150420.1**

* Fix bug regarding split interstitials and filtering defects by species

**150420**

* Disable multiple reference site check in defects filter
* Set the default PBCs for newly loaded systems
* Fix bug preventing lattice attributes being displayed

**150312**

* Improve split interstitial detection to work in clusters of interstitials too
* Improve Atom ID filter to accept ranges and lists of atom IDs
* Improve ACNA refinement of point defects to work in clusters of defects too
* Colouring options are available as soon as a filter is added
* Change default oxygen radius to 0.66
* Fix bug when rendering bonds and colouring by charge

**150303**

* Handle case where file name contains colons (was wrongly detecting them as SFTP files)

**150226**

* Revert change regarding displaying antisites
* Fix bug preventing "Solid colour" from working
* "Crop box" filter works with "Point defects" filter
* "Slice" filter works with "Point defects" filter

**150223**

* Fix bug that could have prevented some bonds being displayed
* No longer require qresources file; doc and icons are added as data instead

**150211**

* Fix bug preventing Lattice being reloaded

**150206**

* Better error handling during type conversion when reading lattice files
* Add option to shift the lattice in periodic directions
* Add dialog to display information about a loaded system

**150204.1**

* Fix bug when colouring by lattice scalars

**150204**

* Better error handling from C code (raise exception instead of exiting)
* Better normalisation of RDF
* Fix bug when picking split interstitials
* Improve "Replicate cell" option
* Display Exception if running filters fails

**150126**

* Added generic file input module (much easier to add new lattice file formats)
* Add progress dialog to status bar
* Show progress when unzipping and reading files
* Show progress when copying file via SFTP browser
* Refresh plot options when running a single list
* Systems dialog moved to tab in toolbar
* Add option to change ambient, specular, specular power on actors
* Replace most icons with icons from the Oxygen icon theme

**150114**

* Try and autodetect the number format for the sequencer
* Add option to normalise a vector before rendering/scaling it
* Fix min value when specifying bin width in plot histogram
* All scalar/vector data shown on atom info windows
* Improved scalar data handling on Lattice
* Ability to read LBOMD xyz files with velocities

**v150105**

* If max index on sequencer is less than min index we auto detect the max
* SFTP browser and sequencer work with zipped files

**v141212**

* Close info windows when clearing a filter list
* Compatible with VTK 5 and 6
* Add ability to load scalar/vector data from a file to a lattice object
* Loading vector data works with the sequencer if the files are named properly
* Add option to render vector data (filter list options)
* Add fine grained actor visibility options to filer lists
* One actor created for all atoms, not per specie (faster when there are multiple species)
* Add option to replicate the cell in periodic directions
* Add SiC 4H (diamond) lattice generator
* Add axes that are always displayed and can be moved around the screen

**v141202**

* Fix bug preventing rendering defects
* Add ability to load custom scalar values from a file for a Lattice

**v141127**

* Display error message if max bonds per atom exceeded
* Use QFormLayout for many forms/dialogs
* FCC/BCC/Pu-Ga lattice generators written in C to improve performance
* Fix bug when reading animation reference format file - species could be wrong if the atoms in the file were not ordered by atom ID

**v141113**

* Histogram min/max are always a multiple of the bin width
* Do not show progress dialog when running filter lists
* Add option to disable mouse wheel events in the VTK window (on by default)

**v141031**

* Make paramiko optional
* Selected ACNA value from point defect filter was not being used

**v141021**

* Rewrite on-screen info dialog
   - Change layout
   - Add option to customise the text
* Fix bug when clicking defect (introduced in last version)
* Add ability to render two render windows side by side (same pipeline only for now)
* Can now be installed using 'python setup.py install' like any other Python package

**v141015**

* Fix bug in defect filter when using ACNA to refine results (seg fault)
* Make pyhull an optional dependency
* Only add specie options for current species (point defect/specie filters)

**v141013**

* Add option to plot distribution of cluster volumes, if they have been calculated
* Fix plotting distribution of defect clusters
* Make Voronoi volume method the default way to calculate cluster volumes
* Show mean, standard deviation and standard error on histogram plot forms

**v141010**

* Add neighbour info window - shows information (separation, etc) about an atoms neighbours
* Display number of clusters on screen (if there are any clusters)
* Add option to plot the cluster size distribution

**v141008**

* Add option to set how many OpenMP threads will be used in C extensions
* Add face area threshold option when computing Voronoi neighbours

**v141003**

* When using Voronoi volumes to compute volume of defect cluster, sum volumes of vacancies as well as interstitials/antisites
* Add option to trace the positions of atoms in a filter list
* Use list widget for bonds options (better when lots of different specie types)
* Only add bond options for species present in the current lattice

**v140929**

* Show progress dialog when applying lists and calculating RDF
* Display error if try to compute RDF out to distance greater than half the cell dimensions
* Add option to use ACNA filter within point defects filter, to refine defect detection
* Write C extension interface to Voro++ => big improvement in speed of Voronoi calculations
* Parallelised some sections of C extensions using OpenMP (bond order, RDF, ACNA)

**v140922**

* Add option to draw displacement vector showing where displaced atoms have moved from
* Use spherical harmonics function from GSL library in bond order filter
* Add adaptive common neighbour analysis
* Convert all ctypes C libraries to C extensions
* Add option to write csv file containing plot data

**v140605**

* Fix bug to do with writing POV-Ray atoms/running Ffmpeg on thread pool

**v140604**

* Fix bug when rendering bonds and colouring by scalar value

**v140603**

* Element editor lets you change any elements settings, not just elements in loaded systems
* Add option to do a rotation after sequencer has finished
* Add option to toggle between parallel/perspective projection
* Add option to write logging output to file

**v140529**

* Add dialog for editing bonds settings with ability to add new bond pairs
* Add option to attempt to eliminate flicker across PBCs when running sequencer

**v140516**

* Enable multiple item selection in SFTP browser

**v140515**

* FFmpeg runs in thread pool
* Defect clusters work with split interstitials
* Calculating volumes of defect clusters works
* Option to compute defect cluster volume using convex hull volume or sum of Voronoi volumes
* If "Voronoi volume" calculator is selected, report the sum of visible atom's volumes
* Remove some buttons from filter list (move up/down replaced by drag/drop; add/remove replaced by combo/right click)
* Add cluster info windows and highlight atoms that are in the selected cluster
* Atoms/defects that belong to a cluster have a link to the cluster info dialog on their info windows
* Add option to all info windows to change the colour of the highlighters

**v140513**

* Add atom index filter
* POV-Ray call runs in separate thread
* Add "KMC step" option to on screen info
* Add SFTP browser
   * Browse and load files on SFTP server
   * Sequencer works with files loaded via SFTP
   * Also looks for Roulette files and copies them too if available
   * Only available on "AUTO DETECT" reader
* Append timestamp to 'rotator' and 'sequencer' directories

**v140417**

* After running sequencer change back to original settings on systems dialog
* Fix bug in renderBonds/Voronoi introduced when adding multiple scalars
* Fix bug: don't try to add scalar plot option if there are no visible atoms
* On filter settings forms use group box with check button to enable filtering
* Add option to "Bond order" calculator to filter by Q4 and/or Q6

**v140415**

* Add drift compensation to "point defects" and "displacement" filters
* Fix bug in picker: ignore filter lists that are not visible

**v140407**

* Added app icon (icns file)
* Added toolbar to help page
* Replaced Q4 filter with "bond order" filter/property calculator, which calculates Q4 and Q6 parameters
* Added some documentation for the "bond order" filter

**v140401**

* Fix bug in window positioning
* Add histogram plots for atom properties (PE, KE, charge)
* Option to specify bin width instead of number of bins for histogram plots

**v140331**

* Add ability to store multiple scalar values
* Rename 'filter list' to 'property/filter list'
* Add context menu to items in property/filter lists with options to edit settings and remove from list
* Items in property/filter lists can be reordered by dragging
* Added quick add combo box to property/filter list (quicker than adding via dialog)
* When selecting atom property from colouring options scalar bar text is automatically determined
* Add option to plot histogram of scalar values (to Plot tab in output dialog)
* Fix not being able to use native file dialog

**v140328**

* Add sphere resolution settings to "Display options" on "Filter list"
* Increase default sphere resolution
* Atom indexes taken from input file instead of using internal index (i.e. now they normally start from 1)
* Better handling of small files (< 5 atoms) in Auto-Detect reader
* Change version numbering to date

**v0.12.2**

* Fix bug when generating lattice (addAtom)

**v0.12.1**

* Fix bug in element editor

**v0.12**

* Improve render atoms speed (rewrite some bits in C)
* Add title to Pipeline Form
* POV-Ray atoms file is written in separate thread (unless in Sequencer)
* Add POV-Ray cell frame radius option in Preferences
* Preferences option to automatically run filter list when less than specified number of atoms in lattice

**v0.11.1**

* Fix bug: AutoDetectReaderForm has no displayWarning method; use one from mainWindow instead

**v0.11**

* Add context menu to systems list widget
* Add display name to system (shown in pipeline combo)
* Add ability to duplicate loaded system
* Add BCC lattice generator
* Fix picker bug: include pick pos when setting max/min pos for spatial decomposition
* Add rock salt (MgO) lattice generator
* Add fluorite lattice generator (HfO2, PuH2, ...)
* Reset counters (number visible) when removing actors from filter list
* Automatically run filter lists on systems with less than 5000 atoms
* Add option to reload a system (if you edit a lattice file that has already been loaded)

**v0.10.1**

* Add 'invert selection' option to crop filter

**v0.10**

* New documentation
  
  * Sphinx documentation
  * Displayed in QtWebKit browser

**v0.9.4**

* Ignore PBCs when picking atoms (was resulting in atoms on opposite boundary being picked)
* Added Miao Yu's changes to atoms/bonds data files

**v0.9.3**

* Only create one info dialog per object even if clicked multiple times
* Object highlighting rewritten to work better with multiple renderer windows/pipelines
* Info windows close automatically when no longer relevant

**v0.9.2**

* Atom/defect info windows open near cursor but not over the atom/defect
* Auto detect input file format
* New logging mechanism, much better logging to stream and console window
* Option to save console window output to file
* Option to set console window logging level (DEBUG, INFO, etc) in preferences
* Option to resize main window to default size

**v0.9.1**

* Added Q4 filter (untested!)
* Adding highlighting picked defects
* Highlighting works much better

**v0.9**

* Added ability to load multiple files at once from file dialog
* Added ability to remove files that have already been loaded
* Multiple files can be selected for removal at once
* Fixed bug in picker, now works much better
* Picker now works by single press not double click
* Fix separate bug to do with picking antisites
* Added option to rotate camera around lattice (note: up/down not working well)
* Cannot load the same file more than once

**v0.8.1**

* Add 'flv' container option when creating movie (make it default too)
* flv can be embedded in pdf with LaTeX media9 package
* Movied ffmpeg container setting to output dialog from preference dialog

**v0.8**

* Added Voronoi tessellation using Voro++
* Voronoi cells can be drawn around visible atoms
* Can filter by Voronoi volume and number of neighbours (num faces on Voronoi cell)
* Can write out Voronoi volumes and num neighbours to file
* Currently only works well with PBCs
* Voronoi tessellation only recalculated if Voronoi settings have changed
* Option added to cluster filter to calculate volumes of clusters by summing Voronoi volumes of the atoms

**v0.7.5**

* Colouring options work with defect filter
* Moved movie framerate/filename options onto sequencer/rotate pages
* Add camera settings dialog for manually inputting position, etc.

**v0.7.4**

* Bug fix: read ref not setting refState on XYZ reader properly
* Got rid of annoying invalid drawable warning
* Antisite occupying atoms rendered using their pos, not ref pos of antisite

**v0.7.3**

* Updated atoms/bonds files (Kenny's changes)
* Scalars array modified when running subsequent filters
* Option to change working directory
* Bonds options now work in additional pipelines
* Sequencer fixed when using xyz files
* Sequencer works with filename with numbers in the prefix (as long as not at the end)
* Fix bug in crop sphere settings (set to centre of lattice button)

**v0.7.2**

* Update parsing of pyhull output to get volume/facet area
* Update to latest version of pyhull

**v0.7.1**

* Element editor now works with changes

**v0.7**

* Add ability to generate lattices (FCC and Pu-Ga so far)
* Add ability to load multiple files
* Can have different ref/input lattices on different pipelines; easy to switch between
* Option to write full lattice or just visible atoms
* Make scalar bar text white when background is black
* PBC settings is an attribute of pipeline
* Highlight atom when it is double clicked (pretty basic at the moment)
* Add antialiasing options to renderer window

**v0.6.1**

* Put quotes round filenames before unzipping

**v0.6**

* Convert to PySide (from PyQt4)
* Better detection of errors during file input
* Preferences option to specify paths to POV-Ray/Ffmpeg (persistent)
* Option to have black or white background
* Bug fix in read lbomd.IN method

**v0.5.4**

* Bug fix: render split interstitials when using POV-Ray
* Add basic splash screen

**v0.5.3**

* Fix bug when reading lbomd.IN file
* Able to specify custom povray/ffmpeg paths/executables
* Added "black background" option
* Added scale atom sizes option to display options on filter list

**v0.5.2**

* Fix bug in colouring of onAntisite atoms when ref/input specie lists differ

**v0.5.1**

* Store mainWindow size and working directory on exit and reload on startup
* Add option to exit message box to clear global settings
* Add progress bar and cancel button to rotator
* Rotator reinits VTK window at every step (looks better)
* Rotator always returns to original camera (even if cancelled/failed)

**v0.5**

* Implement MDI with multiple render windows
* Ability to have multiple analysis (filter) pipelines
* Always look for roulette file (not just in sequencer)
* Added coordination number filter
* Tidied up menus and toolbars
* Convert C libraries from SWIG to ctypes

**v0.4.2**

* Added option to draw bonds between visible atoms
* Added preferences dialog for POV-Ray, ffmpeg, matplotlib, etc options
* Fix POV-Ray rendering in sequencer/rotate
* Added vacancy display options to defect filter

**v0.4.1**

* Added RDF plotter

**v0.4**

* Rewritten file input so that reference and inputs can be different types (eg. lattice reference and xyz input)
* Use pyhull module to interface with qhull instead of subprocess calls
* Can have the same filter multiple times in the same filter list
* Sequencer output files are always numbered 0,1,2,... regardless of start or increment
* One slice plane per slice filter

**v0.3.3**

* Version number automatically determined using "git describe"
* Text position dialogs made modal with "Ok" button
* If a filter list is cleared or a filter removed its settings window is closed
* Use pyhull module to interface with qhull instead of subprocess calls
* Added slice filter

**v0.3.2**

* Added option to show "Energy barrier" on screen (if Roulette file available)

**v0.3.1**

* Fix bug in picker

**v0.3**

* Added picker: double clicking atom/defect shows info window about what you just clicked
* Small change to colouring options
  
  - PE, KE, Q options are always available
  - Displacement (etc) only available if that filter is selected

**v0.2**

* Recognise split interstitials (this can be turned on/off)
  
  - Note the defect cluster filter does not work with this option selected (currently)
* Added options to colour by PE, KE, Q, displacement
* Read time from Roulette files during lattice sequencer assuming Roulette file is:
  
  - in current directory and named like Roulette%d.dat
  - in ../Step%d/Roulette.dat

**v0.1**

* Fix element editor never giving focus back
* Added colouring options (height, solid colour)
* Added scalar bar
* Added on-screen information
  
  - Including atom count, visible count, defect count, (defect) specie count, time
  - Optionally place in top left or top right corner
* Added option to overlay on-screen information and scalar bar onto POV-Ray image
