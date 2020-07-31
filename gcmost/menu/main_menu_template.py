##====GeoCarb MOST main menu====

##/--DATE--/
year = 2007
month = 6
day = 21 #Default is 21 for equinoxes and solstices.
#   Change only if appropriate MODIS file available.

##/--OPTIONAL FILE POINTERS--/
modisAlbedo = None #Default file contains only solstices and equinoxes of 2007
saveDirectory = None #Set save directory. If None, outputs will be saved in './output/'
scanBlockGeoms = None # This should point to candidate scan blocks (fine, medium, coarse resolution)
timewindowFile = None  #Supply a pre-calculated timewindow. If None, timewindow is calculated.
afwindowFile = None  #Supply a pre-calculated airmass window. If None, grids are calculated. 
#   If af_window is provided without timewindow, will be recalculated.
areaOfInterestFile = None # test with 'data/mostpopcities_whem.pkl'
reservedScanBlocksFile = None # test with 'data/testreservation.pkl'
cloudProbabilityMapFile = None # test with 'data/testprecip.pkl'

##/--COMPUTING OPTIONS--/
enableParallel = True #Recommend turning on. A single 2.5Ghz CPU, ~7-10mins to calculate airmass grid.
forceCalcs = False #force calculation of the airmass grid
trimUniverseSet = True  #Default: True. Scan blocks should roughly cover area of interest.
#    This aids in aliasing issues. 

##/--SCANNING OPTIONS--/
scanExtraTime = True #use extra daylight in scan?
scanStartAirmassThreshold = 3.0
scanEndAirmassThreshold = 5.0
scanStartRefCoords = (0, -50) #Macapa, BR (0,-50)
scanEndRefCoords = (19.5, -99.25) #Mexico City, MX (19.5,-99.25)
weightDistPenalty = 1.0
weightOverlapPenalty = 1.0
distanceThreshold = 7  #How many 5-min scan blocks in the E-W direction
#    to consider per algorithm step?
universeCoverageTol = 0.002 #allowable limit of uncovered area

##/--PLOTTING OPTIONS--/
plotCoverset = True
createMov = True #create a movie of the scan?
compareToBaseline = True #plot diagnostic histograms of xco2_uncert, signal-to-noise ratio,
#    solar zenith angle, and satellite zenith angle.

##====end main menu====