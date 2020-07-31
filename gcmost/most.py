#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Geospatial Set Cover using a modified Greedy algorithm
# 
# Author: Jeff Nivitanont, GeoCARB Research Associate
# 
# Location: University of Oklahoma
# 
# Packages: netCDF4, pandas, shapely, geopandas, matplotlib, cartopy, numpy, descartes

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import shapely.geometry as sgeom
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from descartes.patch import PolygonPatch
import multiprocessing as mp
import sys
import os
import warnings
import joblib
import time
import gcmost
from gcmost.gc_set_cover import *

moddir = os.path.dirname(gcmost.__file__)
debug = False
##====included files=======
candidateBlocks = os.path.join(moddir, 'data/scan_blocks_5min_pruned.pkl')
baselineFile = os.path.join(moddir, 'data/baseline_covering_set_pandas.pkl')
modisDefault = os.path.join(moddir,'data/modis_wsa_0.5deg_2007_monthly.pkl')
##====included files====END
intro = '''\n\n\n
\t GeoCarb Mission Observation Scenario Tool (MOST)
\t created by Jeff Nivitanont, U. of Oklahoma (2018)
\t ref. Nivitanont et al (2019)
\n\n\n'''

def main(menu):
    '''
    Main program for generating a GeoCarb scanning strategy.
    Params:
    - menu: (str) pointer to main menu file.
    Returns:
        A daily GeoCarb scanning strategy.
    '''
    starttime = time.time()
    print(intro)
    time.sleep(2)
    ## LOAD MENU VARS
    print('Loading menu...')
    time.sleep(1)
    assert isinstance(menu, str), 'Please provide a valid string pointer to the main menu.'
    g = globals()
    loadmenu(menu, g)
    ## USER CONFIRMATION
    userconfirmation()
    ## START MAIN PROGRAM
    if enableParallel:
        num_cores = mp.cpu_count()
    #end if
    if num_cores>1:
        num_cores = num_cores-1 #save a CPU for other tasks
    else:
        num_cores = 1
    #end ifelse
    directory = './output/'
    if not saveDirectory is None:
        directory = saveDirectory 
    #end if
    if not os.path.exists(directory):
        print(f'Creating save directory {directory}')
        os.makedirs(directory)
    #end if
    meshgridSaveFid = directory+'meshgrid.pkl'
    afWindowSaveFid = directory+f'af_window_{year}{month:02}{day}.pkl'
    timewindowSaveFid = directory+f'timewindow_{year}{month:02}{day}.pkl'
    ## LOAD MODIS ALBEDO 0.5DEG DATA
    if modisAlbedo is None:
        alb_map = joblib.load(modisDefault)[f'{year}{month:02}{day}']
    else:
        modis = Dataset(modisAlbedo, 'r')
        alb_map = modis.variables[f'{year}{month:02}{day}'][:]
            #MODIS data is oriented (x,y) = (0,0) at bottom left corner
        modis.close()
    ## LOAD CANDIDATE BLOCKSET
    if scanBlockGeoms is None:
        blockset = pd.read_pickle(candidateBlocks)
    else:
        nc_blocks = Dataset(scanBlockGeoms, 'r') 
        centroid_lat_lon = [group for group in nc_blocks.groups] 
        ## create geometries from scan block info using the corners
        pgon = list()
        for i in range(len(centroid_lat_lon)):
            temp_lon_arr = (nc_blocks.groups[centroid_lat_lon[i]].variables['longitude_centre'][:].T-360.)
            temp_lon_crns = [temp_lon_arr[0,-1],  #lower left corner
                             temp_lon_arr[0,0],   #lower right corner
                             temp_lon_arr[-1,0],  #upper right corner
                             temp_lon_arr[-1,-1]] #upper left corner
            temp_lat_arr = (nc_blocks.groups[centroid_lat_lon[i]].variables['latitude_centre'][:].T)
            temp_lat_crns = [temp_lat_arr[0,-1],  #lower left corner
                             temp_lat_arr[0,0],   #lower right corner
                             temp_lat_arr[-1,0],  #upper right corner
                             temp_lat_arr[-1,-1]] #upper left corner
            temp_scan = zip(temp_lon_crns, temp_lat_crns) #(lon, lat) to conform to Cartesian (x, y)
            pgon.append(sgeom.Polygon(temp_scan))
        #reframe as GeoSeries
        blockset = gpd.GeoDataFrame({'centroid_lat_lon': centroid_lat_lon, 'geometry': pgon}, crs=latlon)
        blockset = blockset.to_crs(geo_proj4) #convert coords
        blockset = blockset[blockset['geometry'].is_valid].reset_index(drop=True) #takes out the blocks that scan into space
        blockset = blockset.drop(np.where(blockset['geometry'].area > np.mean(blockset.area))[0]).reset_index(drop=True) #drop blocks were generated with errors(size too large)
        if not all(blockset.is_valid):
            raise RuntimeError("Invalid geometries in candidate block set could not be eliminated.")
        #end if
    #end ifelse
    ## CREATE BACKGROUND GEOMETRIES
    df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    df.loc[df['name'].isin(['Panama', 'Trinidad and Tobago']), 'continent'] = 'South America' #technically caribbean
    n_am = df.query('continent == "North America"')
    caribbean = ['Bahamas', 'Cuba', 'Dominican Rep.', 'Haiti', 'Jamaica', 'Puerto Rico', 'Trinidad and Tobago']
    n_am = n_am[np.logical_not(n_am['name'].isin(caribbean))][['continent', 'geometry']] #remove caribbean islands
    s_am = df.query('continent == "South America"')[['continent', 'geometry']]
    frguiana = gpd.GeoDataFrame({'continent': 'South America', 'geometry': df[df['name'] == 'France'].intersection(whembox)}) #french guiana is listed under France < Europe
    s_am = s_am.append(frguiana)
    westhem = n_am.append(s_am)
    continents = westhem.dissolve(by='continent')
    extras = sgeom.box(-79.5, 23, -75, 27.5) #Northern Bermuda islands
    n_am = n_am.difference(extras)
    continents = continents.intersection(whembox)
    ## GENERATE A COST MESH/GRID
    if os.path.exists(meshgridSaveFid):
        try:
            cost_mesh = joblib.load(meshgridSaveFid)
        except FileNotFoundError:
            print(f'Unable to load {meshgridSaveFid}. Will calculate mesh...')
            mesh_gdf = gpd.GeoDataFrame(
                            {'lon': flatgrid5deg[:, 0],
                            'lat': flatgrid5deg[:, 1],
                            'geometry': [sgeom.Point(p) for p in flatgrid5deg]},
                            crs=latlon)
            cost_mesh = mesh_gdf[mesh_gdf.intersects(continents.unary_union)]
            cost_mesh = cost_mesh.reset_index()
            joblib.dump(cost_mesh, meshgridSaveFid)
            print(f'Saved to: {meshgridSaveFid}')
        #end try
    else:
        print('Calculating mesh...')
        mesh_gdf = gpd.GeoDataFrame(
                        {'lon': [p[0] for p in flatgrid5deg],
                        'lat': [p[1] for p in flatgrid5deg],
                        'geometry': [sgeom.Point(p) for p in flatgrid5deg]},
                        crs=latlon)
        cost_mesh = mesh_gdf[mesh_gdf.intersects(continents.unary_union)]
        cost_mesh = cost_mesh.reset_index(drop=True)
        joblib.dump(cost_mesh, meshgridSaveFid)
        print(f'Saved to: {meshgridSaveFid}')
    #end ifelse
    ## CALCULATE AIRMASS ON COST MESH
    if forceCalcs:
        print('Calculating scan window.')
        timewindow = calc_scan_window(
                                year,
                                month,
                                day,
                                start_th=scanStartAirmassThreshold,
                                end_th=scanEndAirmassThreshold,
                                start_coords=scanStartRefCoords,
                                end_coords=scanEndRefCoords)
        joblib.dump(timewindow, timewindowSaveFid)
        print(f'Done. Saved to: {timewindowSaveFid}')
        print('Calculating airmass mesh.')
        if num_cores>1:
            af_window=calc_afmesh_window2(cost_mesh, timewindow, num_cores)
        else:
            af_window=calc_afmesh_window(cost_mesh, timewindow)
        #end ifelse
        joblib.dump(af_window, afWindowSaveFid)
        print(f'Done. Saved to: {afWindowSaveFid}')
    elif not ((timewindowFile is None) or (afWindowFile is None)):
            try:
                timewindow = joblib.load(timewindowFile)
            except:
                raise FileNotFoundError(f'Unable to load "{timewindow}". Check file directory.')
            #end try
            try:
                print('Loading airmass mesh.')
                af_window = joblib.load(afWindowFile)
            except:
                raise FileNotFoundError(f'Unable to load "{afWindowFile}". Check file directory.')
            #end try
    else:
        if timewindowFile is None:
            print('Calculating scan window.')
            timewindow = calc_scan_window(
                                year,
                                month,
                                day,
                                start_th=scanStartAirmassThreshold,
                                end_th=scanEndAirmassThreshold,
                                start_coords=scanStartRefCoords,
                                end_coords=scanEndRefCoords)
            joblib.dump(timewindow, timewindowSaveFid)
            print(f'Done. Saved to: {timewindowSaveFid}')
        else:
            try:
                print('Loading scan window.')
                timewindow = joblib.load(timewindowFile)
            except FileNotFoundError:
                print(f'Unable to load {timewindowFile}. Will recalculate scan window.')
                print('Calculating scan window.')
                timewindow = calc_scan_window(
                                year,
                                month,
                                day,
                                start_th=scanStartAirmassThreshold,
                                end_th=scanEndAirmassThreshold,
                                start_coords=scanStartRefCoords,
                                end_coords=scanEndRefCoords)
                joblib.dump(timewindow, timewindowSaveFid)
                print(f'Done. Saved to: {timewindowSaveFid}')
            #end try
        #end ifelse
        print('Calculating airmass mesh.')
        if num_cores>1:
            af_window=calc_afmesh_window2(cost_mesh, timewindow, num_cores)
        else:
            af_window=calc_afmesh_window(cost_mesh, timewindow)
        #end ifelse
        joblib.dump(af_window, afWindowSaveFid)
        print(f'Done. Saved to: {afWindowSaveFid}')
    #end ifelse
    cost_mesh = cost_mesh.join(pd.DataFrame(af_window.T, columns=timewindow))
    min_mesh = np.amin(af_window, axis=0)
    ## PRUNE SCANNING FOV FOR AIRMASS
    zone5=cost_mesh.iloc[min_mesh<=5.0].buffer(.501) #create a "zone" of Airmass <= 5.0
    U=continents.intersection(zone5.buffer(.501).unary_union) #trim the scanning area to the zone
    U = U.to_crs(geo_proj4).buffer(0)
    print('Scan will start at',timewindow[0])
    blockset = blockset[blockset.intersects(U.unary_union)].reset_index(drop=True) #drop blocks that don't cover any land
    coverage = blockset.unary_union.buffer(0)
    if not coverage.contains(U.unary_union):
        if trimUniverseSet:
            diff = U.difference(coverage) # trim the Universe set
            pctCover = (1-diff.unary_union.area/U.unary_union.area)*100
            print(f'Candidate block set covers {pctCover:2.02f}% of N.am and S.am between 50N and 50S. Will trim universe set to fit...')
            universe_set = U.difference(diff.buffer(1000))
        else:
            raise ValueError('Check candidate block set for full coverage of scanning area.')
        #end ifelse
    #end if
    cost_mesh = cost_mesh.to_crs(geo_proj4)
    min_mesh = gpd.GeoDataFrame(
                    {'lat': cost_mesh['lat'],
                    'lon': cost_mesh['lon'],
                    'min.airmass': min_mesh,
                    'geometry': cost_mesh['geometry']},
                    crs=geo_proj4)
    warnings.filterwarnings('ignore')
    #end if
    ## LOAD AREA OF INTEREST/RESERVED SCAN TIME/CLOUD PROBABILITY MAPS, IF GIVEN
    if not areaOfInterestFile is None:
        areaOfInterest = joblib.load(areaOfInterestFile)
        areaOfInterest = areaOfInterest.unary_union
    else:
        areaOfInterest = None
    if not reservedScanBlocksFile is None:
        reservedScanBlocks = joblib.load(reservedScanBlocksFile)
    else:
        reservedScanBlocks = None
    if not cloudProbabilityMapFile is None:
        cloudProbabilityMap = joblib.load(cloudProbabilityMapFile)
    else:
        cloudProbabilityMap = None
    assert not debug, 'debugging in process' #stop here if debug
    # MAIN SCANNING ALGORITHM
    print('Covering South America')
    cover_sam = greedy_gc_cost(
                        blockset=blockset,
                        universe_set=universe_set['South America'],
                        albedo=alb_map,
                        mesh_df=cost_mesh,
                        min_airmass=min_mesh,
                        interest=areaOfInterest,
                        precip=cloudProbabilityMap,
                        reservations=reservedScanBlocks,
                        tol=universeCoverageTol/2,
                        setmax=len(timewindow),
                        weights=(weightDistPenalty, weightOverlapPenalty),
                        dist_thr=distanceThreshold)
    nam_new = universe_set['North America'].difference(cover_sam.unary_union)
    print('Covering North America')
    cover_nam = greedy_gc_cost(
                        blockset=blockset,
                        universe_set=nam_new,
                        albedo=alb_map,
                        mesh_df=cost_mesh,
                        min_airmass=min_mesh,
                        interest=areaOfInterest,
                        precip=cloudProbabilityMap,
                        reservations=reservedScanBlocks,
                        t=len(cover_sam),
                        tol=universeCoverageTol/2,
                        setmax=len(timewindow)-len(cover_sam),
                        weights=(weightDistPenalty, weightOverlapPenalty),
                        dist_thr=distanceThreshold)

    coverset = gpd.GeoDataFrame({},crs=geo_proj4)
    coverset = coverset.append([cover_sam, cover_nam]).reset_index(drop=True)
    coverset['time'] = timewindow[:len(coverset)]
    coverset = coverset[['time', 'centroid_lat_lon', 'geometry', 'footprint_centroids']]
    ## SAVE OUTPUTS
    coversetSaveFid = directory+'coverset_'+timewindow[0].strftime('%Y%m%d_%H%M')+'_pandas.pkl' 
    coverset.to_pickle(coversetSaveFid)
    print(f'Coverset saved to: {coversetSaveFid}')
    blocklistSaveFid = directory+'blocklist_'+timewindow[0].strftime('%Y%m%d_%H%M')+'.txt'
    namelist=coverset['centroid_lat_lon']
    with open(blocklistSaveFid, 'w') as output:
        output.write(namelist.to_string())
    print(f'Blocklist saved to: {blocklistSaveFid}')

    ## SCAN EXTRA TIME
    if scanExtraTime and len(coverset) < len(timewindow):
        print('Covering extra time')
        coverset_extra = greedy_gc_cost(blockset=blockset,
                                        universe_set=universe_set.unary_union,
                                        albedo=alb_map,
                                        mesh_df=cost_mesh,
                                        min_airmass=min_mesh,
                                        interest=areaOfInterest,
                                        precip=cloudProbabilityMap,
                                        reservations=reservedScanBlocks,
                                        t=len(coverset),
                                        tol=universeCoverageTol,
                                        setmax=len(timewindow)-len(coverset),
                                        weights=(weightDistPenalty, weightOverlapPenalty),
                                        dist_thr=distanceThreshold)
        coverset_extra['time'] = timewindow[len(coverset):]
        coverset_extra = coverset_extra[['time', 'centroid_lat_lon', 'geometry', 'footprint_centroids']]
        #save outputs
        print('Saving extra scan blocks.')
        coversetExtraSaveFid = directory+'coverset_extra_'+timewindow[0].strftime('%Y%m%d_%H%M')+'_pandas.pkl'
        coverset_extra.to_pickle(coversetExtraSaveFid)
        blocklistExtraSaveFid = directory+'blocklist_extra_'+timewindow[0].strftime('%Y%m%d_%H%M')+'.txt'
        namelist=coverset_extra['centroid_lat_lon']
        with open(blocklistExtraSaveFid, 'w') as output:
            output.write(namelist.to_string())

    ## PLOT RESULTS
    if plotCoverset:
        fig = plotCoversetStatic(coverset, figsize=(8, 8))
        plt.savefig(directory + 'coverset_' + timewindow[0].strftime('%Y-%m-%d.png'))
        plt.close()
        if scanExtraTime:
            try:
                fig = plotCoversetStatic(coverset_extra)
                plt.savefig(directory + 'coverset_extra_' + timewindow[0].strftime('%Y-%m-%d.png'))
                plt.close()
            except:
                pass
    if createMov:
        mov = createScanMov(cost_mesh, coverset)
        mywriter = animation.FFMpegWriter(fps=5)
        mov.save(directory+'coverset_'+timewindow[0].strftime('%Y-%m-%d.mov'), writer=mywriter)
        if scanExtraTime:
            try:
                mov = createScanMov(cost_mesh, coverset_extra)
                mov.save(directory+'coverset_extra_'+timewindow[0].strftime('%Y-%m-%d.mov'), writer=mywriter)
            except:
                pass

    ## RUN DIAGNOSTIC
    if compareToBaseline:
        baseline = pd.read_pickle(baselineFile)
        baseline['time'] = timewindow[:len(baseline)]
        fig = compareBaseline(alb_map, cost_mesh, coverset, baseline)
        plt.savefig(directory +'diagnostics_coverset_vs_baseline_'+timewindow[0].strftime('%Y-%m-%d.png'))
        plt.close()
    endtime = time.time()
    totaltime = endtime-starttime
    print(f'Total time elapsed in seconds: {totaltime}')
    return