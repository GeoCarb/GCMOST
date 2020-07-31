#!/usr/bin/env python
# -*- coding: utf-8 -*-

# gc_set_cover.py // Functions used for set cover problem applied to GeoCARB mission
# updated 2020-07-30
# @author: Jeff Nivitanont, University of Oklahoma

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import animation
from descartes.patch import PolygonPatch
import datetime as dt
import time
from numba import jit
from shapely.ops import transform
from functools import partial
import pyproj
import shapely.geometry as sgeom
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import pandas as pd
import geopandas as gpd
from itertools import product
import joblib

## CONSTANTS FOR EARTH CALCULATIONS
geo_lon = -85.0    #satellite lon
geo_lat = 0.0      #satellite lat
geo_ht = 42.336e6 #fixed satellite height
surf_ht = 6.370e6 #uniform surface height
sol_ht = 149.6e9 #fixed solar height
r_earth = 6.371e6 # radius of Earth
north_solst = 354 # date of southern summer solstice
declin = 23.44
earth_sun = 149.6e9 #distance to sun
atrain_alt = 705e3 
dtor = np.pi/180.0   #convert degs to rads

#mesh grid coords lat/lon that includes region of interest
whembox = sgeom.box(-130, -50, -30, 50)
x = np.linspace(-130, -30, 201)
y = np.linspace(50, -50, 201)
xv, yv = np.meshgrid(x, y)

#for converting lat/lon to Geo
geo = ccrs.Geostationary(central_longitude=geo_lon, satellite_height=geo_ht)
geo_proj4 = geo.proj4_init
latlon = 'EPSG:4326'

def userconfirmation():
    '''Helper function to read command line input'''
    print('Review menu options and confirm (Y/N + enter) to start program.')
    response = input()
    if response.strip().lower() in ['y','yes', '1']:
        print('Starting program.')
    else:
        sys.exit('Exiting program.')
    return

def loadmenu(menudir, scope):
    '''helper function to load user defined menu vars'''
    with open(menudir) as f:
        for line in f:
            if not line.startswith('\n'):
                if not line.startswith('#'):
                    key, val = line.split('=')
                    exec(f'{key}={val}', scope)
                print(line)
    return

def cartesian(x,y):
    '''
    Cartesian product of two iterables.
    '''
    return np.array(list(product(x,y)))

flatgrid5deg = cartesian(x,y)

def latlon_to_geo(geom):
    '''
    This function takes a Shapely Geometry projected in lat/lon and returns a Shapely Geometry projected in Geostationary.
    NOTE: geom.crs should be initialized before using this function.
    
    Params:
        - geom: Shapely Geometry projected in lat/lon.
    
    Return: A Shapely Geometry projected in Geostationary.
    '''
    #partially applied function from pyproj for converting from lat/lon to geo
    project = partial(pyproj.transform, pyproj.Proj(latlon), pyproj.Proj(geo_proj4) )
    return transform(project, geom)

def geo_to_latlon(geom):
    '''
    This function takes a Shapely Geometry projected in Geostationary and returns a Shapely Geometry projected in lat/lon.
    NOTE: geom.crs should be initialized before using this function.
    
    Params:
        - geom: Shapely Geometry projected in Geostationary.
    
    Return: A Shapely Geometry projected in lat/lon.
    '''
    #partially applied function from pyproj for converting from lat/lon to geo
    project2 = partial(pyproj.transform, pyproj.Proj(geo_proj4), pyproj.Proj(latlon))
    return transform(project2, geom)


def airmass_serial(lat, lon, time):
    '''
    This function gives the airmass at a lat,lon point of time.
    
    Params:
    - lat: latitude in degrees.
    - lon: longitude in degrees.
    - time: time, DateTime object.
    
    Return:
    - airmass factor, float.
    '''
    start_day = dt.datetime(time.year,1,1,0,0,0)
    view_lat = lat #viewer on ground
    view_lon = lon
    day_dec = (time-start_day).total_seconds()/86400.  #decimal julian date for scalar_subsol func
    subsol_lat,subsol_lon = scalar_subsol(day_dec) #sub-solar lat/lon
    sol = np.array([np.array([sol_ht,subsol_lat,subsol_lon])])
    ex = np.array([np.array([surf_ht,view_lat,view_lon])])
    sol_zenith = zenith_angle_cosine_batch(ex,sol) #(viewer, target)
    sat = np.array([np.array([geo_ht,0.,geo_lon])])
    sat_zenith = zenith_angle_cosine_batch(ex,sat)
    if( sol_zenith > 90. or sat_zenith > 90.): #impossible case
        return 9999.
    else:
        return 1./np.cos(dtor*sol_zenith) + 1./np.cos(dtor*sat_zenith)

airmass = np.vectorize(airmass_serial)
        
#from earth_calcs.py (by Peter Rayner, U. Melbourne)
@jit
def scalar_earth_angle(lat1, lon1, lat2, lon2):
    """ angle in degrees on great circle between two points """
    theta1 = lat1 *dtor
    phi1 = lon1 *dtor
    theta2 = lat2 * dtor
    phi2 = lon2 * dtor
    p1 = np.vstack((np.cos(theta1)*np.cos(phi1),np.cos(theta1)*np.sin(phi1),np.sin( theta1))).T
    p2 = np.vstack((np.cos(theta2)*np.cos(phi2), np.cos( theta2)* np.sin( phi2), np.sin( theta2))).T
    dsq = ((p1-p2)**2).sum(-1)
    return np.arccos((2 -dsq)/2.)/dtor

#from earth_calcs.py (by Peter Rayner, U. Melbourne)
@jit
def scalar_subsol(day):
    '''subsolar lat-lon given decimal day-of-year '''
    lat = -declin*np.cos(2*np.pi* np.mod(365 + day - north_solst,  365.)/365.)
    lon = 180-np.mod(360.*(day-np.floor(day)), 360.)
    return lat, lon

#from earth_calcs.py (by Peter Rayner, U. Melbourne)
@jit
def zenith_angle_cosine_batch(viewer, target):
    """ gives the zenith angle of a target  (r theta, phi) from the viewer (r, theta phi), theta, phi and result are in degrees"""
    centre_angle = scalar_earth_angle( viewer[:,1],viewer[:,2], target[:,1], target[:,2]) # angle between the two locations at centre of earth
    dist = (viewer[:,0]**2 + target[:,0]**2 -2.*target[:,0]*viewer[:,0]*np.cos( centre_angle*dtor))**0.5 # cosine rule
    cos_zenith = -0.5*(dist**2+viewer[:,0]**2-target[:,0]**2)/(dist*viewer[:,0])  # the minus makes it a zenith angle
    return np.arccos(cos_zenith)/dtor

def calc_scan_window(year, month, day, start_th=3., end_th=5., start_coords=(0,-50), end_coords=(19.5,-99.25)):
    '''
    Calculates optimal start and finish time for a scan. Defaults to Macapa, Brazil and Mexico City, Mexico as points of reference.
    Runtime ~5mins.

    Params:
        - year: (int) Year.
        - month: (int) Month.
        - day: (int) Day.
        - start_th: (float) Specified Airmass Factor threshold with which to start the scan.
        - end_th: (float) Specified Airmass Factor threshold with which to end the scan.
        - start_coords: (lat, lon) location for determining scan start time. 
        - end_coords: (lat, lon) location for determining scan end time.
    Return:
        - time_intv: list of datetimes for scan.
    '''
    time_intv = list([dt.datetime(year, month, day, 0)])
    for i in range(1,288):
        time_intv.append(time_intv[i-1]+dt.timedelta(0,300))
    startpt_af = airmass(lat=start_coords[0], lon=start_coords[1], time=time_intv)
    endpt_af = airmass(lat=end_coords[0], lon=end_coords[1], time=time_intv)
    start_ind = 0
    while(startpt_af[start_ind]>start_th):
        start_ind += 1
    end_ind = 287
    while(endpt_af[end_ind]>end_th):
        end_ind -= 1
    return pd.DatetimeIndex(time_intv[start_ind:(end_ind+1)])

def calc_afmesh_window(mesh, time_intv):
    '''
    Calculates airmass for a given mesh and time interval.

    Params:
        - mesh: (GeoDataFrame) a data frame containing lat/lon points of mesh.
        - time_intv: (list) list of times for calculating airmass on mesh.
    
    Return:
        - day_mesh: Numpy array of lat, lon points - [time, lat, lon].
    '''
    window_len = len(time_intv)
    lat = mesh['lat'].values
    lon = mesh['lon'].values
    day_mesh = np.full([window_len, len(lat)], 9999.)
    for i in range(window_len):
        day_mesh[i,:] = airmass(lat=lat, lon=lon, time=time_intv[i])
    day_mesh[day_mesh < 2.0] = 9999.
    return day_mesh

def calc_afmesh_window2(mesh, time_intv, ncores):
    '''
    Calculates airmass for a given mesh and time interval. Utilizes multiple cores for shorter runtime.

    Params:
        - mesh: (GeoDataFrame) a data frame containing lat/lon points of mesh.
        - time_intv: (list) list of times for calculating airmass on mesh.
    
    Return:
        - day_mesh: Numpy array of lat, lon points - [time, lat, lon].
    '''
    mesh_list=joblib.Parallel(n_jobs=ncores)(joblib.delayed(airmass)(lat=mesh['lat'].values, lon=mesh['lon'].values, time = timestep) for timestep in time_intv)
    day_mesh=np.array(mesh_list)
    day_mesh[day_mesh < 2.0] = 9999.
    return day_mesh

def calc_xco2_err(
    albedo,
    mesh_df,
    scan_block,
    aod=0.3
):
    '''
    This function takes in a blockset and calculates sucess based on indicated threshold for Signal-Noise Ratio (SNR).
    
    Params:
        - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
        - scan_block: Pandas.DataFrame slice, Pandas.Series
        - mesh_df: lat/lon mesh grid.
        - aod: Aerosol Optical Depth, default = 0.3.
    
    Return:
        - array of lat/lon grid points where SNR passes threshold. [lon, lat, error, snr]
    
    '''
    block_geom = scan_block['geometry']
    time = scan_block['time']
    start_day = dt.datetime(time.year,1,1,0,0,0)
    af_intx_ind = mesh_df.intersects(block_geom)
    af_intx = mesh_df.loc[af_intx_ind]
    intxpts = af_intx[['lat', 'lon']].values #[y,x]
    if(len(intxpts.shape)!=2):                    #shapely intersection issue 
            return None
    pts_arr = (((intxpts[:,0]+90)*2).round().astype(int), ((intxpts[:,1]+180)*2).round().astype(int)) #lat(row), lon(col)
    albedo_arr = albedo[pts_arr]
    view_lat = intxpts[:,0] #viewer on ground
    view_lon = intxpts[:,1] 
    day_dec = (time-start_day).total_seconds()/86400.  #decimal julian date for scalar_subsol func
    subsol_lat, subsol_lon = scalar_subsol(day_dec) #sub-solar lat/lon
    sol = np.array([[sol_ht, subsol_lat,subsol_lon]])
    ex = np.array([np.repeat(surf_ht, len(view_lat)), view_lat, view_lon]).T
    sol_zenith = zenith_angle_cosine_batch(ex, sol) #(viewer, target)
    sat = np.array([[geo_ht, geo_lat ,geo_lon]])
    sat_zenith = zenith_angle_cosine_batch(ex, sat)
    af = 1./np.cos(sol_zenith*dtor) + 1./np.cos(sat_zenith*dtor)
    af[sat_zenith>90.] = 9999.
    af[sol_zenith>90.] = 9999.
    snr = np.zeros(sol_zenith.size)
    ## Noise model taken from Obrien et al (2016):
    #     Fsun = 2073 #nW cm sr^(-1)  cm^(-2)
    #     n0**2 = (0.1296)**2 = 0.016796159999999997
    #     n1 = 0.00175
    S = 2073*albedo_arr*np.cos(sol_zenith*dtor)*np.exp(-af*aod)
    N = np.sqrt(0.00175*S+0.016796159999999997)
    snr[S>0] = S[S>0]/N[S>0]
    sig_xco2 = 14./(1.+0.0546*snr)
    block_df = pd.DataFrame(
        {'block_id': np.repeat(scan_block['centroid_lat_lon'], len(snr)),
            'time': np.repeat(time, len(snr)),
            'lat': view_lat,
            'lon': view_lon,
            'error': sig_xco2,
            'SNR': snr,
            'SolZA': sol_zenith,
            'SatZA': sat_zenith})
    return block_df
    
def calc_set_err(albedo, mesh_pts, coverset):
    '''
    This function applies calc_xco2_err to each block and returns a Numpy array.
    
    Params:
        - coverset: a covering set from the Greedy Algorithm, GeoPandas.GeoDataFrame.
        - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
        - mesh_pts: lat/lon mesh grid.
    
    Return:
        - error_df: a Pandas.DataFrame
    '''
    error_df = calc_xco2_err(albedo, mesh_pts, coverset.iloc[0])
    for i in np.arange(1,len(coverset)):
        block_df = calc_xco2_err(albedo, mesh_pts, coverset.iloc[i])
        try:
            error_df = error_df.append(block_df, ignore_index = True)
        except ValueError:
            pass
    return error_df


def create_mask_err(error_df):
    '''
    This function creates an array of minimum XCO2 error. Areas not intersecting land are masked with np.nan for plotting.
    
    Params:
    - error_df: error Pandas.DataFrame output from `calc_set_error`.
    
    Return:
    - E: masked array of minimum errors.
    '''
    sorted_df = error_df.iloc[error_df['error'].argsort()].copy() #sorted in ascending order
    e_unqs = sorted_df.drop_duplicates(subset=['lat', 'lon']) #choose minimum error
    E = np.full(xv.shape, np.nan )
    e_pts_arr = [((50-e_unqs['lat'])*2.).round().astype(int), ((e_unqs['lon']+130)*2.).round().astype(int)]
                #lat(row), lon(col)
    E[e_pts_arr] = e_unqs['error'].copy()
    return E
                    
def calc_block_cost(
            block,
            universe,
            mesh_df,
            mesh_airmass,
            min_airmass,
            last,
            covered,
            albedo,
            interest=None,
            precip=None,
            weights=[1.0, 1.0]):
    '''
    Cost function for Greedy Algorithm
    
    (default): cost_d = avg(exp(af)/albedo)*(1+overlap+dist)/block.intersection(universe).area
    (with areas of interest): cost_a = np.exp(airmass - min_airmass - np.exp(2-min_airmass))*cost_d
    see Nivitanont et al. (2019) for details.

    Params:
        - block: candidate scan block.
        - universe: Shapely Geometry to be covered by block.
        - mesh_df: mesh coordinate points; GeoDataFrame of points
        - mesh_airmass: Airmass Factor scores at mesh coordinate points.
        - last: last selected block.
        - covered: union of selected blocks.
        - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
        - interest: a GeoDataFrame of points of interests (i.e. cities, forests, etc.)
        - precip: a precipitation probabilty map.
        - weights: an iterable object of weights, i.e. tuples, lists, arrays, etc. Must contain 2 items. (w_dist, w_overlap). Default is [1., 1.]
    Returns:
        the "cost" of a scanning block according to the equations above.
    '''
    coverage = block.intersection(universe).area
    if block.intersects(mesh_df.unary_union)==False or coverage<1.:
        return float('inf')
    else:
        af_ind = mesh_df.intersects(block)
        af_intx = mesh_df[af_ind]
        intxpts = af_intx[['lat','lon']].values
        if(len(intxpts.shape)!=2): #if no points
            return float('inf')
        else:
            af_arr = mesh_airmass[af_ind]
            cmg_ind = (((intxpts[:,0]+90)*2.).round().astype(int), ((intxpts[:,1]+180)*2.).round().astype(int)) #(lat/y/row), (lon/x/col)
            alb_arr = albedo[cmg_ind]
            avg_exp = np.nanmedian(np.exp(af_arr)/alb_arr)
            overlap = block.intersection(covered).area
            if overlap<block.area*.04: #overlap tolerance
                overlap = 0.0
            #end if
            dist = block.distance(last)
            cost = avg_exp*(1.0+weights[0]*dist**2+weights[1]*overlap)/coverage
            if not precip is None:
                cost = cost*np.nanmedian(precip[cmg_ind])
            #end if
            if not interest is None:
                if block.intersects(interest):
                    minaf_arr = min_airmass['min.airmass'].loc[af_ind]
                    city_scaling_th = minaf_arr+np.exp(2-minaf_arr)
                    cost = cost*np.nanmedian(np.exp(af_arr-city_scaling_th))
                #end if
            #end if
            return cost
        #end ifelse
    #end ifelse


def greedy_gc_cost(
        blockset,
        universe_set,
        albedo,
        mesh_df,
        min_airmass,
        interest=None,
        precip=None,
        reservations=None,
        t=0,
        tol=0.005,
        setmax=144,
        weights=[1.0, 1.0],
        dist_thr=6):
    '''
    This Greedy Algorithm selects scan blocks by the cost function, which utilizes the mesh inputs.
    block.
    
    Params:
        - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
        - universe_set: A Shapely Geometry object for the area required to be covered.
        - albedo: 360x720 albedo map, MODIS product wsa-band6/7.
        - mesh_df: A Shapely.MultiPoint object that contains the mesh pts in lat/lon to be passed to the cost function.
        - min_airmass: The minimum possible airmass factors for each 0.5deg grid point.
        - interest: A Shapely.MultiPolygon object that indicates areas of interest; Used for 'temporary campaign' mode.
        - precip: 360x720 precipitation probability map on 0.5deg grid, similar to the albedo map.
        - reservations: A GeoPandas.GeoDataFrame containing block centroid lat/lon, time, geometry information for
            reserved scan times.
        - t: Timestep to start algorithm (t*5-minutes) from beginning of time window.
        - tol: Tolerance for uncovered land.
        - setmax: Max covering set size.
        - weights: An iterable object of weights, i.e. tuples, lists, arrays, etc. Must contain 2 items. (w_dist, w_overlap). Default is [1., 1.]
        - dist_thr: Threshold number of blocks that are adjacent in the E-W direction to the last selected block to consider per algorithm step.
            The centroid distance to the last farthest block is used to define a cutoff radius for the centroids of candidate blocks.
            Default value is 6. Higher values result in longer run times. 
    Returns:
        A covering set for the universe set.
    '''
    
    if not blockset.is_valid.all():
        raise RuntimeError('Invalid geometries in blocks.')
    if (not reservations is None) and reservations['time'].duplicated().any():
        raise RuntimeError('Duplicate reservation times.')
    universe = universe_set
    timewindow = pd.DatetimeIndex(mesh_df.columns[3:])
    mesh_ind = mesh_df.intersects(universe)
    mesh_unv = mesh_df[mesh_ind]
    minaf_unv = min_airmass[mesh_ind]
    scan_blocks = blockset[blockset.intersects(universe)].reset_index(drop=True)
    init_area = universe.area
    if(universe_set.difference(blockset.unary_union).area > tol*init_area):
        raise RuntimeError('Candidate scan blocks do not cover universe set.')
    cover_set = gpd.GeoDataFrame({}, crs=geo_proj4)
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_cost = []
    sel_ind = None
    ii = 0
    print('Selecting scan blocks...', end=' ')
    while((universe.area>init_area*tol) and (ii<setmax)):
        print(t+1, end=' ')
        mesh_ind = mesh_unv.intersects(universe).values
        if (not reservations is None) and np.any(timewindow[t]==reservations['time']):
            sel_block = reservations.loc[timewindow[t]==reservations['time']]
            if len(sel_block)!=1:
                raise IndexError('Possible duplicate reservation times.')
            cover_set = cover_set.append(sel_block, ignore_index=True)
            covered = cover_set.unary_union.buffer(0)
            universe = universe.difference(covered).buffer(0)
            if not interest is None:
                interest = interest.difference(covered).buffer(0)
            lastgeom = sel_block.unary_union
        else:
            if(universe.area>init_area*0.05):   #if majority of area is uncovered
                cost_mesh_f = partial(calc_block_cost,
                                      universe=universe,
                                      albedo=albedo,
                                      mesh_df=mesh_unv.loc[mesh_ind],
                                      mesh_airmass=mesh_unv[timewindow[t]].loc[mesh_ind],
                                      min_airmass=minaf_unv.loc[mesh_ind],
                                      interest=interest,
                                      precip=precip,
                                      last=lastgeom,
                                      covered=covered,
                                      weights=weights)
                if ii==0:
                    dist_idx = np.full(len(scan_blocks), True)
                else:
                    # 200,000 is the geostationary coordinate width of one 5-min scan block
                    dist_idx = np.array(scan_blocks.centroid.distance(lastgeom.centroid)<dist_thr*200000.0)
                block_cost = np.full(len(scan_blocks), float('inf'))
                block_cost[dist_idx] = list(map(cost_mesh_f, scan_blocks[dist_idx]['geometry']))
                sel_ind = np.argmin(block_cost)
            else:
                cost_mesh_f = partial(calc_block_cost,
                                      universe=universe,
                                      albedo=albedo,
                                      mesh_df=mesh_unv.loc[mesh_ind],
                                      mesh_airmass=mesh_unv[timewindow[t]].loc[mesh_ind],
                                      min_airmass=minaf_unv.loc[mesh_ind],
                                      interest=interest,
                                      precip=precip,
                                      last=lastgeom,
                                      covered=covered,
                                      weights=weights)
                dist_idx = np.array(scan_blocks.intersects(universe)) #remove distance penalty
                block_cost = np.full(len(scan_blocks), float('inf'))
                block_cost[dist_idx] = list(map(cost_mesh_f, scan_blocks[dist_idx]['geometry']))
                sel_ind = np.argmin(block_cost)
            #end ifelse
            cover_set = cover_set.append(scan_blocks.iloc[sel_ind], ignore_index=True)
            covered = cover_set.unary_union.buffer(0)
            if not interest is None:
                interest = interest.difference(covered).buffer(0)
            universe = universe.difference(covered).buffer(0)
            lastgeom = scan_blocks.geometry[sel_ind]
            scan_blocks = scan_blocks.drop(sel_ind).reset_index(drop=True)
        ii = ii+1
        t = t+1
    #end while
    print('\n')
    return cover_set

def greedy_gc_unif(blockset, universe_set, tol= 0.0):
    '''
    This Greedy Algorithm gives uniform weight to all areas of land. Most basic weight function.
    
    Params:
        - blockset: A set of candidate scan blocks in GeoPandas.GeoDataFrame format.
        - universe_set: A Shapely Geometry object for the area required to be covered.
        - tol: allowable remaining uncovered area.
    Returns:
        A minimal covering set.
    '''
    if(universe_set.difference(blockset.unary_union).area > 0.0):
        print('Error: Blocks do not cover universe.')
        return
    if(all(blockset.is_valid)==False):
        print('Error: invalid geometries in blocks.')
        return
    scan_blocks = blockset
    universe = universe_set
    init_area = universe.area
    cover_set = gpd.GeoDataFrame()
    cover_set.crs = geo_proj4
    covered = sgeom.Point()
    lastgeom = sgeom.Point()
    block_weight = []
    max_ind = []
    ii=0
    while (universe.area>init_area*tol) and (ii<168):
        dist = scan_blocks.centroid.distance(lastgeom.centroid)
        block_weight = scan_blocks.intersection(universe).area - dist**2
        max_ind = block_weight.idxmax()
        lastgeom = scan_blocks.geometry[max_ind]
        cover_set = cover_set.append(scan_blocks.iloc[max_ind])
        covered = cover_set.unary_union
        universe = universe.difference(covered).buffer(0)
        scan_blocks = scan_blocks.drop(max_ind).reset_index(drop=True)
        ii+=1
    cover_set = cover_set.reset_index(drop=True)
    return cover_set

def plotCoversetStatic(coverset, **fig_kwargs):
    '''
    This function creates a static image of the resulting scanning strategy.

    Params:
        - coverset: resulting covering set produced by algorithm.
        - fig_kwargs: kwargs to pass to matplotlib.pyplot.figure.
    Returns:
        a matplotlib figure object.
    '''
    plt.style.use('seaborn-white')
    fig = plt.figure(**fig_kwargs)
    ax1 = plt.axes(projection=geo)
    ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                    linewidth=0.5,facecolor='silver', edgecolor='black')
    ax1.gridlines(zorder=3)
    ax1.axis('scaled')
    ax1.set_facecolor('paleturquoise')
    coverset.plot(ax=ax1, zorder=7, alpha=0.4, color=[f'{s:1.03}' for s in coverset.index/len(coverset)], cmap='plasma', edgecolor='black')
    plt.colorbar(cm.ScalarMappable(norm=Normalize(0,len(coverset)), cmap='plasma'), ax=ax1, label='scan order')
    return fig

def createScanMov(mesh_df, coverset, interest=None):
    '''
    This function creates an animated movie of the resulting scanning strategy.

    Params:
        - mesh_df: the calculated usable scan window geopandas.GeoDataFrame object.
        - coverset: resulting covering set produced by algorithm.
        - (optional) interest: a geopandas.GeoDataFrame containing areas of high interest. default: None.
    Returns:
        a matplotlib figure object.
    '''
    plt.style.use('seaborn-white')
    timewindow = coverset['time']
    fig = plt.figure(figsize=(10,10))
    ax1 = plt.axes(projection=geo)
    ax1.axis('scaled')
    ax1.gridlines(zorder=3)
    ax1.add_feature(cfeat.NaturalEarthFeature('cultural', 'admin_0_countries', '110m'),
                    facecolor='white', edgecolor='black', linewidth=0.3, zorder=0)
    coords=np.array([[geom.x, geom.y] for geom in mesh_df['geometry']])
    scatter = ax1.scatter(coords[:,0], coords[:,1], c=mesh_df[timewindow.iloc[0]], vmin=2, vmax=5, s=.5, cmap='viridis_r')
    plt.colorbar(cm.ScalarMappable(norm=Normalize(2,5), cmap='viridis_r'), ax=ax1,label='airmass factor')
    timeText = ax1.text(0.7, 1., '', transform=ax1.transAxes)
    counter = ax1.text(0.1, 1., '', transform=ax1.transAxes)
    if not interest is None:
        interest.plot(ax=ax1, color='red')
    #this function updates the plot given index i    
    def snapshot(i):
        scatter.set_array(mesh_df[timewindow[i]])
        ax1.add_patch(PolygonPatch(polygon=coverset.geometry[i], alpha=0.2, zorder=4, color='red'))
        counter.set_text(str(i+1) + ' Blocks')
        timeText.set_text(str(timewindow[i])+' GMT')
    anim = animation.FuncAnimation(fig, snapshot, frames=len(coverset))
    return anim

def histFooprints2d(coverset, baseline):
    coverset_lat = [np.array(geom)[:,1] for geom in coverset['footprint_centroids']]
    coverset_lon = [np.array(geom)[:,0] for geom in coverset['footprint_centroids']]
    baseline_lat = [np.array(geom)[:,1] for geom in baseline['footprint_centroids']]
    baseline_lon = [np.array(geom)[:,0] for geom in baseline['footprint_centroids']]
    #.5 degree width bins with grid points in middle
    #    (should probably change this domain to include whole map later)
    xbins = np.linspace(-130.25, -29.75, 202)
    ybins = np.linspace(-50.25, 50.25, 202)
    #Binning the footprints for counts
    h_list = np.array([np.histogram2d(arr_list[0].ravel(order='F'), arr_list[1].ravel(order = 'F'), bins=[xbins, ybins])[0]
                              for arr_list in list(zip(coverset_lon, coverset_lat))])

    H = np.amax(h_list, axis=0)
    H = H.T #rows in line with grid
    H = np.flip(H, axis=0) #flip because hist2d takes increasing bins only
    base_list = np.array([np.histogram2d(arr_list[0].ravel(order='F'), arr_list[1].ravel(order='F'), bins=[xbins, ybins])[0]
                              for arr_list in list(zip(baseline_lon, baseline_lat))])
    H_base = np.amax(base_list, axis=0)
    H_base = H_base.T
    H_base = np.flip(H_base, axis=0)
    
    return H, H_base

def compareBaseline(albedo, mesh_df, coverset, baseline, snr_th=100, bins=21):
    xvars = ['error', 'SNR', 'SolZA', 'SatZA']
    xlabels = [r'XCO$_2$ uncert', 'Signal-to-Noise Ratio', 'Solar Zenith Angle', 'Satellite Zenith Angle']
    binranges = [[0, 5], [0, 600], [0, 90], [0, 90]]
    error = calc_set_err(albedo, mesh_df, coverset)
    quality_passed = error[error['SNR']>snr_th]
    base_error = calc_set_err(albedo, mesh_df, baseline)
    base_quality_passed = base_error[base_error['SNR']>snr_th]
    #convert latlon coords to matrix indices within (-130, -50, -30, 50) = (minlon, minlat, maxlon, maxlat)
    quality_pts_arr = [((50-quality_passed['lat'])*2.).round().astype(int), ((quality_passed['lon']+130)*2.).round().astype(int)]
                #lat(row), lon(col)
    base_pts_arr = [((50-base_quality_passed['lat'])*2.).round().astype(int), ((base_quality_passed['lon']+130)*2.).round().astype(int)]
    H, H_base = histFooprints2d(coverset, baseline)
    total_err = np.repeat(quality_passed['error'], H[quality_pts_arr].astype(int))
    base_total_err = np.repeat(base_quality_passed['error'], H_base[base_pts_arr].astype(int))

    fig = plt.figure(figsize=(10,8),constrained_layout=True)
    gs = fig.add_gridspec(3,2)
    axs = [fig.add_subplot(gs[0,0]),
            fig.add_subplot(gs[0,1]),
            fig.add_subplot(gs[1,0]),
            fig.add_subplot(gs[1,1])]
    celltext=[[np.sum(H_base).astype(int), np.sum(H).astype(int)]]
    colLabs=['Count']
    for i, var in enumerate(xvars):
        thisax = axs[i]
        total_var = np.repeat(quality_passed[var], H[quality_pts_arr].astype(int))
        base_total_var = np.repeat(base_quality_passed[var], H_base[base_pts_arr].astype(int))
        l1 = thisax.hist(
                x=total_var,
                range=binranges[i],
                bins=bins,
                label='baseline',
                edgecolor='white')
        l2 = thisax.hist(
                x=base_total_var,
                range=binranges[i],
                bins=bins,
                label='algorithm',
                edgecolor='white',
                alpha=0.5)
        if i%2==0:
            thisax.set_ylabel('Frequency')
        else:
            thisax.set_ylabel(None)
        thisax.set_xlabel(xlabels[i])
        celltext.append([base_total_var.median().round(4), total_var.median().round(4)])
        colLabs.append(xlabels[i])
    fig.tight_layout()
    celltext = np.array(celltext).T
    # plt.subplots_adjust(bottom=0.2, top=0.9)
    # tax = plt.axes([.2, 0, .8, 0.15])
    tax = fig.add_subplot(gs[2:, :])
    tax.axis('off')
    tab = tax.table(
            celltext,
            rowColours=['tab:blue', 'tab:orange'],
            rowLabels=['Baseline', 'Algorithm'],
            colLabels=colLabs,
            bbox=[0,0,1,1])
    tab.auto_set_font_size(False)
    tab.set_fontsize(11)
    tax.text(-.073,.82,'Median', fontsize=11)
    return fig