# GeoCarb Mission Observation Scenario Tool (MOST)

![scangif](https://github.com/GeoCarb/GCMOST/images/sample_scan.gif)

![version](https://img.shields.io/badge/version-1.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.7-critical)
![license](https://img.shields.io/badge/license-MIT-yellow)

## Description
This tool was created to create an optimized scanning strategy for the GeoCarb, geostationary carbon cycle observatory, instrument.

## Dependencies
* numpy
* matplotlib
* descartes
* numba
* shapely
* pyproj
* cartopy
* pandas
* geopandas
* joblib
* netCDF4

## Installation
(If you have a package manager, i.e., `conda`, it is recommended to create a separate environment due to the number of dependencies required.) To install run,
`pip install git+https://github.com/GeoCarb/GCMOST#egg=gcmost`.

## Menu
After successful installation, alter the file `menus/main_menu_template.py` and save it to your working directory (we will call it `main_menu.py`).

## Run
Run the program by typing to the command line,
`gcmost-main main_menu.py`.

By default, outputs will be saved to a folder in the working directory called `./output/`.