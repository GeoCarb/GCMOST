from setuptools import setup

setup(
    # Needed to silence warnings
    name='gcmost',
    url='https://github.com/GeoCarb/MOST',
    author='Jeff Nivitanont',
    author_email='jeffniv@yahoo.com',
    # Needed to actually package something
    packages=['gcmost'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3',
    # Needed for dependencies
    install_requires=['numpy', 'matplotlib', 'descartes',
    'numba', 'shapely', 'pyproj', 'cartopy', 'pandas',
    'geopandas', 'joblib', 'netCDF4',],
    # *strongly* suggested for sharing
    version='1.0',
    license='MIT',
    description='GeoCarb Mission Observation Scenario Tool(MOST)',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
    # if there are any scripts
    entry_points = {'console_scripts': ['gcmost-main=gcmost.command_line:main']},
    include_package_data=True,
    package_data={'': ['data/*', 'menu/*']},
)