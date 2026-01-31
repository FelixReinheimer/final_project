GLOBAL CLIMATE ZONE CLASSIFICATION FOR CURRENT AND FUTURE PERIODS
AND SIMULATED POPULATION RESETTLEMENT UNDER CLIMATE CHANGE

============================================================

OVERVIEW
--------
This project focuses on the global classification of climate zones for historical
and future climate conditions and the simulation of population redistribution
under climate change.

Using gridded precipitation and temperature data from historical records and
future climate scenarios (SSP1-2.6 and SSP5-8.5), climate zones are classified
based on the De Martonne aridity index. These classifications are then combined
with gridded population data to assess population exposure, climate class
transitions, and hypothetical resettlement patterns driven by increasing
aridity.

The historical climate classification is based on data from 1981 to 2010.
Future climate scenarios (SSP1-2.6 and SSP5-8.5) are available for the period
2031 to 2100. Population analyses and resettlement simulations use population
data from the year 2005 as a fixed reference to isolate the effects of climate
change.


PIPELINE SUMMARY
----------------
1. data_import.py  
   Loads and prepares climate, population, and geographic input data.

2. climate_classification.py  
   Computes the De Martonne aridity index, classifies climate zones, and
   generates maps via a command-line driven workflow.

3. population.py  
   Aggregates population by climate class and simulates climate-driven
   resettlement patterns. 
   
4. extra_task.py  
   Computes country-level fractions of climate classes for multiple scenarios
   and time periods via a command-line driven workflow.


SCIENTIFIC CONTEXT
------------------
The project explores how climate change alters global climate zones and how
these changes may affect population distribution. The simulated resettlement
represents a possible projection under the assumption that populations relocate
from increasingly dry regions to the nearest areas with wetter climatic
conditions.


REQUIREMENTS
------------
Python ≥ 3.9

Required Python packages:
- geopandas
- matplotlib
- numpy
- pandas
- scipy
- shapely
- xarray


USAGE
-----
The Python scripts are run from the main project folder. This is the folder
that contains the Python files and the data directories.

To run an analysis, open a terminal, navigate to the project folder, and
execute the desired script, for example:

- climate_classification.py
- extra_task.py
- population.py


OUTPUTS
-------
- Climate classification maps (PNG)
- Population-by-climate-class tables (CSV)
- Population summary plots (PNG)
- Resettled population maps (PNG)


AUTHORS
-------
Bilal Billouch  
Felix Reinheimer  
Programming for Geographers with Python – 2025/2026
