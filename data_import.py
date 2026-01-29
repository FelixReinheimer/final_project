"""
Data loading utilities for climate impact analysis.

This module provides helper functions to load and prepare climate and
socioeconomic datasets used in the analysis.

"""

from pathlib import Path
import xarray as xr
import geopandas as gpd
import pandas as pd

def load_timeseries(base_path: str) -> dict[str, xr.Dataset]:
    """
    Load NetCDF files from historical, ssp126, ssp585 folders
    and concatenate them along the time dimension.
    
    """
    base = Path(base_path)
    scenarios = ["historical", "ssp126", "ssp585"]
    data = {}

    for scenario in scenarios:
        folder = base / scenario
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        files = sorted(folder.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No .nc files in {folder}")

        datasets = [xr.open_dataset(f) for f in files]
        data[scenario] = xr.concat(datasets, dim="time")

    return data

def load_population(
    pop_path: str = "./data/population/population_histsoc_0p5deg_annual_1861-2005.nc4"
) -> xr.Dataset:
    """
    Load historical population data and reconstruct a valid annual time axis.

    Non-standard CF time units prevent reliable automatic decoding, so the
    time coordinate is replaced with a yearly DateTimeIndex (1861–2005).
    
    """
    pop_file = Path(pop_path)
    if not pop_file.exists():
        raise FileNotFoundError(f"Population file not found: {pop_file}")
    
    population_fixedtime = xr.open_dataset(pop_file, decode_times=False)
    population_fixedtime["time"] = pd.date_range(start="1861-01-01", end="2005-01-01",freq="YS")
    
    return population_fixedtime
    
def load_countries_shp(
    shp_dir: str = "./Countries_Area", shp_name: str = "countries.shp",
) -> gpd.GeoDataFrame:
    """
    Load the countries shapefile and return it as a GeoDataFrame.

    """
    base_dir = Path(__file__).resolve().parent
    shp_path = (base_dir / shp_dir / shp_name).resolve()

    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    return gpd.read_file(shp_path)


# -------------------------------------------------------------
# Load datasets ON IMPORT 
# -------------------------------------------------------------

# Load precipitation (pr)
pr = load_timeseries("./data/pr")
pr_historical = pr["historical"]
pr_ssp126 = pr["ssp126"]
pr_ssp585 = pr["ssp585"]

# Load temperature (tas)
tas = load_timeseries("./data/tas")
tas_historical = tas["historical"]
tas_ssp126 = tas["ssp126"]
tas_ssp585 = tas["ssp585"]

# Load population
population = load_population()

# Load countries shapefile
countries = load_countries_shp("./Countries_Area", "countries.shp")

