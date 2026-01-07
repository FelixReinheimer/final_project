# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:55:34 2026

@author: felix
"""

from pathlib import Path
import xarray as xr
import geopandas as gpd


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


# def load_population(
#     pop_path: str = "./data/population/population_histsoc_0p5deg_annual_1861-2005.nc4"
# ) -> xr.Dataset:
#     """
#     Load population NetCDF file using cftime for time decoding.
#     (This may raise a ValueError for 'years since ...' time units.)
#     """
#     pop_file = Path(pop_path)
#     if not pop_file.exists():
#         raise FileNotFoundError(f"Population file not found: {pop_file}")
#
#     return xr.open_dataset(pop_file, use_cftime=True)


def load_countries_shp(
    shp_dir: str = "./Countries_Area",
    shp_name: str = "countries.shp",
) -> gpd.GeoDataFrame:
    """
    Load the countries shapefile using GeoPandas.

    Parameters
    ----------
    shp_dir : str
        Directory that contains the shapefile (and its sidecar files).
    shp_name : str
        Main shapefile name.

    Returns
    -------
    gpd.GeoDataFrame
        Countries geometry and attributes.
    """
    base_dir = Path(__file__).resolve().parent
    shp_path = (base_dir / shp_dir / shp_name).resolve()

    if not shp_path.exists():
        raise FileNotFoundError(f"Shapefile not found: {shp_path}")

    return gpd.read_file(shp_path)


# ------------------------------------------------------------
# Load datasets ON IMPORT (so they can be imported elsewhere)
# ------------------------------------------------------------

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

# Load population (as originally attempted)
# population = load_population()

# Load countries shapefile
countries = load_countries_shp("./Countries_Area", "countries.shp")


# ------------------------------------------------------------
# Debug prints if run directly
# ------------------------------------------------------------
if __name__ == "__main__":

    print(
        "PR historical:",
        pr_historical.time.min().values,
        pr_historical.time.max().values
    )

    print(
        "TAS historical:",
        tas_historical.time.min().values,
        tas_historical.time.max().values
    )

    # print(
    #     "POPULATION time axis:",
    #     population.time.values[:5],
    #     "...",
    #     population.time.values[-5:]
    # )

    print("Countries shapefile loaded.")
    print("Number of features:", len(countries))
    print("CRS:", countries.crs)
    print("Columns:", list(countries.columns))
