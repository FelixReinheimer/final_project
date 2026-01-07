# -*- coding: utf-8 -*-
"""
Created on Wed Jan  7 16:55:34 2026

@author: felix
"""

from pathlib import Path
import xarray as xr




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
    Load population NetCDF file using cftime for correct time decoding.
    """
    pop_file = Path(pop_path)
    if not pop_file.exists():
        raise FileNotFoundError(f"Population file not found: {pop_file}")

    return xr.open_dataset(
        pop_file,
        use_cftime=True
    )


# ------------------------------------------------------------
# Automatic execution
# ------------------------------------------------------------
if __name__ == "__main__":

    # Load precipitation (pr)
    pr = load_timeseries("./data/pr")
    pr_historical = pr["historical"]
    pr_ssp126 = pr["ssp126"]
    pr_ssp585 = pr["ssp585"]

    print(
        "PR historical:",
        pr_historical.time.min().values,
        pr_historical.time.max().values
    )

    # Load temperature (tas)
    tas = load_timeseries("./data/tas")
    tas_historical = tas["historical"]
    tas_ssp126 = tas["ssp126"]
    tas_ssp585 = tas["ssp585"]

    print(
        "TAS historical:",
        tas_historical.time.min().values,
        tas_historical.time.max().values
    )

    # Load population (with cftime)
    population = load_population()

    print("POPULATION time axis:")
    print(population.time.values[:5], "...", population.time.values[-5:])
