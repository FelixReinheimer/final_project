# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 17:41:01 2026

@author: bbill
"""


import numpy as np
import geopandas as gpd
from shapely.geometry import Point

from data_import import (
    pr_historical, pr_ssp126, pr_ssp585,
    tas_historical, tas_ssp126, tas_ssp585,
    countries
)

from climate_classification import (
    run_climate_classification,
    compute_period_mean_idm,
    NEAR_FUTURE, FAR_FUTURE
)

LABEL = {
    1: "Arid",
    2: "Semi-arid",
    3: "Mediterranean",
    4: "Semi-humid",
    5: "Humid",
    6: "Very Humid",
    7: "Extremely Humid",
}

def latlon(da):
    lat = next(c for c in ["lat", "latitude", "y"] if c in da.coords)
    lon = next(c for c in ["lon", "longitude", "x"] if c in da.coords)
    return lat, lon

def fractions_in_country(code, country):
    hit = countries[countries["CNTRY_NAME"].str.lower() == country.lower()]
    if hit.empty:
        raise ValueError(f"Country '{country}' not found in shapefile.")

    geom = hit.dissolve().geometry.iloc[0]

    latn, lonn = latlon(code)
    lats, lons = code[latn].values, code[lonn].values
    LON, LAT = np.meshgrid(lons, lats)

    inside = gpd.GeoSeries(
        [Point(xy) for xy in zip(LON.ravel(), LAT.ravel())],
        crs="EPSG:4326"
    ).within(geom).values.reshape(code.shape)

    c = code.where(inside)
    total = np.isfinite(c).sum().item()

    return {
        k: (np.sum(c.values == float(k)) / total if total else np.nan)
        for k in range(1, 8)
    }

# ------------------------------------------------------------
# Computing climate classifications per country 
# ------------------------------------------------------------
if __name__ == "__main__":

    print("Computing climate classifications ...")

    idm_hist = run_climate_classification(pr_historical, tas_historical)
    idm_126  = run_climate_classification(pr_ssp126, tas_ssp126)
    idm_585  = run_climate_classification(pr_ssp585, tas_ssp585)

    maps = [
        ("Historical",
         compute_period_mean_idm(idm_hist, None)["climate_class_code"]),
        ("SSP1-2.6 Near future",
         compute_period_mean_idm(idm_126, NEAR_FUTURE)["climate_class_code"]),
        ("SSP1-2.6 Far future",
         compute_period_mean_idm(idm_126, FAR_FUTURE)["climate_class_code"]),
        ("SSP5-8.5 Near future",
         compute_period_mean_idm(idm_585, NEAR_FUTURE)["climate_class_code"]),
        ("SSP5-8.5 Far future",
         compute_period_mean_idm(idm_585, FAR_FUTURE)["climate_class_code"]),
    ]

    print("\nType a country name to compute fractions.")
    print("Type 'exit' to quit.\n")

    while True:
        country = input("Country: ").strip()
        if country.lower() == "exit":
            print("Exiting program.")
            break

        try:
            print(f"\nFraction of De Martonne climate classes within {country}:")
            for title, code in maps:
                f = fractions_in_country(code, country)
                print(f"\n{title}:")
                for k in range(1, 8):
                    print(f"  {LABEL[k]:15s}: {f[k]*100:6.2f} %")

        except ValueError as e:
            print(f"Error: {e}")
