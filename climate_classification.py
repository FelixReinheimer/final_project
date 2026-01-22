1# -*- coding: utf-8 -*-
"""
De Martonne Climate Classification with an interactive console menu.

Improvements implemented:
1) If the user selects "Historical", no period selection is requested (fixed 30-year set).
2) If the selected period is shorter than 30 years (or not enough years exist in the dataset),
   the program does not crash. It prints a message and asks again.
3) After creating a figure, the program returns to the main menu so the user can create more maps.
4) An "exit" option is provided to quit the program gracefully.

Added:
- Country boundaries overlay from countries.shp (GeoPandas).
- Correct handling of NaN values: NaNs remain NaN and are shown in the plot (grey).
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import geopandas as gpd

# Import climate data + countries shapefile (loaded in data_import.py)
from data_import import (
    pr_historical, pr_ssp126, pr_ssp585,
    tas_historical, tas_ssp126, tas_ssp585,
    countries
)


# ------------------------------------------------------------
# Temperature conversion
# ------------------------------------------------------------
def tas_to_celsius(tas: xr.DataArray) -> xr.DataArray:
    """
    Convert temperature from Kelvin to Celsius if necessary.
    """
    units = (tas.attrs.get("units") or "").lower()
    if units in ["k", "kelvin"]:
        tas_c = tas - 273.15
        tas_c.attrs = tas.attrs.copy()
        tas_c.attrs["units"] = "°C"
        return tas_c
    return tas


# ------------------------------------------------------------
# De Martonne Index
# ------------------------------------------------------------
def de_martonne_index(pr: xr.DataArray, tas_c: xr.DataArray) -> xr.DataArray:
    """
    Compute the De Martonne aridity index:
        I = P / (T + 10)
    """
    I = pr / (tas_c + 10.0)
    I.name = "I_DM"
    I.attrs["long_name"] = "De Martonne aridity index"
    I.attrs["formula"] = "I = P / (T + 10)"
    I.attrs["P_units"] = "mm/year"
    I.attrs["T_units"] = "°C"
    return I


# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------
def classify_de_martonne(I: xr.DataArray) -> xr.Dataset:
    """
    Classify the De Martonne index into climate classes.

    Important:
    - We keep the class code as float so NaNs remain NaN.
      Casting to int would destroy NaNs and could incorrectly assign them to class 1.
    """
    code = xr.full_like(I, fill_value=np.nan, dtype=float)

    code = xr.where(I < 10, 1.0, code)
    code = xr.where((I >= 10) & (I < 20), 2.0, code)
    code = xr.where((I >= 20) & (I < 24), 3.0, code)
    code = xr.where((I >= 24) & (I < 28), 4.0, code)
    code = xr.where((I >= 28) & (I < 35), 5.0, code)
    code = xr.where((I >= 35) & (I < 55), 6.0, code)
    code = xr.where(I >= 55, 7.0, code)

    # Keep as float to preserve NaNs
    code = code.astype("float32")
    code.name = "climate_class_code"

    label = xr.full_like(code, "", dtype=object)
    label = xr.where(code == 1, "Arid", label)
    label = xr.where(code == 2, "Semi-arid", label)
    label = xr.where(code == 3, "Mediterranean", label)
    label = xr.where(code == 4, "Semi-humid", label)
    label = xr.where(code == 5, "Humid", label)
    label = xr.where(code == 6, "Very Humid", label)
    label = xr.where(code == 7, "Extremely Humid", label)

    # Explicit label for missing cells
    label = xr.where(xr.ufuncs.isnan(code), "No data", label)

    label = label.astype(str)
    label.name = "climate_class_label"

    idm = xr.Dataset(
        {
            "I_DM": I,
            "climate_class_code": code,
            "climate_class_label": label,
        }
    )

    idm.attrs["class_codes"] = (
        "1=Arid; 2=Semi-arid; 3=Mediterranean; 4=Semi-humid; "
        "5=Humid; 6=Very Humid; 7=Extremely Humid"
    )
    idm.attrs["index_definition"] = "De Martonne aridity index: I = P / (T + 10)"
    idm.attrs["P_units"] = "mm/year"
    idm.attrs["T_units"] = "°C"

    return idm


# ------------------------------------------------------------
# Apply classification to a scenario
# ------------------------------------------------------------
def run_climate_classification(pr_ds: xr.Dataset, tas_ds: xr.Dataset) -> xr.Dataset:
    tas_c = tas_to_celsius(tas_ds["tas"])
    I = de_martonne_index(pr_ds["pr"], tas_c)
    return classify_de_martonne(I)


# ------------------------------------------------------------
# Period handling
# ------------------------------------------------------------
class Period:
    def __init__(self, name: str, start_year: int, end_year: int):
        self.name = name
        self.start_year = start_year
        self.end_year = end_year

    def length_years(self) -> int:
        return self.end_year - self.start_year + 1

    def validate_min_length(self, min_years: int = 30) -> bool:
        if self.end_year < self.start_year:
            return False
        return self.length_years() >= min_years


NEAR_FUTURE = Period("Near future (2031–2060)", 2031, 2060)
FAR_FUTURE = Period("Far future (2071–2100)", 2071, 2100)


# ------------------------------------------------------------
# Time conversion (works for string years like '2031')
# ------------------------------------------------------------
def time_to_years(time_coord: xr.DataArray) -> xr.DataArray:
    if np.issubdtype(time_coord.dtype, np.integer):
        return time_coord.astype(int)

    if np.issubdtype(time_coord.dtype, np.floating):
        return time_coord.round().astype(int)

    if time_coord.dtype.kind in {"U", "S"}:
        return xr.DataArray(
            time_coord.values.astype(int),
            dims=time_coord.dims,
            coords=time_coord.coords
        )

    try:
        return time_coord.dt.year.astype(int)
    except Exception as e:
        raise TypeError(
            f"Could not convert time coordinate to years (dtype={time_coord.dtype})."
        ) from e


def compute_period_mean_idm(idm: xr.Dataset, period: Period | None) -> xr.Dataset:
    """
    Compute mean I_DM over a selected period (or full range) and classify it.
    """
    if "time" not in idm.dims:
        raise ValueError("Dataset has no 'time' dimension.")

    years = time_to_years(idm["time"])

    if period is None:
        mask = xr.ones_like(years, dtype=bool)
        period_name = "Full available time range"
        period_years = f"{int(years.min())}-{int(years.max())}"
    else:
        mask = (years >= period.start_year) & (years <= period.end_year)
        period_name = period.name
        period_years = f"{period.start_year}-{period.end_year}"

    n_years = int(mask.sum().values)
    if n_years < 30:
        raise ValueError(f"Selected range contains only {n_years} years (minimum is 30).")

    I_mean = idm["I_DM"].where(mask).mean(dim="time", skipna=True)
    I_mean.name = "I_DM"

    out = classify_de_martonne(I_mean)
    out.attrs["period_name"] = period_name
    out.attrs["period_years"] = period_years
    return out


# ------------------------------------------------------------
# Plotting (discrete legend + country boundaries + NaN visualization)
# ------------------------------------------------------------
CLASS_LABELS = {
    1: "Arid (<10)",
    2: "Semi-arid (10–<20)",
    3: "Mediterranean (20–<24)",
    4: "Semi-humid (24–<28)",
    5: "Humid (28–<35)",
    6: "Very Humid (35–<55)",
    7: "Extremely Humid (≥55)",
}

CLASS_COLORS = [
    "#d73027", "#fc8d59", "#fee08b",
    "#d9ef8b", "#91cf60", "#1a9850", "#2c7bb6"
]

CMAP = ListedColormap(CLASS_COLORS)
CMAP.set_bad(color="lightgrey", alpha=1.0)  # show NaNs as grey
NORM = BoundaryNorm(np.arange(0.5, 8.5, 1), CMAP.N)


def infer_lat_lon(da: xr.DataArray):
    lat = next(c for c in ["lat", "latitude", "y"] if c in da.coords)
    lon = next(c for c in ["lon", "longitude", "x"] if c in da.coords)
    return lat, lon


def plot_class_map(
    class_ds: xr.Dataset,
    title: str,
    subtitle: str,
    outfile: str,
    countries_gdf: gpd.GeoDataFrame
):
    """
    Plot a classification map with:
    - discrete legend
    - country boundaries overlay
    - NaNs displayed as grey (CMAP.set_bad)
    """
    code = class_ds["climate_class_code"]
    lat, lon = infer_lat_lon(code)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title(f"{title}\n{subtitle}", fontsize=13)

    ax.pcolormesh(
        code[lon], code[lat], code,
        cmap=CMAP, norm=NORM, shading="auto"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # Overlay: country boundaries (assumes lon/lat = EPSG:4326)
    if countries_gdf is not None and len(countries_gdf) > 0:
        gdf = countries_gdf
        if gdf.crs is not None and str(gdf.crs).lower() not in ["epsg:4326", "crs84"]:
            gdf = gdf.to_crs("EPSG:4326")
        gdf.boundary.plot(ax=ax, linewidth=0.5, color="black", alpha=0.8, zorder=3)

    legend = [
        Patch(facecolor=CLASS_COLORS[i - 1], edgecolor="black",
              label=f"{CLASS_LABELS[i]}")
        for i in range(1, 8)
    ]
    # Optional: include "No data" in the legend
    legend.append(Patch(facecolor="lightgrey", edgecolor="black", label="No data"))

    ax.legend(
        handles=legend,
        title="De Martonne classes",
        loc="lower left",
        fontsize=9,
        title_fontsize=10,
        frameon=True
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.show(fig)



# ------------------------------------------------------------
# Interactive input helpers (repeat until valid)
# ------------------------------------------------------------
def prompt_int(prompt: str) -> int:
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("Please enter a valid integer year.")


def ask_user_for_map_request_loop():
    """
    Main menu:
    - user selects scenario or exits
    - for future scenarios, user selects a period (near/far/custom/full) with validation loops
    - returns a tuple (scenario_key, scenario_label, period_or_none)
    """
    print("\n=== Map Generator Menu ===")
    print("Type 'exit' to quit at any prompt.\n")

    while True:
        print("Select scenario:")
        print("  1) Historical")
        print("  2) SSP1-2.6 (ssp126)")
        print("  3) SSP5-8.5 (ssp585)")
        print("  exit) Quit program")
        s = input("Choice: ").strip().lower()

        if s == "exit":
            return None

        scenario_map = {
            "1": ("historical", "Historical"),
            "2": ("ssp126", "SSP1-2.6"),
            "3": ("ssp585", "SSP5-8.5"),
        }
        if s not in scenario_map:
            print("Invalid scenario selection. Please try again.\n")
            continue

        scenario_key, scenario_label = scenario_map[s]

        # Historical: no period selection requested
        if scenario_key == "historical":
            print("\nHistorical selected: using the full available historical time range.\n")
            return scenario_key, scenario_label, None

        # Future scenarios: select a period
        while True:
            print("\nSelect period:")
            print("  1) Near future (2031–2060)")
            print("  2) Far future (2071–2100)")
            print("  3) Custom period (>= 30 years)")
            print("  4) Full available range (>= 30 years)")
            print("  back) Return to scenario selection")
            print("  exit) Quit program")
            p = input("Choice: ").strip().lower()

            if p == "exit":
                return None
            if p == "back":
                print("")
                break

            if p == "1":
                return scenario_key, scenario_label, NEAR_FUTURE
            if p == "2":
                return scenario_key, scenario_label, FAR_FUTURE
            if p == "4":
                return scenario_key, scenario_label, None

            if p == "3":
                start = input("Start year: ").strip().lower()
                if start == "exit":
                    return None
                end = input("End year: ").strip().lower()
                if end == "exit":
                    return None

                try:
                    start_year = int(start)
                    end_year = int(end)
                except ValueError:
                    print("Invalid year input. Please enter integer years.\n")
                    continue

                period = Period(f"Custom ({start_year}-{end_year})", start_year, end_year)
                if not period.validate_min_length(min_years=30):
                    years_len = period.length_years() if end_year >= start_year else 0
                    print(
                        f"Selected period is invalid or too short ({years_len} years). "
                        "Please choose at least 30 years.\n"
                    )
                    continue

                return scenario_key, scenario_label, period

            print("Invalid period selection. Please try again.\n")


# ------------------------------------------------------------
# Main: compute datasets once, then loop for repeated map creation
# ------------------------------------------------------------
if __name__ == "__main__":

    # Compute classification datasets for each scenario once
    idm_historical = run_climate_classification(pr_historical, tas_historical)
    idm_ssp126 = run_climate_classification(pr_ssp126, tas_ssp126)
    idm_ssp585 = run_climate_classification(pr_ssp585, tas_ssp585)

    # Map scenario keys to datasets
    scenario_to_idm = {
        "historical": idm_historical,
        "ssp126": idm_ssp126,
        "ssp585": idm_ssp585,
    }

    # Interactive loop: create multiple maps until user exits
    while True:
        request = ask_user_for_map_request_loop()
        if request is None:
            print("Exiting program.")
            break

        scenario_key, scenario_label, period = request
        idm_selected = scenario_to_idm[scenario_key]

        while True:
            try:
                idm_period = compute_period_mean_idm(idm_selected, period)
                break
            except ValueError as e:
                print(f"\nSelection not possible: {e}\n")
                print("Please choose a different period.\n")

                if scenario_key == "historical":
                    print("Historical has no alternative period selection. Returning to main menu.\n")
                    idm_period = None
                    break

                print("Select a new period:")
                print("  1) Near future (2031–2060)")
                print("  2) Far future (2071–2100)")
                print("  3) Custom period (>= 30 years)")
                print("  4) Full available range (>= 30 years)")
                print("  back) Return to main menu")
                print("  exit) Quit program")
                p = input("Choice: ").strip().lower()

                if p == "exit":
                    print("Exiting program.")
                    raise SystemExit
                if p == "back":
                    idm_period = None
                    break
                if p == "1":
                    period = NEAR_FUTURE
                elif p == "2":
                    period = FAR_FUTURE
                elif p == "4":
                    period = None
                elif p == "3":
                    start_year = prompt_int("Start year: ")
                    end_year = prompt_int("End year: ")
                    period = Period(f"Custom ({start_year}-{end_year})", start_year, end_year)
                    if not period.validate_min_length(min_years=30):
                        print("Custom period must be at least 30 years. Returning to main menu.\n")
                        idm_period = None
                        break
                else:
                    print("Invalid selection. Returning to main menu.\n")
                    idm_period = None
                    break

        if idm_period is None:
            continue

        outfile = f"idm_map_{scenario_key}_{idm_period.attrs.get('period_years', 'range')}.png"
        plot_class_map(
            idm_period,
            title="De Martonne Climate Classification Map",
            subtitle=f"{scenario_label} | {idm_period.attrs.get('period_name', '')} | {idm_period.attrs.get('period_years', '')}",
            outfile=outfile,
            countries_gdf=countries
        )

        print(f"Map saved to: {outfile}\n")
