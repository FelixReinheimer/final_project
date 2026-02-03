"""
This script combines climate classification results (De Martonne classes) with gridded population data to
(1) summarize the 2005 population distribution across climate classes for multiple scenarios/periods and
(2) generate maps of baseline population and a simple resettlement experiment where population is moved
from locations projected to become drier to nearby locations with the same or wetter future climate class.
Outputs include a CSV summary table and several PNG maps.

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree
from matplotlib.colors import ListedColormap, BoundaryNorm

from data_import import (
    pr_historical, pr_ssp126, pr_ssp585,
    tas_historical, tas_ssp126, tas_ssp585,
    population
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

# ------------------------------------------------------------
# Population: convert time coordinate from YYYY-01-01 to YYYY
# ------------------------------------------------------------
pop = population["number_of_people"]
pop = pop.assign_coords(time=pop.time.dt.year)
pop2005 = pop.sel(time=2005)

# ------------------------------------------------------------
# Climate classifications
# ------------------------------------------------------------
idm_hist = run_climate_classification(pr_historical, tas_historical)
idm_126  = run_climate_classification(pr_ssp126, tas_ssp126)
idm_585  = run_climate_classification(pr_ssp585, tas_ssp585)

# ppsat = population per scenario and time
ppsat = [
    ("Historical", "Full range", idm_hist, None),
    ("SSP1-2.6", NEAR_FUTURE.name, idm_126, NEAR_FUTURE),
    ("SSP1-2.6", FAR_FUTURE.name,  idm_126, FAR_FUTURE),
    ("SSP5-8.5", NEAR_FUTURE.name, idm_585, NEAR_FUTURE),
    ("SSP5-8.5", FAR_FUTURE.name,  idm_585, FAR_FUTURE),
]

# ------------------------------------------------------------
# Population aggregation per climate class
# ------------------------------------------------------------
rows = []

for scen, pername, idm, per in ppsat:
    code = compute_period_mean_idm(idm, per)["climate_class_code"] # climate class per grid cell
    pop_aligned = pop2005.interp_like(code, method="nearest")      # align population grid to climate grid

    total = float(pop_aligned.where(~xr.ufuncs.isnan(code)).sum(skipna=True)) # total population on land cells with valid climate class
    for k in range(1, 8):
        pk = float(pop_aligned.where(code == k).sum(skipna=True)) # population in climate class k
        rows.append([
            scen, pername, k, LABEL[k],
            pk, (pk / total * 100 if total > 0 else np.nan)
        ]) # share of total population (%)

df = pd.DataFrame(
    rows,
    columns=["scenario", "period", "class", "label", "population_2005", "share_percent"]
)

# ------------------------------------------------------------
# Outputs: table + plot
# ------------------------------------------------------------
df.to_csv(
    "population_by_climate_class_2005.csv",
    index=False,
    encoding="utf-8-sig"
)

pivot = (
    df.pivot_table(
        index="label",
        columns=["scenario", "period"],
        values="population_2005",
        aggfunc="sum"
    )
    .reindex([LABEL[i] for i in range(1, 8)])
)

desired_cols = [
    ("Historical", "Full range"),
    ("SSP1-2.6", NEAR_FUTURE.name),
    ("SSP1-2.6", FAR_FUTURE.name),
    ("SSP5-8.5", NEAR_FUTURE.name),
    ("SSP5-8.5", FAR_FUTURE.name),
]

pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(desired_cols))

pivot.columns = [f"{scen} – {per}" for scen, per in pivot.columns]

ax = pivot.plot(kind="bar", figsize=(12, 6))
ax.set_xlabel("De Martonne climate class")
ax.set_ylabel("Population (2005)")
ax.set_title("Population (2005) by climate class (population held constant)")
plt.tight_layout()
plt.savefig("population_by_climate_class_2005.png", dpi=200)
plt.show()

print(df)
print("\nSaved:")
print(" - population_by_climate_class_2005.csv")
print(" - population_by_climate_class_2005.png")


# ------------------------------------------------------------
# Resettlement maps
# ------------------------------------------------------------
def _latlon(da):
    lat = next(c for c in ["lat", "latitude", "y"] if c in da.coords)
    lon = next(c for c in ["lon", "longitude", "x"] if c in da.coords)
    return lat, lon


# ------------------------------------------------------------
# Plotting population with absolute values using classes
# ------------------------------------------------------------
POP_CLASS_LABELS = [
    "0",
    r"$>0$–$<10^{1}$",
    r"$10^{1}$–$<10^{2}$",
    r"$10^{2}$–$<10^{3}$",
    r"$10^{3}$–$<10^{4}$",
    r"$10^{4}$–$<10^{5}$",
    r"$10^{5}$–$<10^{6}$",
    r"$10^{6}$–$<10^{7}$",
    r"$\geq 10^{7}$",
]

N_POP_CLASSES = 9


def classify_population(pop2d: xr.DataArray) -> xr.DataArray:
    """
    Classify absolute population values per grid cell into discrete
    magnitude-based classes for visualization.

    Class codes (people per grid cell):
      1: 0   ≤ p < 1
      2: 1   ≤ p < 10
      3: 10  ≤ p < 100
      4: 100 ≤ p < 1,000
      5: 1,000 ≤ p < 10,000
      6: 10,000 ≤ p < 100,000
      7: 100,000 ≤ p < 1,000,000
      8: 1,000,000 ≤ p < 10,000,000
      9: p ≥ 10,000,000

    NaN values remain NaN (e.g. ocean or no-data cells).
    
    """
    pop_class = xr.full_like(pop2d, np.nan, dtype=np.float32)

    pop_class = xr.where((pop2d >= 0) & (pop2d < 1), 1, pop_class)
    pop_class = xr.where((pop2d >= 1) & (pop2d < 10), 2, pop_class)
    pop_class = xr.where((pop2d >= 10) & (pop2d < 100), 3, pop_class)
    pop_class = xr.where((pop2d >= 100) & (pop2d < 1000), 4, pop_class)
    pop_class = xr.where((pop2d >= 1000) & (pop2d < 10000), 5, pop_class)
    pop_class = xr.where((pop2d >= 10000) & (pop2d < 100000), 6, pop_class)
    pop_class = xr.where((pop2d >= 100000) & (pop2d < 1000000), 7, pop_class)
    pop_class = xr.where((pop2d >= 1000000) & (pop2d < 10000000), 8, pop_class)
    pop_class = xr.where(pop2d >= 10000000, 9, pop_class)

    pop_class.name = "population_class"
    return pop_class

def plot_pop_map(pop2d, title, outfile):
    """
    Plot population as discrete classes based on absolute values.
    
    """
    # Keep NaNs (ocean) as NaN
    pop2d = pop2d.where(np.isfinite(pop2d))

    lat, lon = _latlon(pop2d)

    pop_class = classify_population(pop2d)
    
    # Discrete colormap for 9 classes
    base = plt.cm.viridis(np.linspace(0, 1, N_POP_CLASSES))
    cmap = ListedColormap(base)
    cmap.set_bad(color="lightgrey")  # NaNs = ocean

    norm = BoundaryNorm(np.arange(0.5, N_POP_CLASSES + 1.5, 1.0), cmap.N) # discrete color mapping for population classes
    

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title(title)

    m = ax.pcolormesh(
        pop_class[lon], pop_class[lat], pop_class,
        shading="auto",
        cmap=cmap,
        norm=norm
    )

    cbar = plt.colorbar(m, ax=ax, ticks=np.arange(1, N_POP_CLASSES + 1))
    cbar.set_ticklabels(POP_CLASS_LABELS)
    cbar.set_label("People per grid cell (classes from absolute values)")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()

def resettle(pop_src, hist_code, fut_code):
    """
    Redistribute population from grid cells that become drier in the future to the
    nearest grid cell with the same or a wetter future climate class, while keeping
    ocean/no-data cells as NaN.
    
    """
    
    # Align population and historical climate data to the future climate grid
    pop_src = pop_src.interp_like(fut_code, method="nearest") 
    hist_code = hist_code.interp_like(fut_code, method="nearest") 

    # Prepare latitude/longitude grids for spatial distance calculations
    latn, lonn = _latlon(fut_code)
    lat = fut_code[latn].values
    lon = fut_code[lonn].values
    LON, LAT = np.meshgrid(lon, lat)

    # Extract NumPy arrays and set appropriate dtypes for numerical processing
    h = hist_code.values.astype(np.float32)
    f = fut_code.values.astype(np.float32)
    p = pop_src.values.astype(np.float64)

    # Identify valid cells, determine which population moves or stays
    valid = np.isfinite(h) & np.isfinite(f) & np.isfinite(p)

    move = valid & (f < h) & (p > 0)
    stay = valid & (~move)

    after = np.full_like(p, np.nan)
    after[valid] = 0.0
    after[stay] = p[stay]

    # KDTree per future climate class (only on valid cells)
    trees, pos_cache = {}, {}
    for c in range(1, 8):
        m = valid & (f == c)
        if np.any(m):
            pos_cache[c] = np.argwhere(m)
            trees[c] = cKDTree(np.column_stack([LAT[m], LON[m]]))

    for i, j in np.argwhere(move):
        k = int(h[i, j])  # historical climate class at source
        src = np.array([[LAT[i, j], LON[i, j]]])

        placed = False
        for tc in range(k, 8):  # prefer same class, then wetter
            if tc not in trees:
                continue
            _, idx = trees[tc].query(src, k=1)
            di, dj = pos_cache[tc][int(idx[0])]
            after[di, dj] += p[i, j]
            placed = True
            break

        if not placed:
            after[i, j] += p[i, j]

    return xr.DataArray(after, coords=fut_code.coords, dims=fut_code.dims, name="pop_resettled")


# --------------------------------------------------------------------
# Baseline population map under historical climate (no resettlement)
# --------------------------------------------------------------------
hist_code = compute_period_mean_idm(idm_hist, None)["climate_class_code"]
pop_hist = pop2005.interp_like(hist_code, method="nearest")
pop_hist = pop_hist.where(np.isfinite(hist_code))

plot_pop_map(
    pop_hist,
    "Population 2005 (Historical baseline, no resettlement)",
    "pop_2005_historical_baseline.png"
)

# ------------------------------------------------------------------
# Population resettlement maps relative to the historical baseline
# ------------------------------------------------------------------
future_cases = [
    ("SSP1-2.6", NEAR_FUTURE.name, idm_126, NEAR_FUTURE),
    ("SSP1-2.6", FAR_FUTURE.name,  idm_126, FAR_FUTURE),
    ("SSP5-8.5", NEAR_FUTURE.name, idm_585, NEAR_FUTURE),
    ("SSP5-8.5", FAR_FUTURE.name,  idm_585, FAR_FUTURE),
]

for scen, pername, idm, per in future_cases:
    fut_code = compute_period_mean_idm(idm, per)["climate_class_code"]

    pop_src = pop2005.interp_like(fut_code, method="nearest")
    pop_res = resettle(pop_src, hist_code, fut_code)

    safe_scen = scen.replace(".", "").replace(" ", "_")
    safe_per  = pername.replace("–", "-").replace("(", "").replace(")", "").replace(" ", "_")

    plot_pop_map(
        pop_res,
        f"Population 2005 after resettlement | {scen} | {pername}",
        f"pop_2005_resettled_{safe_scen}_{safe_per}.png"
    )

print("\nSaved maps:")
print(" - pop_2005_historical_baseline.png")
print(" - pop_2005_resettled_*.png")