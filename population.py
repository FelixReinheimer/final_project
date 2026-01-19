# -*- coding: utf-8 -*-
<<<<<<< HEAD
"""
Created on Mon Jan 19 14:21:38 2026

@author: bbill
"""

=======

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from scipy.spatial import cKDTree
#from matplotlib.colors import LogNorm

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
# Climate classifications (computed once)
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

# scen = scenario / pername = period name / idm = de Martonne climate class / per = timeperiod
for scen, pername, idm, per in ppsat:
    code = compute_period_mean_idm(idm, per)["climate_class_code"]
    pop_aligned = pop2005.interp_like(code, method="nearest")

    total = float(pop_aligned.where(~xr.ufuncs.isnan(code)).sum(skipna=True))
    for k in range(1, 8):
        pk = float(pop_aligned.where(code == k).sum(skipna=True))
        rows.append([
            scen, pername, k, LABEL[k],
            pk, (pk / total * 100 if total > 0 else np.nan)
        ])

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

# Remove the brackets in the legend
pivot.columns = [
    f"{scen} – {per}" for scen, per in pivot.columns
]

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

def plot_pop_map(pop2d, title, outfile):
    # keep NaNs (ocean) as NaN
    pop2d = pop2d.where(np.isfinite(pop2d))

    lat, lon = _latlon(pop2d)

    # log10(x + 1): makes zeros defined (0 -> 0), preserves NaNs
    pop_plot = np.log10(pop2d + 1.0)

    cmap = plt.cm.viridis.copy() #colour scheme
    cmap.set_bad(color="lightgrey")  # NaNs = ocean

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.set_title(title)

    m = ax.pcolormesh(
        pop_plot[lon], pop_plot[lat], pop_plot,
        shading="auto",
        cmap=cmap
    )

    plt.colorbar(
        m, ax=ax,
        label="People per grid cell (log10)"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.show()


def resettle(pop_src, hist_code, fut_code):
    """
    Move only if fut < hist (drier than historical). Oceans/NoData stay NaN.
    IMPORTANT FIX: moved-from land cells become 0 (not NaN).
    """
    pop_src = pop_src.interp_like(fut_code, method="nearest")
    hist_code = hist_code.interp_like(fut_code, method="nearest")

    latn, lonn = _latlon(fut_code)
    lat = fut_code[latn].values
    lon = fut_code[lonn].values
    LON, LAT = np.meshgrid(lon, lat)

    h = hist_code.values.astype(np.float32)
    f = fut_code.values.astype(np.float32)
    p = pop_src.values.astype(np.float64)

    # valid = cells where ALL needed inputs exist (land); oceans will be invalid -> remain NaN
    valid = np.isfinite(h) & np.isfinite(f) & np.isfinite(p)

    move = valid & (f < h) & (p > 0)
    stay = valid & (~move)  # includes p==0 cells

    # Start with NaN everywhere (keeps ocean as NaN)
    after = np.full_like(p, np.nan)

    # Set ALL valid land cells to 0 first (so moved-from cells become 0, not NaN)
    after[valid] = 0.0

    # People who stay: copy their population
    after[stay] = p[stay]

    # KDTree per future class (only on valid cells)
    trees, pos_cache = {}, {}
    for c in range(1, 8):
        m = valid & (f == c)
        if np.any(m):
            pos_cache[c] = np.argwhere(m)
            trees[c] = cKDTree(np.column_stack([LAT[m], LON[m]]))

    # Move people from drying cells
    for i, j in np.argwhere(move):
        k = int(h[i, j])  # historical class at source
        src = np.array([[LAT[i, j], LON[i, j]]])

        placed = False
        for tc in range(k, 8):  # prefer hist class, then wetter
            if tc not in trees:
                continue
            _, idx = trees[tc].query(src, k=1)
            di, dj = pos_cache[tc][int(idx[0])]
            after[di, dj] += p[i, j]
            placed = True
            break

        # fallback: if no destination found, keep them in place
        if not placed:
            after[i, j] += p[i, j]

    return xr.DataArray(after, coords=fut_code.coords, dims=fut_code.dims, name="pop_resettled")


# ---- baseline historical climate map (no resettlement) ----
hist_code = compute_period_mean_idm(idm_hist, None)["climate_class_code"]
pop_hist = pop2005.interp_like(hist_code, method="nearest")
pop_hist = pop_hist.where(np.isfinite(hist_code))
plot_pop_map(pop_hist, "Population 2005 (Historical baseline, no resettlement)", "pop_2005_historical_baseline.png")

# ---- future resettlement maps (relative to historical baseline) ----
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
>>>>>>> Felix
