#!/home/b/b382728/miniconda3/envs/tobac_flow/bin/python
import warnings
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import intake
import healpy

synsat_path = pathlib.Path("/work/bb1376/user/fabian/data/synsat/ngc4008a-zoom9/maxzen")

synsat_files = sorted(list(synsat_path.glob("synsat_ngc4008a-zoom9_maxzen_2033*.nc")))

mask = xr.open_dataset("/work/bb1376/user/fabian/data/synsat/ngc4008a-zoom9/aux/ngc4008a-zoom9_maxzen80_mask_for_embedding.nc")

def regrid_synsat(input_filename, grid_spacing=0.1, limits=[-45,45,-30,30]):
    lon = xr.DataArray(
        np.arange(limits[0]+grid_spacing/2, limits[1], grid_spacing)%360, 
        dims=("lon",), name="lon", attrs=dict(units="degrees", standard_name="longitude")
    )
    lat = xr.DataArray(
        np.arange(limits[2]+grid_spacing/2, limits[3], grid_spacing), 
        dims=("lat",), name="lat", attrs=dict(units="degrees", standard_name="latitude")
    )
    pix = xr.DataArray(
        healpy.ang2pix(mask.crs.healpix_nside, *np.meshgrid(lon, lat), nest=True, lonlat=True),
        coords=(lat, lon),
    )

    with xr.open_dataset(input_filename) as dataset:
        bt = xr.DataArray(
            np.full((1,)+mask.zen_mask.shape, np.nan, dtype=dataset.bt108.data.dtype), 
            coords = dict(crs=mask.crs, time=dataset.time, cell=mask.cell), 
            dims = ("time", "cell"), 
            attrs = dataset.bt108.attrs
        )
        bt[0][mask.zen_mask.data==1] = dataset.bt108.data[0]
        bt = bt.isel(cell=pix)

        wvd = xr.DataArray(
            np.full((1,)+mask.zen_mask.shape, np.nan, dtype=dataset.bt062.data.dtype), 
            coords = dict(crs=mask.crs, time=dataset.time, cell=mask.cell), 
            dims = ("time", "cell"), 
            attrs = dataset.bt062.attrs
        )
        wvd[0][mask.zen_mask.data==1] = dataset.bt062.data[0] - dataset.bt073.data[0]
        wvd = wvd.isel(cell=pix)

        swd = xr.DataArray(
            np.full((1,)+mask.zen_mask.shape, np.nan, dtype=dataset.bt087.data.dtype), 
            coords = dict(crs=mask.crs, time=dataset.time, cell=mask.cell), 
            dims = ("time", "cell"), 
            attrs = dataset.bt087.attrs
        )
        swd[0][mask.zen_mask.data==1] = dataset.bt087.data[0] - dataset.bt120.data[0]
        swd = swd.isel(cell=pix)

    return bt, wvd, swd

regrid_stack = [regrid_synsat(f, grid_spacing=0.1, limits=[-75,75,-75,75]) for f in synsat_files]

bt, wvd, swd = [xr.concat(z, "time").rename(time="t") for z in zip(*regrid_stack)]

from tobac_flow.flow import create_flow
flow = create_flow(
    bt, model="Farneback", vr_steps=1, smoothing_passes=1, interp_method="linear"
)
wvd_threshold = 0.25
bt_threshold = 0.25
overlap = 0.5
absolute_overlap = 1
subsegment_shrink = 0.0
min_length = 2
from tobac_flow.detection import (
    detect_cores,
    get_anvil_markers,
    detect_anvils,
    relabel_anvils,
)
core_labels = detect_cores(
    flow,
    bt,
    wvd,
    swd,
    wvd_threshold=wvd_threshold,
    bt_threshold=bt_threshold,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    subsegment_shrink=subsegment_shrink,
    min_length=min_length,
    use_wvd=False,
)
upper_threshold = -5
lower_threshold = -10
erode_distance = 2

anvil_markers = get_anvil_markers(
    flow,
    wvd - np.maximum(swd,0),
    threshold=upper_threshold,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    subsegment_shrink=subsegment_shrink,
    min_length=min_length,
)

print("Final thick anvil markers: area =", np.sum(anvil_markers != 0), flush=True)
print("Final thick anvil markers: n =", anvil_markers.max(), flush=True)
thick_anvil_labels = detect_anvils(
    flow,
    wvd - np.maximum(swd,0),
    markers=anvil_markers,
    upper_threshold=upper_threshold,
    lower_threshold=lower_threshold,
    erode_distance=erode_distance,
    min_length=min_length,
)
print("Initial detected thick anvils: area =", np.sum(thick_anvil_labels != 0), flush=True)
print("Initial detected thick anvils: n =", thick_anvil_labels.max(), flush=True)
thick_anvil_labels = relabel_anvils(
    flow,
    thick_anvil_labels,
    markers=anvil_markers,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    min_length=min_length,
)

print("Final detected thick anvils: area =", np.sum(thick_anvil_labels != 0), flush=True)
print("Final detected thick anvils: n =", thick_anvil_labels.max(), flush=True)
thin_anvil_labels = detect_anvils(
    flow,
    wvd + np.maximum(swd,0),
    markers=thick_anvil_labels,
    upper_threshold=upper_threshold + 5,
    lower_threshold=lower_threshold + 5,
    erode_distance=erode_distance,
    min_length=min_length,
)

print("Detected thin anvils: area =", np.sum(thin_anvil_labels != 0), flush=True)
print("Detected thin anvils: n =", np.max(thin_anvil_labels), flush=True)


# Process output 
ds = xr.Dataset()
ds["core_labels"] = bt.copy(data=core_labels)
ds["thick_anvil_labels"] = bt.copy(data=thick_anvil_labels)
ds["thin_anvil_labels"] = bt.copy(data=thin_anvil_labels)

comp = dict(zlib=True, complevel=5, shuffle=True)
for var in ds.data_vars:
    ds[var].encoding.update(comp)

ds.to_netcdf("/scratch/b/b382728/synsat/synsat_tracking_zoom9_2033.nc")