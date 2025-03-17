#!/home/b/b382728/miniconda3/envs/tobac_flow/bin/python
#SBATCH --job-name=synsat_tracking_2021
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=512G
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --account=bb1376
#SBATCH --output=tobac.%j.out

import warnings
import pathlib
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr

import intake
import healpy

from tobac_flow.analysis import get_label_stats, weighted_statistics_on_labels
from tobac_flow.linking import process_file
from tobac_flow.utils.datetime_utils import get_dates_from_filename
from tobac_flow.utils.xarray_utils import add_compression_encoding, add_dataarray_to_ds
from tobac_flow.flow import create_flow

from tobac_flow.dataset import (
    add_label_coords,
    add_step_labels,
    add_label_coords,
    flag_edge_labels,
    flag_nan_adjacent_labels,
    calculate_label_properties,
    link_cores_and_anvils,
    link_step_labels
)
from tobac_flow.analysis import (
    get_label_stats,
    weighted_statistics_on_labels,
)
from tobac_flow.utils import (
    add_dataarray_to_ds,
    create_dataarray
)
from tobac_flow.utils.label_utils import labeled_comprehension
from tobac_flow.postprocess import (
    add_validity_flags,
    process_core_properties,
    process_thick_anvil_properties,
    process_thin_anvil_properties,
)
from tobac_flow.utils import (
    remove_orphan_coords,
    filter_cores,
    filter_anvils,
)
from tobac_flow.detection import (
    detect_cores,
    get_anvil_markers,
    detect_anvils,
    relabel_anvils,
)
synsat_path = pathlib.Path("/work/bb1376/user/fabian/data/synsat/ngc4008a-zoom9/maxzen")

synsat_files = sorted(list(synsat_path.glob("synsat_ngc4008a-zoom9_maxzen_2021*.nc")))

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

t_inds = np.unique(bt.t, return_index=True)[1]

bt = bt.isel(t=t_inds)
wvd = wvd.isel(t=t_inds)
swd = swd.isel(t=t_inds)

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

print("Final thick anvil markers: area =", np.sum(anvil_markers != 0).item(), flush=True)
print("Final thick anvil markers: n =", anvil_markers.max().item(), flush=True)
thick_anvil_labels = detect_anvils(
    flow,
    wvd - np.maximum(swd,0),
    markers=anvil_markers,
    upper_threshold=upper_threshold,
    lower_threshold=lower_threshold,
    erode_distance=erode_distance,
    min_length=min_length,
)
print("Initial detected thick anvils: area =", np.sum(thick_anvil_labels != 0).item(), flush=True)
print("Initial detected thick anvils: n =", thick_anvil_labels.max().item(), flush=True)
thick_anvil_labels = relabel_anvils(
    flow,
    thick_anvil_labels,
    markers=anvil_markers,
    overlap=overlap,
    absolute_overlap=absolute_overlap,
    min_length=min_length,
)

print("Final detected thick anvils: area =", np.sum(thick_anvil_labels != 0).item(), flush=True)
print("Final detected thick anvils: n =", thick_anvil_labels.max().item(), flush=True)
thin_anvil_labels = detect_anvils(
    flow,
    wvd + np.maximum(swd,0),
    markers=thick_anvil_labels,
    upper_threshold=upper_threshold + 5,
    lower_threshold=lower_threshold + 5,
    erode_distance=erode_distance,
    min_length=min_length,
)

print("Detected thin anvils: area =", np.sum(thin_anvil_labels != 0).item(), flush=True)
print("Detected thin anvils: n =", np.max(thin_anvil_labels).item(), flush=True)


# Process output 
dataset = xr.Dataset()
dataset["core_label"] = core_labels
dataset["thick_anvil_label"] = thick_anvil_labels
dataset["thin_anvil_label"] = thin_anvil_labels
dataset["bt"] = bt

if "longitude" in dataset:
    dataset = dataset.rename(longitude="lon")
if "latitude" in dataset:
    dataset = dataset.rename(latitude="lat")

# Postprocessing

dataset = add_label_coords(dataset)

link_cores_and_anvils(dataset)

add_step_labels(dataset)

dataset = add_label_coords(dataset)

link_step_labels(dataset)

flag_edge_labels(dataset, max_time_gap=1500)

if len(dataset.lat.shape)==1:
    dataset["area"] = xr.DataArray(
        np.tile((6_378 * np.radians(0.1))**2 * np.cos(np.radians(dataset.lat)), [dataset.lon.size, 1]).T, 
        coords={"lat":dataset.lat, "lon":dataset.lon}, 
        dims=("lat", "lon")
    )
elif len(dataset.lat.shape)==2:
    dataset["area"] = xr.DataArray(
        (6_378 * np.radians(0.1))**2 * np.cos(np.radians(dataset.lat)), 
        coords={"y":dataset.y, "x":dataset.x}, 
        dims=("y", "x")
    )

calculate_label_properties(dataset)

dataset = remove_orphan_coords(dataset)
print(datetime.now(), "Removing orphaned items", flush=True)

# Remove invalid cores and process core properties
print(datetime.now(), "Filtering and processing cores", flush=True)
dataset = filter_cores(dataset, verbose=True)
dataset = process_core_properties(dataset)

print(datetime.now(), "Filtering and processing anvils", flush=True)
dataset = filter_anvils(dataset, verbose=True)
dataset = process_thick_anvil_properties(dataset)
dataset = process_thin_anvil_properties(dataset)

print(datetime.now(), "Flagging core and anvil quality", flush=True)
dataset = remove_orphan_coords(dataset)
dataset = add_validity_flags(dataset)

dataset.core_label.data = np.where(np.isin(dataset.core_label.data, dataset.core), dataset.core_label.data, 0)
dataset.core_step_label.data = np.where(np.isin(dataset.core_step_label.data, dataset.core_step), dataset.core_step_label.data, 0)

dataset.thick_anvil_label.data = np.where(np.isin(dataset.thick_anvil_label.data, dataset.anvil), dataset.thick_anvil_label.data, 0)
dataset.thick_anvil_step_label.data = np.where(np.isin(dataset.thick_anvil_step_label.data, dataset.thick_anvil_step), dataset.thick_anvil_step_label.data, 0)

dataset.thin_anvil_label.data = np.where(np.isin(dataset.thin_anvil_label.data, dataset.anvil), dataset.thin_anvil_label.data, 0)
dataset.thin_anvil_step_label.data = np.where(np.isin(dataset.thin_anvil_step_label.data, dataset.thin_anvil_step), dataset.thin_anvil_step_label.data, 0)

print(f"Final core count: {dataset.core.size}")
print(f"Final valid core count: {dataset.core_is_valid.data.sum()}")
print(f"Final anvil count: {dataset.anvil.size}")
print(f"Final valid thick anvil count: {dataset.thick_anvil_is_valid.data.sum()}")
print(f"Final valid thin anvil count: {dataset.thin_anvil_is_valid.data.sum()}")

dataset.drop_vars(["bt"])

comp = dict(zlib=True, complevel=5, shuffle=True)
for var in dataset.data_vars:
    dataset[var].encoding.update(comp)

dataset.to_netcdf("./synsat_tracking_zoom9_2021.nc")