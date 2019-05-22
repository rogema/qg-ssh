#!/usr/bin/env python
from os import path, system
from sys import argv, exit
import xarray as xr
import numpy as np
from numpy import array, ma, isnan
import glob
import qg
import pandas as pd
import cftime
from mpi4py import MPI

#==============================================================
# MPI initialization
#------------------------------------------------------------
comm = MPI.COMM_WORLD
nb_proc = comm.Get_size()
proc_indx = comm.Get_rank()

#==============================================================
# Config file
#------------------------------------------------------------
def read_config_file(config_file):
    from configparser import ConfigParser
    cfg = ConfigParser()
    cfg.read(config_file)
    return cfg

#==============================================================
# 'USAGE   : %s <START (YYYYMMDD)> <END (YYYYMMDD)> <PARAM_FILE.yaml>\n' %argv[0]

START = argv[1]
END = argv[2]
param_file = argv[3]
    
#============================================ Select dates in ACCESS-OM outpouts  ==============================================

dstart = pd.to_datetime(START, format='%Y%m%d')
dend = pd.to_datetime(END, format='%Y%m%d')
vectime =  pd.date_range(start=dstart, end=dend, freq='D')

files_ocean = sorted(glob.glob('/g/data/hh5/tmp/cosima/access-om2-01/01deg_jra55v13_iaf/output*/ocean/ocean_daily.nc'))
grid_access01 = xr.open_dataset('/g/data/hh5/tmp/cosima/access-om2-01/01deg_jra55v13_iaf/output001/ocean/ocean_grid.nc')

ds_daily = xr.open_mfdataset(files_ocean[-108:-78], concat_dim='time')
ds_daily = xr.merge([ds_daily, grid_access01.isel(time=0).drop('time')])
ds_daily = ds_daily.set_coords(['geolat_t', 'geolon_t', 'geolat_c', 'geolon_c'])

ssh_ACC = ds_daily.sel(time=slice(vectime[0],vectime[-1]), xt_ocean=slice(-250, -220), yt_ocean=slice(-60, -45)).eta_t

lon = ssh_ACC.geolon_t.data
lat = ssh_ACC.geolat_t.data

#=================================================================================================================================
#==============================================  Load parameters from param_file   ===============================================
cfg = read_config_file(param_file)

nbd_adv = int(cfg['GENERAL']['nbd_adv'])
filename_out = cfg['GENERAL']['filename_out']

rd =  float(cfg['PHYSICAL']['Rd'])

dtout = float(cfg['NUMERICAL']['dtout'])
dt = float(cfg['NUMERICAL']['dt'])
Snu1 = float(cfg['NUMERICAL']['snu1'])

# ---------------------------------------------------
Rd = xr.ones_like(ssh_ACC.geolon_t).data * rd
snu1 = xr.ones_like(ssh_ACC.geolon_t).data * Snu1


tint = dtout * nbd_adv
#=================================================================================================================================
list_ssh_adv = []
for tref in range(len(vectime)-1):
    if (tref % nb_proc) == proc_indx:
        #print('tref=', tref)
        #print('proc_indx=', proc_indx)
        Ht0 = ssh_ACC.isel(time=tref).squeeze().load().data
        ssh_adv = qg.run_qg(h1i=Ht0, Rd=Rd, lon=lon, lat=lat, tint=tint, dtout=dtout, dt=dt, snu1=snu1)
        dr_ssh_adv = xr.DataArray(ssh_adv[-1,:,:], 
                                  dims= ssh_ACC.isel(time=tref).dims,
                                  coords= ssh_ACC.isel(time=tref).coords)        
        list_ssh_adv.append(dr_ssh_adv)

ds_ssh = xr.concat(list_ssh_adv, dim='time').to_dataset(name='ssh_adv')

#=================================================================================================================================
ds_ssh['ssh_adv'].encoding = {'zlib': True, 'complevel': 5, 'shuffle': True}
#ds_ssh.to_netcdf(filename_out)
ds_ssh.to_netcdf(filename_out + '_core_%s' % proc_indx)
#print(filename_out + '_core_%s' % proc_indx)
#=================================================================================================================================







