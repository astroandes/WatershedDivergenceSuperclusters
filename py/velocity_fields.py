import abacus_cosmos.Halos as ach
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from collections import Counter
import scipy.ndimage as scpimg
import h5py

def smooth_data(path, L_cell=10.0, vmax_cut=300.0, sigma_smooth=1.0, L_box=720.0):
    N_side = np.int(L_box/L_cell)
    print(L_box, L_cell, N_side)
    output_path = os.path.join(path, "fields")
    output_name = "AbacusCosmos_720box_planck_00_0_FoF"

    halo_data = ach.read_halos_FoF(path)
    print("Done reading data")
 
    vmax = halo_data['vcirc_max']
    ii = (vmax>vmax_cut)
    pos_cut = halo_data['pos'][ii]+L_box*0.5
    vel_cut = halo_data['vel'][ii]
    print("Done selecting data by vmax")

    N_side = np.int(L_box/L_cell)
    ii = np.int_(pos_cut[:,0]/L_cell)
    jj = np.int_(pos_cut[:,1]/L_cell)
    kk = np.int_(pos_cut[:,2]/L_cell)

    print(np.min(ii), np.max(ii))
    print(np.min(jj), np.max(jj))
    print(np.min(kk), np.max(kk))

    n_grid = np.zeros((N_side, N_side, N_side))

    vel_x_grid = np.zeros((N_side, N_side, N_side))
    vel_x = vel_cut[:,0]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_x))):
        vel_x_grid[i,j,k] += vel_x[t]
        n_grid[i,j,k] += 1

    vel_y_grid = np.zeros((N_side, N_side, N_side))
    vel_y = vel_cut[:,1]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_y))):
        vel_y_grid[i,j,k] += vel_y[t]

    vel_z_grid = np.zeros((N_side, N_side, N_side))
    vel_z = vel_cut[:,2]
    for i,j,k,t in zip(ii,jj,kk,range(len(vel_z))):
        vel_z_grid[i,j,k] += vel_z[t]
    
    zz = n_grid>0
    vel_x_grid[zz] = vel_x_grid[zz]/n_grid[zz]
    vel_y_grid[zz] = vel_y_grid[zz]/n_grid[zz]
    vel_z_grid[zz] = vel_z_grid[zz]/n_grid[zz]

    print("Done NGP interpolation")
    
    vel_x_grid_smooth = scpimg.filters.gaussian_filter(vel_x_grid,sigma_smooth)
    vel_y_grid_smooth = scpimg.filters.gaussian_filter(vel_y_grid,sigma_smooth)
    vel_z_grid_smooth = scpimg.filters.gaussian_filter(vel_z_grid,sigma_smooth)
    print("Done Gaussian Smoothing")
    
    vel_x_grid_smooth_dx = scpimg.filters.correlate1d(vel_x_grid_smooth, [-1,0,1], axis=0, mode='wrap') * (1.0/(2.0*L_cell))
    vel_y_grid_smooth_dy = scpimg.filters.correlate1d(vel_y_grid_smooth, [-1,0,1], axis=1, mode='wrap') * (1.0/(2.0*L_cell))
    vel_z_grid_smooth_dz = scpimg.filters.correlate1d(vel_z_grid_smooth, [-1,0,1], axis=2, mode='wrap') * (1.0/(2.0*L_cell))
    divergence = vel_x_grid_smooth_dx + vel_y_grid_smooth_dy + vel_z_grid_smooth_dz
    print("Finished Divergence")

    output_path = os.path.join(path, "fields")
    output_filename = os.path.join(output_path, "mock_{}_vmax_{}_s_{:.1f}_nside_{}.hdf5".format(output_name, vmax_cut, sigma_smooth*L_cell, N_side))
    
    h5f = h5py.File(output_filename, 'w')
#    h5f.create_dataset('vel_x', data=vel_x_grid_smooth)
#    h5f.create_dataset('vel_y', data=vel_y_grid_smooth)
#    h5f.create_dataset('vel_z', data=vel_z_grid_smooth)
    h5f.create_dataset('divergence', data=divergence)
    h5f.close()
    print("Finished writing to {}".format(output_filename))

full_computation = True
if full_computation:
    path = "/Users/forero/data/AbacusCosmos/AbacusCosmos_720box_planck_00_00_FoF_halos_z0.100/"
    L_cell = 4.0
    for sigma_smooth in [1.0, 2.0, 5.0, 10.0, 15.0]:
        for vmax_cut in [150.0]:
            smooth_data(path, L_cell=L_cell, vmax_cut=vmax_cut, sigma_smooth=sigma_smooth)
