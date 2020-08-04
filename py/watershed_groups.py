import abacus_cosmos.Halos as ach
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import glob
from collections import Counter
import scipy.ndimage as scpimg
import h5py

import glob


def find_watershed(divergence_filename):
    print("processing: {}".format(divergence_filename))
    f = h5py.File(divergence_filename, 'r')
    divergence_field = f['divergence'][...]
    f.close()
    
    sort_index = np.argsort(divergence_field.flatten(order='C'))
    watershed_group = np.ones(np.shape(divergence_field), dtype=int)*-1
    N_side = np.shape(divergence_field)[0]
    print("N_side={}".format(N_side))
    n_total = len(sort_index)

    n_group = 0
    n_cell = 0
    for l in range(len(sort_index)):
        n_cell += 1
        ijk = np.unravel_index(sort_index[l], np.shape(divergence_field), order='C')
        groups = []
        for i in [-1,0, 1]:
            for j in [-1,0, 1]:
                for k in [-1,0, 1]:
                    ijk_test = tuple((np.array(ijk)+(i, j, k))%N_side)
                    groups.append(watershed_group[ijk_test])
        c = Counter(groups)
    
        if -1 in c.keys():
            c.pop(-1)
        if len(c)==0:
            watershed_group[ijk] = n_group
            n_group += 1
        else:
            g = c.most_common(1)
            watershed_group[ijk] = g[0][0]
        
        if((n_cell%100000)==0):
            print("{:.2f} % of total. {} groups".format(100*n_cell/n_total,n_group))
    watershed_group.reshape((N_side, N_side, N_side))
    output_filename = divergence_filename.replace("velocity_", "watershed_")
    h5f = h5py.File(output_filename, 'w')
    h5f.create_dataset('watershed_group', data=watershed_group)
    h5f.close()
    print("Finished writing to {}".format(output_filename))
    return 
    
compute_all = True

if compute_all:
    div_files = glob.glob("/Users/forero/data/AbacusCosmos/AbacusCosmos_720box_planck_00_00_FoF_halos_z0.100/fields/velocity_*.hdf5")
    print(div_files)
    for div_file in div_files:
        find_watershed(div_file)
