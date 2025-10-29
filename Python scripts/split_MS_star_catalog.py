# imports
import numpy as np
from tqdm import tqdm
import pickle
import os

if __name__ == "__main__":
    # load relevant main sequence stars
    f = open("data/Gaia_relevant_stars_MS.pkl", "rb")
    all_source_ids = pickle.load(f)
    all_ras = pickle.load(f)
    all_decs = pickle.load(f)
    all_parallaxes = pickle.load(f)
    all_pmras = pickle.load(f)
    all_pmdecs = pickle.load(f)
    all_G_mags = pickle.load(f)
    all_BPs = pickle.load(f)
    all_RPs = pickle.load(f)
    all_ruwes = pickle.load(f)
    all_masses = pickle.load(f)
    f.close()
    
    # split catalogs, assuming 8*96 = 768 cores will be used
    num_operations = len(all_source_ids)
    num_nodes = 8*96
    base_operations = num_operations // num_nodes
    remainder = num_operations % num_nodes
    loop_inds = [0]
    for i in range(num_nodes):
        loop_inds.append(loop_inds[-1] + base_operations + (1 if i < remainder else 0))

    # print number of sources that need to fit by each core
    print('len(all_source_ids):', len(all_source_ids))
    
    # collect properties for the subsets and save them in separate files
    for run_num in tqdm(range(num_nodes)):
        source_ids, ras, decs, parallaxes, pmras, pmdecs = [], [], [], [], [], []
        G_mags, BPs, RPs, ruwes, masses = [], [], [], [], []
        for i in range(len(all_source_ids)):
            if loop_inds[run_num] <= i < loop_inds[run_num+1]:
                source_ids.append(all_source_ids[i])
                ras.append(all_ras[i])
                decs.append(all_decs[i])
                parallaxes.append(all_parallaxes[i])
                pmras.append(all_pmras[i])
                pmdecs.append(all_pmdecs[i])
                G_mags.append(all_G_mags[i])
                BPs.append(all_BPs[i])
                RPs.append(all_RPs[i])
                ruwes.append(all_ruwes[i])
                masses.append(all_masses[i])

        # convert to numpy arrays
        source_ids, ras, decs, parallaxes, pmras, pmdecs = np.array(source_ids), np.array(ras), np.array(decs), np.array(parallaxes), np.array(pmras), np.array(pmdecs)
        G_mags, BPs, RPs, ruwes, masses = np.array(G_mags), np.array(BPs), np.array(RPs), np.array(ruwes), np.array(masses)
        
        # save files (still too big for GitHub)
        with open("/scratch/gpfs/cl5968/gaia_data/Gaia_relevant_stars_MS_final_chunked_sep17/file" + str(run_num) + ".pkl", "wb") as f:
            pickle.dump(source_ids, f)
            pickle.dump(ras, f)
            pickle.dump(decs, f)
            pickle.dump(parallaxes, f)
            pickle.dump(pmras, f)
            pickle.dump(pmdecs, f)
            pickle.dump(G_mags, f)
            pickle.dump(BPs, f)
            pickle.dump(RPs, f)
            pickle.dump(ruwes, f)
            pickle.dump(masses, f)
            