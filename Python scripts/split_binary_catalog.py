# imports
import numpy as np
from tqdm import tqdm
import pickle
import os

if __name__ == "__main__":
    # load unresolved binaries
    f = open("data/COSMIC_binary_props.pkl", "rb")
    all_primary_masses = pickle.load(f)
    all_secondary_masses = pickle.load(f)
    all_binary_Porbs = pickle.load(f)
    all_binary_eccs = pickle.load(f)
    all_primary_MGs = pickle.load(f)
    all_secondary_MGs = pickle.load(f)
    all_primary_MBPs = pickle.load(f)
    all_secondary_MBPs = pickle.load(f)
    all_primary_MRPs = pickle.load(f)
    all_secondary_MRPs = pickle.load(f)
    all_dists = pickle.load(f)
    all_primary_ras = pickle.load(f)
    all_primary_decs = pickle.load(f)
    all_secondary_ras = pickle.load(f)
    all_secondary_decs = pickle.load(f)
    f.close()
    
    # split catalogs, assuming 8*96 = 768 cores will be used
    num_operations = len(all_primary_masses)
    num_nodes = 5*96
    base_operations = num_operations // num_nodes
    remainder = num_operations % num_nodes
    loop_inds = [0]
    for i in range(num_nodes):
        loop_inds.append(loop_inds[-1] + base_operations + (1 if i < remainder else 0))
    
    # print number of sources that need to fit by each core
    print('len(all_primary_masses):', len(all_primary_masses))
    
    # collect properties for the subsets and save them in separate files
    for run_num in tqdm(range(480)):
        primary_masses, secondary_masses, binary_Porbs, binary_eccs, primary_MGs, secondary_MGs, primary_MBPs, secondary_MBPs, primary_MRPs, secondary_MRPs = [], [], [], [], [], [], [], [], [], []
        dists, ras, decs = [], [], []
        for i in range(len(all_primary_masses)):
            if loop_inds[run_num] <= i < loop_inds[run_num+1]:
                primary_masses.append(all_primary_masses[i])
                secondary_masses.append(all_secondary_masses[i])
                binary_Porbs.append(all_binary_Porbs[i])
                binary_eccs.append(all_binary_eccs[i])
                primary_MGs.append(all_primary_MGs[i])
                secondary_MGs.append(all_secondary_MGs[i])
                primary_MBPs.append(all_primary_MBPs[i])
                secondary_MBPs.append(all_secondary_MBPs[i])
                primary_MRPs.append(all_primary_MRPs[i])
                secondary_MRPs.append(all_secondary_MRPs[i])
                dists.append(all_dists[i])
                ras.append(all_primary_ras[i])
                decs.append(all_primary_decs[i])

        # convert to numpy arrays
        primary_masses, secondary_masses, binary_Porbs, binary_eccs, primary_MGs, secondary_MGs, primary_MBPs, secondary_MBPs, primary_MRPs, secondary_MRPs = np.array(primary_masses), np.array(secondary_masses), np.array(binary_Porbs), np.array(binary_eccs), np.array(primary_MGs), np.array(secondary_MGs), np.array(primary_MBPs), np.array(secondary_MBPs), np.array(primary_MRPs), np.array(secondary_MRPs)
        dists, ras, decs = np.array(dists), np.array(ras), np.array(decs)

        # save files (still too big for GitHub)
        with open("data/COSMIC_binary_props_chunked/file" + str(run_num) + ".pkl", "wb") as f:
            pickle.dump(primary_masses, f)
            pickle.dump(secondary_masses, f)
            pickle.dump(binary_Porbs, f)
            pickle.dump(binary_eccs, f)
            pickle.dump(primary_MGs, f)
            pickle.dump(secondary_MGs, f)
            pickle.dump(primary_MBPs, f)
            pickle.dump(secondary_MBPs, f)
            pickle.dump(primary_MRPs, f)
            pickle.dump(secondary_MRPs, f)
            pickle.dump(dists, f)
            pickle.dump(ras, f)
            pickle.dump(decs, f)
