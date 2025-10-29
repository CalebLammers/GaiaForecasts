# imports
import numpy as np
from tqdm import tqdm
import pickle
import os

if __name__ == "__main__":
    # lists for storing all quantities (note: requires a lot of memory to load!)
    all_single_masses, all_primary_masses, all_secondary_masses, all_binary_Porbs, all_binary_eccs = [], [], [], [], []
    all_primary_MGs, all_secondary_MGs, all_primary_MBPs, all_secondary_MBPs, all_primary_MRPs, all_secondary_MRPs = [], [], [], [], [], []
    all_dists, all_primary_ras, all_primary_decs, all_secondary_ras, all_secondary_decs = [], [], [], [], []
    
    # load groups of binaries
    for run_num in tqdm(range(192)):
        f = open("data/COSMIC_binaries/file" + str(run_num) + ".pkl", "rb")
        single_masses = pickle.load(f)
        primary_masses = pickle.load(f)
        secondary_masses = pickle.load(f)
        binary_Porbs = pickle.load(f)
        binary_eccs = pickle.load(f)
        primary_MGs = pickle.load(f)
        secondary_MGs = pickle.load(f)
        primary_MBPs = pickle.load(f)
        secondary_MBPs = pickle.load(f)
        primary_MRPs = pickle.load(f)
        secondary_MRPs = pickle.load(f)
        dists = pickle.load(f)
        primary_ras = pickle.load(f)
        primary_decs = pickle.load(f)
        secondary_ras = pickle.load(f)
        secondary_decs = pickle.load(f)
        f.close()
        
        # add to lists
        all_single_masses.extend(single_masses)
        all_primary_masses.extend(primary_masses)
        all_secondary_masses.extend(secondary_masses)
        all_binary_Porbs.extend(binary_Porbs)
        all_binary_eccs.extend(binary_eccs)
        all_primary_MGs.extend(primary_MGs)
        all_secondary_MGs.extend(secondary_MGs)
        all_primary_MBPs.extend(primary_MBPs)
        all_secondary_MBPs.extend(secondary_MBPs)
        all_primary_MRPs.extend(primary_MRPs)
        all_secondary_MRPs.extend(secondary_MRPs)
        all_dists.extend(dists)
        all_primary_ras.extend(primary_ras)
        all_primary_decs.extend(primary_decs)
        all_secondary_ras.extend(secondary_ras)
        all_secondary_decs.extend(secondary_decs)
     
    # convert to numpy arrays
    all_single_masses = np.array(all_single_masses)
    all_primary_masses = np.array(all_primary_masses)
    all_secondary_masses = np.array(all_secondary_masses)
    all_binary_Porbs = np.array(all_binary_Porbs)
    all_binary_eccs = np.array(all_binary_eccs)
    all_primary_MGs = np.array(all_primary_MGs)
    all_secondary_MGs = np.array(all_secondary_MGs)
    all_primary_MBPs = np.array(all_primary_MBPs)
    all_secondary_MBPs = np.array(all_secondary_MBPs)
    all_primary_MRPs = np.array(all_primary_MRPs)
    all_secondary_MRPs = np.array(all_secondary_MRPs)
    all_dists = np.array(all_dists)
    all_primary_ras = np.array(all_primary_ras)
    all_primary_decs = np.array(all_primary_decs)
    all_secondary_ras = np.array(all_secondary_ras)
    all_secondary_decs = np.array(all_secondary_decs)
    
    # consistency check: print list lengths
    print('len(all_single_masses):', len(all_single_masses))
    print('len(all_primary_masses):', len(all_primary_masses))
    print('len(all_secondary_masses):', len(all_secondary_masses))
    print('len(all_binary_Porbs):', len(all_binary_Porbs))
    print('len(all_binary_eccs):', len(all_binary_eccs))
    print('len(all_primary_MGs):', len(all_primary_MGs))
    print('len(all_secondary_MGs):', len(all_secondary_MGs))
    print('len(all_secondary_MBPs):', len(all_secondary_MBPs))
    print('len(all_secondary_MRPs):', len(all_secondary_MRPs))
    print('len(all_dists):', len(all_dists))
    print('len(all_primary_ras):', len(all_primary_ras))
    print('len(all_primary_decs):', len(all_primary_decs))
    print('len(all_secondary_ras):', len(all_secondary_ras))
    print('len(all_secondary_decs):', len(all_secondary_decs))
    print('num systems:', len(all_single_masses) + len(all_primary_masses))

    # get random subset of ~approximately correct number within 2 kpc (from El-Badry et al. 2024)
    rand_inds = np.random.choice(np.arange(len(all_primary_masses)), size=185000000, replace=False)
    subset_primary_masses = all_primary_masses[rand_inds]
    subset_secondary_masses = all_secondary_masses[rand_inds]
    subset_binary_Porbs = all_binary_Porbs[rand_inds]
    subset_binary_eccs = all_binary_eccs[rand_inds]
    subset_primary_MGs = all_primary_MGs[rand_inds]
    subset_secondary_MGs = all_secondary_MGs[rand_inds]
    subset_primary_MBPs = all_primary_MBPs[rand_inds]
    subset_secondary_MBPs = all_secondary_MBPs[rand_inds]
    subset_primary_MRPs = all_primary_MRPs[rand_inds]
    subset_secondary_MRPs = all_secondary_MRPs[rand_inds]
    subset_dists = all_dists[rand_inds]
    subset_primary_ras = all_primary_ras[rand_inds]
    subset_primary_decs = all_primary_decs[rand_inds]
    subset_secondary_ras = all_secondary_ras[rand_inds]
    subset_secondary_decs = all_secondary_decs[rand_inds]
    
    # remove sources with M < 0.1 M_sun, which don't have MIST isochrones, and stars that would have died
    mask_bad_primary_lums = ((np.isnan(subset_primary_MGs) | np.isnan(subset_primary_MBPs)) | np.isnan(subset_primary_MRPs))
    mask_bad_secondary_lums = ((np.isnan(subset_secondary_MGs) | np.isnan(subset_secondary_MBPs)) | np.isnan(subset_secondary_MRPs))
    lum_mask = np.invert((mask_bad_primary_lums | mask_bad_secondary_lums))
    subset_primary_masses = subset_primary_masses[lum_mask]
    subset_secondary_masses = subset_secondary_masses[lum_mask]
    subset_binary_Porbs = subset_binary_Porbs[lum_mask]
    subset_binary_eccs = subset_binary_eccs[lum_mask]
    subset_primary_MGs = subset_primary_MGs[lum_mask]
    subset_secondary_MGs = subset_secondary_MGs[lum_mask]
    subset_primary_MBPs = subset_primary_MBPs[lum_mask]
    subset_secondary_MBPs = subset_secondary_MBPs[lum_mask]
    subset_primary_MRPs = subset_primary_MRPs[lum_mask]
    subset_secondary_MRPs = subset_secondary_MRPs[lum_mask]
    subset_dists = subset_dists[lum_mask]
    subset_primary_ras = subset_primary_ras[lum_mask]
    subset_primary_decs = subset_primary_decs[lum_mask]
    subset_secondary_ras = subset_secondary_ras[lum_mask]
    subset_secondary_decs = subset_secondary_decs[lum_mask]
    
    # remove faint binaries
    primary_Gs = subset_primary_MGs + 5.0*np.log10(subset_dists) - 5.0
    secondary_Gs = subset_secondary_MGs + 5.0*np.log10(subset_dists) - 5.0
    G0 = 8.77267 # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
    primary_fluxes = 10**((G0 - primary_Gs)/2.5)
    secondary_fluxes = 10**((G0 - secondary_Gs)/2.5)
    combined_Gs = -2.5*np.log10(primary_fluxes + secondary_fluxes) + G0
    mask_Gs = (combined_Gs < 19.0)
    print('np.sum(mask_Gs):', np.sum(mask_Gs))

    # remove resolved binaries
    ra1, ra2 = (np.pi/180.0)*subset_primary_ras, (np.pi/180.0)*subset_secondary_ras
    dec1, dec2 = (np.pi/180.0)*subset_primary_decs, (np.pi/180.0)*subset_secondary_decs
    ddec = dec2 - dec1
    dra = ra2 - ra1
    hav = np.sin(ddec/2.0)**2 + np.cos(dec1)*np.cos(dec2)*np.sin(dra/2.0)**2
    angular_sep = 2.0 * np.arcsin(np.sqrt(np.clip(hav, 0, 1.0)))
    Delta_Gs = secondary_Gs - primary_Gs
    mask_unresolved = (Delta_Gs > (1/25)*(angular_sep*2.0626e8 - 200))
    print('np.sum(mask_unresolved):', np.sum(mask_unresolved))

    # joint mask
    joint_mask = (mask_Gs & mask_unresolved)
    print('np.sum(joint_mask):', np.sum(joint_mask))
    
    # save results
    with open("data/COSMIC_binary_props.pkl", "wb") as f:
        pickle.dump(subset_primary_masses[joint_mask], f)
        pickle.dump(subset_secondary_masses[joint_mask], f)
        pickle.dump(subset_binary_Porbs[joint_mask], f)
        pickle.dump(subset_binary_eccs[joint_mask], f)
        pickle.dump(subset_primary_MGs[joint_mask], f)
        pickle.dump(subset_secondary_MGs[joint_mask], f)
        pickle.dump(subset_primary_MBPs[joint_mask], f)
        pickle.dump(subset_secondary_MBPs[joint_mask], f)
        pickle.dump(subset_primary_MRPs[joint_mask], f)
        pickle.dump(subset_secondary_MRPs[joint_mask], f)
        pickle.dump(subset_dists[joint_mask], f)
        pickle.dump(subset_primary_ras[joint_mask], f)
        pickle.dump(subset_primary_decs[joint_mask], f)
        pickle.dump(subset_secondary_ras[joint_mask], f)
        pickle.dump(subset_secondary_decs[joint_mask], f)
