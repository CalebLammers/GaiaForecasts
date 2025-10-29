# imports
import numpy as np
from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import multidim
import gaiamock # note: will only work if executed in the same directory as 'gaiamock.py'
import pickle
import os
import time
from isochrones import get_ichrone
tracks = get_ichrone('mist', tracks=True)

# read in the compiled C functions
c_funcs = gaiamock.read_in_C_functions()

if __name__ == "__main__":
    # generate binaries in parallelized bunches of 1e6
    run_num = int(os.environ['SLURM_PROCID'])
    sf_start_myr = 12_000.0 # 12 Gyr ago
    sf_duration_myr = 12_000.0 # uniform over 12 Gyr
    num_binaries = int(1e6)
    
    start = time.time()
    ibt, _, _, _, _ = InitialBinaryTable.sampler("multidim", final_kstar1=np.linspace(0, 14, 15), final_kstar2=np.linspace(0, 14, 15), rand_seed=run_num, nproc=1, SF_start=sf_start_myr, SF_duration=sf_duration_myr, met=0.02, size=num_binaries)
    end = time.time()
    print('Time elapsed generating binaries:', end - start)
    
    # parameters to record
    kstar2 = np.array(ibt['kstar_2'])
    mass1s = np.array(ibt['mass_1'])
    mass2s = np.array(ibt['mass_2'])
    Porbs = np.array(ibt['porb'])
    eccs = np.array(ibt['ecc'])
    ages = np.array(ibt['tphysf']) # Myr ("time to evolve the binary")
    single_mask = (kstar2 == 15.0)
    binary_mask = (kstar2 != 15.0)
    single_masses = mass1s[single_mask]
    primary_masses = mass1s[binary_mask]
    secondary_masses = mass2s[binary_mask]
    binary_Porbs = Porbs[binary_mask]
    binary_eccs = eccs[binary_mask]
    binary_ages = ages[binary_mask]

    # assign luminosities to primaries and secondaries using MIST isochrones
    start = time.time()
    primary_MGs, primary_MBPs, primary_MRPs = np.zeros(len(primary_masses)), np.zeros(len(primary_masses)), np.zeros(len(primary_masses))
    secondary_MGs, secondary_MBPs, secondary_MRPs = np.zeros(len(primary_masses)), np.zeros(len(primary_masses)), np.zeros(len(primary_masses))
    for i in range(len(primary_masses)):
        rand_met = np.random.normal(loc=0.0, scale=0.2)
        iso_dict_primary = tracks.generate(mass=primary_masses[i], age=np.log10(1e6*binary_ages[i]), feh=rand_met, return_dict=True)
        iso_dict_secondary = tracks.generate(mass=secondary_masses[i], age=np.log10(1e6*binary_ages[i]), feh=rand_met, return_dict=True)
        primary_MGs[i] = iso_dict_primary['G_mag']
        primary_MBPs[i] = iso_dict_primary['BP_mag']
        primary_MRPs[i] = iso_dict_primary['RP_mag']
        secondary_MGs[i] = iso_dict_secondary['G_mag']
        secondary_MBPs[i] = iso_dict_secondary['BP_mag']
        secondary_MRPs[i] = iso_dict_secondary['RP_mag']
    end = time.time()
    print('Time elapsed assigning luminosities:', end-start)

    # generate positions for the primary stars
    start = time.time()
    primary_ras, primary_decs, dists, primary_xs, primary_ys, primary_zs = gaiamock.generate_coordinates_at_a_given_distance_exponential_disk(0.0, 2000.0, N_stars=len(primary_masses))
    end = time.time()
    print('Time elapsed generating positions:', end-start)

    # get sky positions for the secondary stars
    semi_as = ((binary_Porbs/365.25)**(2/3))*((primary_masses + secondary_masses)**(1/3)) # AU
    incs = np.arccos(np.random.uniform(-1.0, 1.0, size=len(primary_masses))) # 0 to pi
    omegas = np.random.uniform(0.0, 2*np.pi, size=len(primary_masses)) # rads
    Omegas = np.random.uniform(0.0, 2*np.pi, size=len(primary_masses)) # rads
    mean_anomalies = np.random.uniform(0.0, 2*np.pi, size=len(primary_masses)) # rads

    # solve Kepler's equation using Gaiamock's solver
    eccentric_anomalies = np.zeros(len(primary_masses)) # rads
    for i in range(len(primary_masses)):
        eccentric_anomalies[i] = gaiamock.solve_kepler_eqn_on_array(np.array([mean_anomalies[i]]), binary_eccs[i], c_funcs)[0]
    
    # calculate true anomalies and orbital radii
    true_anomalies = 2*np.arctan2(np.sqrt(1 + binary_eccs)*np.sin(eccentric_anomalies/2), np.sqrt(1 - binary_eccs)*np.cos(eccentric_anomalies/2)) # rads
    orbital_rads = (1/206265)*semi_as*(1 - binary_eccs*np.cos(eccentric_anomalies)) # pc

    # position of secondaries with respect to focus of orbit
    xs = orbital_rads*(np.cos(Omegas)*np.cos(omegas + true_anomalies) - np.sin(Omegas)*np.sin(omegas + true_anomalies)*np.cos(incs))
    ys = orbital_rads*(np.sin(Omegas)*np.cos(omegas + true_anomalies) + np.cos(Omegas)*np.sin(omegas + true_anomalies)*np.cos(incs))
    zs = orbital_rads*np.sin(omegas + true_anomalies)*np.sin(incs)

    # get RAs and Decs for companions
    secondary_ras, secondary_decs = gaiamock.xyz_to_radec(primary_xs + xs, primary_ys + ys, primary_zs + zs)

    # save results
    with open("data/COSMIC_binaries/file" + str(run_num) + ".pkl", "wb") as f:
        pickle.dump(single_masses, f)
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
        pickle.dump(primary_ras, f)
        pickle.dump(primary_decs, f)
        pickle.dump(secondary_ras, f)
        pickle.dump(secondary_decs, f)        
