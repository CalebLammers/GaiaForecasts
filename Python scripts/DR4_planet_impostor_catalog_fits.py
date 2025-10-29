# imports
import numpy as np
from tqdm import tqdm
import gaiamock # note: will only work if executed in the same directory as 'gaiamock.py'
import pickle
import os
import time
import emcee
from isochrones import get_ichrone
from scipy import optimize
import numba.np.unsafe.ndarray # this line is needed to avoid a crash when fitting isochrones
mist = get_ichrone('mist')

# read in the compiled C functions
c_funcs = gaiamock.read_in_C_functions()

# simple model for Gaia's astrometric uncertainty
def piecewise(x, a):
    return a*np.max([1.0*np.ones(len(x)), 10**(0.2*(x - 14.0))], axis=0)

# MCMC function: fit orbit and calculate log probability (if parameters are physical)
def log_probability(theta, t_ast_yr, psi, plx_factor, ast_obs, ast_err):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    # convert dist to parallax
    theta_copy = np.copy(theta)
    theta_copy[2] = 1000.0/theta[2]

    resids_norm = gaiamock.get_astrometric_residuals_12par(t_ast_yr=t_ast_yr, psi=psi, plx_factor=plx_factor, ast_obs=ast_obs, ast_err=ast_err, theta_array=theta_copy, c_funcs=c_funcs)
    log_prob = lp + -0.5*np.sum(resids_norm**2)

    if np.isnan(log_prob):
        return -np.inf

    return log_prob

# MCMC function: uniform prior on physically plausible values
def log_prior(theta):
    ra_off, dec_off, dist, pmra, pmdec, period, ecc, phi_p, A, B, F, G = theta

    if not (0.0 < period):
        return -np.inf

    if not (0.0 < dist):
        return -np.inf

    if not (0.0 < ecc < 1.0):
        return -np.inf

    if not (0.0 < phi_p < 2*np.pi):
        return -np.inf

    return 0.0

# MCMC function: add a small amount of noise to the best-fit parameters (makes walkers independent)
def get_rand_p0(best_fit, nwalkers):
    p0_arr = best_fit + np.zeros((nwalkers, len(best_fit))) # many copies of best_fit
    p0_arr[:,0] = p0_arr[:,0] + 1e-3*np.random.randn(nwalkers) # ra_off
    p0_arr[:,1] = p0_arr[:,1] + 1e-3*np.random.randn(nwalkers) # dec_off
    p0_arr[:,2] = p0_arr[:,2] + 1e-3*np.random.randn(nwalkers) # dist
    p0_arr[:,3] = p0_arr[:,3] + 1e-3*np.random.randn(nwalkers) # pmra
    p0_arr[:,4] = p0_arr[:,4] + 1e-3*np.random.randn(nwalkers) # pmdec
    p0_arr[:,5] = p0_arr[:,5] + 1e-3*np.random.randn(nwalkers) # period
    p0_arr[:,6] = p0_arr[:,6] + 1e-3*np.random.randn(nwalkers) # ecc
    p0_arr[:,7] = p0_arr[:,7] + 1e-3*np.random.randn(nwalkers) # phi_p
    p0_arr[:,8] = p0_arr[:,8] + 1e-3*np.random.randn(nwalkers) # A
    p0_arr[:,9] = p0_arr[:,9] + 1e-3*np.random.randn(nwalkers) # B
    p0_arr[:,10] = p0_arr[:,10] + 1e-3*np.random.randn(nwalkers) # F
    p0_arr[:,11] = p0_arr[:,11] + 1e-3*np.random.randn(nwalkers) # G

    return p0_arr

# function to get planet mass by solving the relevant cubic (see Appendix A of paper)
def get_planet_mass(alpha, period, stellar_mass):
    # common factor
    C = (alpha**3)/(period**2)

    # cubic coefficients: x^3 + a x^2 + b x + c = 0
    a = -C
    b = -2*C*stellar_mass
    c = -C*(stellar_mass**2)

    # depressed‐cubic parameters
    p = b - (a*a)/3.0
    q = (2.0*a*a*a)/27.0 - (a*b)/3.0 + c

    # discriminant
    D = (q*0.5)**2 + (p/3.0)**3

    # Cardano’s formula for the single real root
    sqrt_D = np.sqrt(D)
    u = np.cbrt(-q*0.5 + sqrt_D)
    v = np.cbrt(-q*0.5 - sqrt_D)
    y = u + v

    # substitute back in
    x = y - a/3.0

    return x

# function that predicts G, BP, and RP mags given mass, age, feh, and distance to a star (assumes Av=0.0)
def predict_mags(mass, age, feh, dist, accurate=True):
    eep = mist.get_eep(mass, np.log10(1e9*age), feh, accurate=accurate)
    mags = mist.interp_mag([eep, np.log10(1e9*age), feh, dist, 0.0], bands=['G', 'BP', 'RP'])
    return mags[3]

# log_likelihood function for isochrone fitting
def log_likelihood(theta, true_G, true_BP, true_RP, dist):
    try:
        mass, age, feh = theta
        pred_G, pred_BP, pred_RP = predict_mags(mass, age, feh, dist, accurate=True)
    except Exception as e: # catch instances where the mass, age, or feh result in a crash
        return -np.inf
    
    log_like = -0.5*((pred_G - true_G)**2) - 0.5*((pred_BP - true_BP)**2) - 0.5*((pred_RP - true_RP)**2)

    if np.isnan(log_like):
        return -np.inf
    
    return log_like

if __name__ == "__main__":
    run_num = int(os.environ['SLURM_PROCID'])
    
    # load subset of unresolved binaries
    # these files are too big for GitHub, but can be re-generated using the notebooks on GitHub
    f = open("data/COSMIC_binary_props_chunked/file" + str(run_num) + ".pkl", "rb")
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
    ras = pickle.load(f)
    decs = pickle.load(f)
    f.close()

    # assign stellar quantities
    parallaxes = 1000/dists # mas
    simga_v = 30 # km/s # random velocities
    sigma_mu = 1e3*simga_v/(4.74*dists) # mas/yr
    pmras = np.random.normal(0.0, sigma_mu)
    pmdecs = np.random.normal(0.0, sigma_mu)
    primary_Gs = primary_MGs + 5.0*np.log10(dists) - 5.0
    secondary_Gs = secondary_MGs + 5.0*np.log10(dists) - 5.0
    primary_BPs = primary_MBPs + 5.0*np.log10(dists) - 5.0
    secondary_BPs = secondary_MBPs + 5.0*np.log10(dists) - 5.0
    primary_RPs = primary_MRPs + 5.0*np.log10(dists) - 5.0
    secondary_RPs = secondary_MRPs + 5.0*np.log10(dists) - 5.0
    
    # calculate combined magnitudes for the binaries (using Gaia's reported zero-point mags)
    G0, BP0, RP0 = 8.77267, 8.87618, 8.51848 # from https://coolwiki.ipac.caltech.edu/index.php/Central_wavelengths_and_zero_points
    primary_G_fluxes = 10**((G0 - primary_Gs)/2.5)
    secondary_G_fluxes = 10**((G0 - secondary_Gs)/2.5)
    primary_BP_fluxes = 10**((BP0 - primary_BPs)/2.5)
    secondary_BP_fluxes = 10**((BP0 - secondary_BPs)/2.5)
    primary_RP_fluxes = 10**((RP0 - primary_RPs)/2.5)
    secondary_RP_fluxes = 10**((RP0 - secondary_RPs)/2.5)
    combined_Gs = -2.5*np.log10(primary_G_fluxes + secondary_G_fluxes) + G0
    combined_BPs = -2.5*np.log10(primary_BP_fluxes + secondary_BP_fluxes) + BP0
    combined_RPs = -2.5*np.log10(primary_RP_fluxes + secondary_RP_fluxes) + RP0
    
    # calculate photocenter radii
    q = secondary_masses/primary_masses
    eps = secondary_G_fluxes/primary_G_fluxes
    semi_as = ((binary_Porbs/365.25)**(2/3))*((primary_masses + secondary_masses)**(1/3)) # AU
    photocenter_rads = np.abs(semi_as*(1/(1+eps) - 1/(1+q))) # AU

    # perform fit and record the following parameters
    # stellar_params: [ra, dec, parallax, pmra, pmdec, primary mass, secondary mass, combined mass, primary G-mag, secondary G-mag, primary BP-mag, secondary BP-mag, primary RP-mag, secondary RP-mag]
    # astrom_data: [t_ast_yr, psi, plx_factor, ast_obs, ast_err]
    # true_params, recov_params: [apparent planet mass, period, Tp, ecc, omega, inc, w]
    # no_pl_chi2s, one_pl_chi2s: floats
    recorded_count = 0
    start = time.time()
    for i in range(len(primary_masses)):
        if binary_Porbs[i]/365.25 < 7.0: # continue if Porb < 7 years
            # calculate astrometric SNR, and the implied planet mass if the 'apparent' stellar mass is simply the primary mass
            astrom_signal = 1e6*photocenter_rads[i]/dists[i] # mircro-arcsec
            sigma_fov = piecewise(np.array([combined_Gs[i]]), 53.89888779)[0]
            inf_planet_mass = 1047.57*get_planet_mass(photocenter_rads[i], binary_Porbs[i]/365.25, primary_masses[i])
            
            # continue if N_sigma > 0.5 and the nominal mp < 30 M_J (the true mp will be lower mass)
            if astrom_signal/sigma_fov > 0.5 and inf_planet_mass < 30.0:
                # perform isochrone fitting
                bounds = [(0.1, 3.0), (0.0, 12.0), (-0.3, 0.3)]
                neg_log_like = lambda *args: -log_likelihood(*args)
                soln = optimize.differential_evolution(neg_log_like, bounds=bounds, args=(combined_Gs[i], combined_BPs[i], combined_RPs[i], dists[i]))
                combined_stellar_mass = soln.x[0]
                inf_planet_mass = 1047.57*get_planet_mass(photocenter_rads[i], binary_Porbs[i]/365.25, combined_stellar_mass)

                # continue if the planet mass implied by the 'apparent' stellar mass is < 30 M_J
                if inf_planet_mass < 30.0:
                    # choose random orbital orentiation and phase
                    Tp = np.random.uniform(0.0, binary_Porbs[i])/(2*np.pi)
                    omega = np.random.uniform(0.0, 2*np.pi)
                    w = np.random.uniform(0.0, 2*np.pi)
                    inc = np.arccos(np.random.uniform(-1.0, 1.0))

                    # generate synthetic astrometric data
                    t_ast_yr, psi, plx_factor, ast_obs, ast_err = gaiamock.predict_astrometry_luminous_binary(ra=ras[i], dec=decs[i], parallax=parallaxes[i], pmra=pmras[i], pmdec=pmdecs[i], m1=primary_masses[i], m2=secondary_masses[i], period=binary_Porbs[i], Tp=Tp, ecc=binary_eccs[i], omega=omega, inc=inc, w=w, phot_g_mean_mag=combined_Gs[i], f=eps[i], data_release='dr4', c_funcs=c_funcs)
                    astrom_data = np.array([t_ast_yr, psi, plx_factor, ast_obs, ast_err])
                    stellar_params = np.array([ras[i], decs[i], parallaxes[i], pmras[i], pmdecs[i], primary_masses[i], secondary_masses[i], combined_stellar_mass, primary_Gs[i], secondary_Gs[i], primary_BPs[i], secondary_BPs[i], primary_RPs[i], secondary_RPs[i]])
                    true_params = np.array([inf_planet_mass, binary_Porbs[i], Tp, binary_eccs[i], omega, inc, w])

                    # get no-planet solution
                    Cinv = np.diag(1/ast_err**2)
                    M = np.vstack([np.sin(psi), t_ast_yr*np.sin(psi), np.cos(psi), t_ast_yr*np.cos(psi), plx_factor]).T
                    mu = np.linalg.solve(M.T @ Cinv @ M, M.T @ Cinv @ ast_obs) #  ra, pmra, dec, pmdec, parallax
                    no_pl_Lambda_pred = np.dot(M, mu)
                    no_pl_chi2s = np.sum((ast_obs - no_pl_Lambda_pred)**2/ast_err**2)

                    # get non-linear astrometric parameters (P, phi_p, ecc)
                    theta_array = gaiamock.fit_orbital_solution_nonlinear(t_ast_yr=t_ast_yr, psi=psi, plx_factor=plx_factor, ast_obs=ast_obs, ast_err=ast_err, c_funcs=c_funcs)

                    # solve for linear parameters
                    chi2, mu_linear = gaiamock.get_astrometric_chi2(t_ast_yr=t_ast_yr, psi=psi, plx_factor=plx_factor, ast_obs=ast_obs, ast_err=ast_err, P=theta_array[0], phi_p=theta_array[1], ecc=theta_array[2], c_funcs=c_funcs)
                    ra_off, pmra, dec_off, pmdec, plx, B, G, A, F = mu_linear
                    a0, Omega, w, inc = gaiamock.get_Campbell_elements(A, B, F, G)

                    # convert to apparent planet mass (assume stellar mass is known)
                    mass_pl = get_planet_mass(a0/plx, theta_array[0]/365.25, combined_stellar_mass)
                    one_pl_chi2s = chi2
                    recov_params = np.array([mass_pl, theta_array[0], theta_array[0]*theta_array[1]/(2*np.pi), theta_array[2], Omega, inc, w])

                    # if \Delta \chi2 > 50 and mp < 13 M_J, run MCMC analysis and save the results
                    if no_pl_chi2s - one_pl_chi2s > 50.0 and 1047.57*mass_pl < 13.0:
                        with open("data/Gaia_binaries_recover_DR4/file" + str(run_num) + "_" + str(recorded_count) + ".pkl", "wb") as f:
                            pickle.dump(stellar_params, f)
                            pickle.dump(astrom_data, f)
                            pickle.dump(true_params, f)
                            pickle.dump(recov_params, f)
                            pickle.dump(no_pl_chi2s, f)
                            pickle.dump(one_pl_chi2s, f)

                        best_fit = np.array([ra_off, dec_off, 1000/plx, pmra, pmdec, theta_array[0], theta_array[2], theta_array[1], A, B, F, G])

                        n_walkers = 100
                        p0 = get_rand_p0(best_fit, n_walkers)
                        sampler = emcee.EnsembleSampler(n_walkers, 12, log_probability, args=(t_ast_yr, psi, plx_factor, ast_obs, ast_err))
                        samples = sampler.run_mcmc(p0, 10_000, progress=False)
                        corner_chain = sampler.chain[:, 5000:, :].reshape((-1, 12))

                        # convert to parameters of interest
                        # ra_off, dec_off, dist, pmra, pmdec, period, ecc, phi_p, A, B, F, G
                        MCMC_ra_offs, MCMC_dec_offs = corner_chain[:,0], corner_chain[:,1]
                        MCMC_dists = corner_chain[:,2]
                        MCMC_parallaxes = 1000/MCMC_dists
                        MCMC_pmras, MCMC_pmdecs = corner_chain[:,3], corner_chain[:,4]
                        MCMC_periods, MCMC_eccs, MCMC_phi_ps = corner_chain[:,5], corner_chain[:,6], corner_chain[:,7]
                        MCMC_As, MCMC_Bs, MCMC_Fs, MCMC_Gs = corner_chain[:,8], corner_chain[:,9], corner_chain[:,10], corner_chain[:,11]
                        MCMC_a0s, MCMC_Omegas, MCMC_ws, MCMC_incs = gaiamock.get_Campbell_elements(MCMC_As, MCMC_Bs, MCMC_Fs, MCMC_Gs)
                        MCMC_planet_masses = get_planet_mass(MCMC_a0s/MCMC_parallaxes, MCMC_periods/365.25, combined_stellar_mass)

                        # save median, 1-sigma range, and 95% range
                        percentiles_to_record = np.array([5.0, 15.87, 50.0, 84.14, 95.0])
                        MCMC_results = np.zeros((5, len(percentiles_to_record)))

                        dist_percentiles = []
                        for percentile in percentiles_to_record:
                            dist_percentiles.append(np.percentile(MCMC_dists, percentile))
                        MCMC_results[0] = np.array(dist_percentiles)

                        period_percentiles = []
                        for percentile in percentiles_to_record:
                            period_percentiles.append(np.percentile(MCMC_periods, percentile))
                        MCMC_results[1] = np.array(period_percentiles)

                        mass_percentiles = []
                        for percentile in percentiles_to_record:
                            mass_percentiles.append(np.percentile(MCMC_planet_masses, percentile))
                        MCMC_results[2] = np.array(mass_percentiles)

                        ecc_percentiles = []
                        for percentile in percentiles_to_record:
                            ecc_percentiles.append(np.percentile(MCMC_eccs, percentile))
                        MCMC_results[3] = np.array(ecc_percentiles)

                        inc_percentiles = []
                        for percentile in percentiles_to_record:
                            inc_percentiles.append(np.percentile(MCMC_incs, percentile))
                        MCMC_results[4] = np.array(inc_percentiles)

                        with open("data/Gaia_binaries_recover_DR4/MCMC" + str(run_num) + "_" + str(recorded_count) + ".pkl", "wb") as f:
                            pickle.dump(MCMC_results, f) # dists, periods, masses, eccs, incs

                        recorded_count += 1
    
    end = time.time()
    print('done! time elapsed:', end-start)
