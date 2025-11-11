# imports
import numpy as np
import gaiamock # note: will only work if executed in the same directory as 'gaiamock.py'
import pickle
import os
import time
from tqdm import tqdm
import emcee

# read in the compiled C functions
c_funcs = gaiamock.read_in_C_functions()

# simple model for Gaia's astrometric uncertainty
def piecewise(x, a):
    return a*np.max([1.0*np.ones(len(x)), 10**(0.2*(x - 14.0))], axis=0)

# calculate the planet occurrence rate for a given stellar mass, semi-major axis, and planet mass (from Fulton et al. 2021)
def planet_occurrence(M_star, a, C=350.0, beta=-0.86, a0 = 3.6, gamma=1.59):
    C_new = C*(M_star/0.9)
    occurrence_rate = C_new * (a**beta) * (1.0 - np.exp(-(a/a0)**gamma))
    dlogm = np.log(6000) - np.log(30)
    occurrence_rate = occurrence_rate/(719.0*dlogm*0.63)

    return occurrence_rate # returns dNplanets / (dNstars * dlnm * dlna)

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

if __name__ == "__main__":
    run_num = int(os.environ['SLURM_PROCID'])
    out_path = "data/Gaia_real_stars_recover_DR5/file" + str(run_num) + ".pkl"

    # load subset of searchable main-sequence stars
    # these files are too big for GitHub, but can be re-generated using the notebook on GitHub
    f = open("data/Gaia_relevant_stars_MS_final_chunked/file" + str(run_num) + ".pkl", "rb")
    source_ids = pickle.load(f)
    ras = pickle.load(f)
    decs = pickle.load(f)
    parallaxes = pickle.load(f)
    pmras = pickle.load(f)
    pmdecs = pickle.load(f)
    G_mags = pickle.load(f)
    BPs = pickle.load(f)
    RPs = pickle.load(f)
    ruwes = pickle.load(f)
    masses = pickle.load(f)
    f.close()
    dists = 1000/parallaxes # pc

    # assign planets: described via [mass, period, Tp, ecc, omega, inc, w]
    planets = np.zeros((len(source_ids), 7))
    a_min, a_max = 0.125, 7.0
    m_min, m_max = 0.3, 13.0
    alog = np.linspace(np.log(a_min), np.log(a_max), 1000)
    mlog = np.linspace(np.log(m_min), np.log(m_max), 1000)
    dlna = alog[1]-alog[0]
    dlnm = mlog[1]-mlog[0]
    amesh, mmesh = np.meshgrid(np.exp(alog), np.exp(mlog))
    solar_mass_occmesh = planet_occurrence(1.0, amesh)
    solar_max_occ = np.max(solar_mass_occmesh) # useful for rejection sampling later
    solar_mass_occ = np.sum(solar_mass_occmesh) * dlnm * dlna
    occ_rates = solar_mass_occ*(masses/1.0) # rescale based on stellar mass
    max_occ_rates = solar_max_occ*(masses/1.0) # useful for rejection sampling later
    rand_nums = np.random.uniform(0.0, 1.0, size=len(occ_rates))
    for i in range(len(rand_nums)):
        if rand_nums[i] < occ_rates[i]: # stars around which to assign planets
            planet_mass, semi_a = -1, -1

            # rejection sampling to select planet masses semi-major axes
            while planet_mass < 0 and semi_a < 0:
                rand_a = np.random.uniform(a_min, a_max)
                rand_lnmp = np.random.uniform(np.log(m_min), np.log(m_max))

                if np.random.rand() < planet_occurrence(masses[i], rand_a)/(max_occ_rates[i]):
                    semi_a, planet_mass = rand_a, np.exp(rand_lnmp)/1047.57
                    period = 365.25*(semi_a**1.5)/(masses[i]**0.5) # days
                    Tp = np.random.uniform(0.0, period)/(2*np.pi)
                    ecc = np.random.beta(a=0.867, b=3.03)
                    omega = np.random.uniform(0.0, 2*np.pi)
                    inc = np.arccos(np.random.uniform(-1.0, 1.0)) # between 0 and pi
                    w = np.random.uniform(0.0, 2*np.pi)
                    planets[i] = [planet_mass, period, Tp, ecc, omega, inc, w]

    # perform fit and record the following parameters
    # stellar_params: [ra, dec, parallax, pmra, pmdec, mass, G-mag]
    # astrom_data: [t_ast_yr, psi, plx_factor, ast_obs, ast_err]
    # true_params, recov_params: [planet_mass, period, Tp, ecc, omega, inc, w]
    # no_pl_chi2s, one_pl_chi2s: floats
    start = time.time()
    with open(out_path, "ab") as fout: # open file, which will be written to continually
        for i in range(len(source_ids)):
            if planets[i][0] != 0.0:
                # calculate astrometric SNR
                semi_a = ((planets[i][1]/365.25)**(2/3))*(masses[i]**(1/3)) # AU
                astrom_signal = 1e6*(planets[i][0]/masses[i])*(semi_a/dists[i])
                sigma_fov = piecewise(np.array([G_mags[i]]), 53.89888779)[0]

                # proceed if N_sigma > 0.5 and Porb < 14 years
                if astrom_signal/sigma_fov > 0.5 and planets[i][1]/365.25 < 14.0:
                    # generate synthetic astrometric data
                    t_ast_yr, psi, plx_factor, ast_obs, ast_err = gaiamock.predict_astrometry_luminous_binary(ra=ras[i], dec=decs[i], parallax=parallaxes[i], pmra=pmras[i], pmdec=pmdecs[i], m1=masses[i], m2=planets[i][0], period=planets[i][1], Tp=planets[i][2], ecc=planets[i][3], omega=planets[i][4], inc=planets[i][5], w=planets[i][6], phot_g_mean_mag=G_mags[i], f=0.0, data_release='dr5', c_funcs=c_funcs)
                    astrom_data = np.array([t_ast_yr, psi, plx_factor, ast_obs, ast_err])
                    stellar_params = np.array([source_ids[i], ras[i], decs[i], parallaxes[i], pmras[i], pmdecs[i], masses[i], G_mags[i], BPs[i], RPs[i], ruwes[i]])
                    true_params = np.array(planets[i])

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

                    # convert to planet mass (assume stellar mass is known)
                    mass_pl = get_planet_mass(a0/plx, theta_array[0]/365.25, masses[i])
                    one_pl_chi2s = chi2
                    recov_params = np.array([mass_pl, theta_array[0], theta_array[0]*theta_array[1]/(2*np.pi), theta_array[2], Omega, inc, w])

                    # if \Delta \chi2 > 50, run MCMC analysis and save the results
                    if no_pl_chi2s - one_pl_chi2s > 50.0:
                        best_fit = np.array([ra_off, dec_off, 1000/plx, pmra, pmdec, theta_array[0], theta_array[2], theta_array[1], A, B, F, G])
                        unc_factor = np.max([1.0, np.sqrt(chi2/(len(ast_obs) - 12))]) # inflate uncertainty if needed

                        n_walkers = 100
                        p0 = get_rand_p0(best_fit, n_walkers)
                        sampler = emcee.EnsembleSampler(n_walkers, 12, log_probability, args=(t_ast_yr, psi, plx_factor, ast_obs, unc_factor*ast_err))
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
                        MCMC_planet_masses = get_planet_mass(MCMC_a0s/MCMC_parallaxes, MCMC_periods/365.25, masses[i])

                        # save median, 1-sigma range, and 90% range
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
                        
                        # save to file
                        list_to_record = [stellar_params, astrom_data, true_params, recov_params, no_pl_chi2s, one_pl_chi2s, MCMC_results]
                        pickle_data = pickle.dumps(list_to_record, protocol=pickle.HIGHEST_PROTOCOL)
                        fout.write(pickle_data)
                        fout.flush()
                        os.fsync(fout.fileno())
    
    end = time.time()
    print('Done! time elapsed:', end-start)
