# imports
import numpy as np
from tqdm import tqdm
import gaiamock # note: will only work if executed in the same directory as 'gaiamock.py'
import pickle
from scipy.misc import derivative
from scipy.integrate import trapz, cumtrapz
import os
from tqdm import tqdm
from scipy.interpolate import CubicSpline

# read in the compiled C functions
c_funcs = gaiamock.read_in_C_functions()

# function to calculate absolute G-band magnitude from stellar mass
def calc_absGmag(mass, coeff=np.array([16.91206805,  24.95258266,   4.0433877 , -13.63108438, 4.77092427])):
    return np.polyval(coeff, np.log10(mass))

# load data tabulated from Fig. 16 of Gaia Collaboration et al. (2021)
data = np.genfromtxt('data/Gaia_VLF_tabulated.csv', delimiter=', ')
Gaia_MGs, Gaia_VLFs = data[:,0], data[:,1]
VLF_cubic_spline = CubicSpline(Gaia_MGs, Gaia_VLFs)

# calculate VLF = dN/dVdMG for some set of absolute mags
def calc_dN_dVdMG(MGs):
    return VLF_cubic_spline(MGs)

# calculate the derivative of absolute magnitude with respect to mass
def calc_dabsGmag_dmass(mass):
    return derivative(calc_absGmag, mass, dx=1e-10)

# function to calculate VMF = dN/dVdM for a set of stellar masses
def VMF(M_star):
    MG = calc_absGmag(M_star)
    dN_dVdMG = calc_dN_dVdMG(MG)
    dMG_dM = calc_dabsGmag_dmass(M_star)
    return dN_dVdMG * np.abs(dMG_dM)

# setup VMF grid and calculate CDF
M_min, M_max = 0.1, 2.0
M_grid = np.linspace(M_min, M_max, 10_000)
VMF_grid = VMF(M_grid)
VMF_PDF = VMF_grid/trapz(y=VMF_grid, x=M_grid)
VMF_CDF = cumtrapz(y=VMF_PDF, x=M_grid, initial=0.0)
def sample_VMF(VMF_CDF, N=1):
    u = np.random.uniform(0.0, 1.0, N)
    return np.interp(u, VMF_CDF, M_grid)

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

if __name__ == "__main__":
    run_num = int(os.environ['SLURM_PROCID'])

    # sample positions from an exponential disk
    num_stars = 100_000
    d_min, d_max = 0.0, 800.0 # pc
    ras, decs, dists, _, _, _ = gaiamock.generate_coordinates_at_a_given_distance_exponential_disk(d_min=d_min, d_max=d_max, N_stars=num_stars)

    # choose random proper motions
    parallaxes = 1000/dists # mas
    simga_v = 30 # km/s
    sigma_mu = 1e3*simga_v/(4.74*dists) # mas/yr
    pmras = np.random.normal(0.0, sigma_mu)
    pmdecs = np.random.normal(0.0, sigma_mu)

    # sample stellar mass
    masses = sample_VMF(VMF_CDF, N=num_stars)
    MGs = calc_absGmag(masses)
    Gs = MGs + 5*np.log10(dists) - 5.0

    # assign planets: described via [mass, period, Tp, ecc, omega, inc, w]
    planets = np.zeros((num_stars, 7))
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
    for i in range(len(max_occ_rates)):
        if rand_nums[i] < occ_rates[i]: # stars around which to assign planets
            planet_mass, semi_a = -1, -1

            # rejection sampling to determine planet mass and semi-major axis
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
    recorded_count = 0
    for i in tqdm(range(num_stars)):
        if planets[i][0] != 0.0:
            # calculate astrometric SNR
            semi_a = ((planets[i][1]/365.25)**(2/3))*(masses[i]**(1/3)) # AU
            dist = 1000/parallaxes[i] # pc
            astrom_signal = 1e6*(planets[i][0]/masses[i])*(semi_a/dist)
            sigma_fov = piecewise(np.array([Gs[i]]), 53.89888779)[0]
            
            if astrom_signal/sigma_fov > 1.0:
                # generate synthetic astrometric data
                t_ast_yr, psi, plx_factor, ast_obs, ast_err = gaiamock.predict_astrometry_luminous_binary(ra=ras[i], dec=decs[i], parallax=parallaxes[i], pmra=pmras[i], pmdec=pmdecs[i], m1=masses[i], m2=planets[i][0], period=planets[i][1], Tp=planets[i][2], ecc=planets[i][3], omega=planets[i][4], inc=planets[i][5], w=planets[i][6], phot_g_mean_mag=Gs[i], f=0.0, data_release='dr5', c_funcs=c_funcs)
                astrom_data = np.array([t_ast_yr, psi, plx_factor, ast_obs, ast_err])
                stellar_params = np.array([ras[i], decs[i], parallaxes[i], pmras[i], pmdecs[i], masses[i], Gs[i]])
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
                semi_a_pl = ((theta_array[0]/365.25)**(2/3))*(masses[i]**(1/3))
                mass_pl = a0*masses[i]/(plx*semi_a_pl)

                one_pl_chi2s = chi2
                recov_params = np.array([mass_pl, theta_array[0], theta_array[0]*theta_array[1]/(2*np.pi), theta_array[2], Omega, inc, w])

                # save results
                with open("data/Pmax_experiment_DR5/file" + str(run_num) + "_" + str(recorded_count) + ".pkl", "wb") as f:
                    pickle.dump(stellar_params, f)
                    pickle.dump(astrom_data, f)
                    pickle.dump(true_params, f)
                    pickle.dump(recov_params, f)
                    pickle.dump(no_pl_chi2s, f)
                    pickle.dump(one_pl_chi2s, f)
                    
                recorded_count += 1
