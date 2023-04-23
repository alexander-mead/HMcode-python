# Standard imports
import sys

# Third-part imports
import numpy as np
import matplotlib.pyplot as plt

# Project imports
import hmcode
import hmcode.camb_stuff as camb_stuff
import hmcode.utility as util

# Make plots or not
plotting = False

# Vary these parameters
vary_Omega_k = True
vary_w0 = True
vary_wa = True
vary_m_nu = True

# Parameter ranges
Omega_c_min, Omega_c_max = 0.2, 0.4
Omega_b_min, Omega_b_max = 0.035, 0.065
Omega_k_min, Omega_k_max = -0.05, 0.05
h_min, h_max = 0.5, 0.9
ns_min, ns_max = 0.9, 1.0
sigma_8_min, sigma_8_max = 0.7, 0.9
w0_min, w0_max = -1.3, -0.7
wa_min, wa_max = -1.73, 1.28
m_nu_min, m_nu_max = 0., 1.
log10T_AGN_min, log10T_AGN_max = 7.6, 8.3

# Number of cosmological models to test
try:
    ncos = int(sys.argv[1])
except:
    ncos = 5
verbose = (ncos == 1)

# Wavenumbers [h/Mpc]
kmin, kmax = 1e-3, 1e1
nk = 128
k = util.logspace(kmin, kmax, nk)

# Redshifts
zs = [1.5, 1., 0.5, 0.]
zs = np.array(zs)

# Seed random number generator
rng = np.random.default_rng(seed=42)

# Loop over cosmologies
max_errors = []
for icos in range(ncos):

    # Cosmology
    Omega_c = rng.uniform(Omega_c_min, Omega_c_max)
    Omega_b = rng.uniform(Omega_b_min, Omega_b_max)
    Omega_k = rng.uniform(Omega_k_min, Omega_k_max) if vary_Omega_k else 0.
    h = rng.uniform(h_min, h_max)
    ns = rng.uniform(ns_min, ns_max)
    sigma_8 = rng.uniform(sigma_8_min, sigma_8_max)
    log10T_AGN = rng.uniform(log10T_AGN_min, log10T_AGN_max)

    w0 = rng.uniform(w0_min, w0_max) if vary_w0 else -1.
    while True: # Ensure that dark energy does not dominate the early universe
        wa = rng.uniform(wa_min, wa_max) if vary_wa else 0.
        if w0+wa < 0.: break
    m_nu = rng.uniform(m_nu_min, m_nu_max) if vary_m_nu else 0.

    ### Gravity only spectra ###

    # Get stuff from CAMB
    _, results_gravity, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa)
    Pk_nonlin_interp = results_gravity.get_matter_power_interpolator(nonlinear=True).P

    # Arrays for CAMB non-linear spectrum
    Pk_CAMB_gravity = np.zeros((len(zs), len(k)))
    for iz, z in enumerate(zs):
        Pk_CAMB_gravity[iz, :] = Pk_nonlin_interp(z, k)

    # Get the new pyHMcode spectrum
    Pk_HMcode_gravity = hmcode.power(k, zs, results_gravity, verbose=verbose)

    ### ###

    ### Spectra with feedback ###

    # Get stuff from CAMB
    _, results_feedback, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa, log10_T_AGN=log10T_AGN)
    Pk_nonlin_interp = results_feedback.get_matter_power_interpolator(nonlinear=True).P

    # Arrays for CAMB non-linear spectrum
    Pk_CAMB_feedback = np.zeros((len(zs), len(k)))
    for iz, z in enumerate(zs):
        Pk_CAMB_feedback[iz, :] = Pk_nonlin_interp(z, k)

    # Get the new pyHMcode spectrum
    Pk_HMcode_feedback = hmcode.power(k, zs, results_gravity, verbose=verbose, T_AGN=np.power(10, log10T_AGN))

    ### ###

    # Calculate suppressions
    Rk_CAMB=Pk_CAMB_feedback/Pk_CAMB_gravity
    Rk_HMcode=Pk_HMcode_feedback/Pk_HMcode_gravity

    # Plotting
    if plotting:
        fig, axs=plt.subplots(nrows=2, sharex=True, figsize=(10,7))
        plt.subplots_adjust(hspace=0.001, bottom=0.2)

        fig.suptitle('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu, log10(T_AGN/K)) = \n \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2})'.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, log10T_AGN))
        axs[0].set_ylabel(r'$R(k)=P(k)/P_\mathrm{no-feedback}(k)$')
        axs[1].set_ylabel(r'$R_\mathrm{CAMB}(k)/R_\mathrm{HMcode-Python}(k)-1$')

        for ax in axs:
           ax.set_xscale('log')
        axs[1].set_xlabel(r'$k [h/\mathrm{Mpc}]$')

        for i,z in enumerate(zs):
            axs[0].plot(k, Rk_CAMB[i], label=f"CAMB, z={z}", color=f"C{i}", ls='--')
            axs[0].plot(k, Rk_HMcode[i], label=f"HMcode-Python, z={z}", color=f"C{i}")
            axs[1].plot(k, 1-(Rk_CAMB[i]/Rk_HMcode[i]), label=f"z={z}", color=f"C{i}")

        axs[0].legend()
        axs[1].legend()
        plt.savefig(f"plots/Feedback_Cosmology_{icos}.png")
        plt.close()

    # Calculate maximum deviation between pyHMcode and the version in CAMB
    max_error = np.max(np.abs(-1.+Rk_HMcode/Rk_CAMB))
    max_errors.append(max_error)
    print('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu, log10(T_AGN/K)) = \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}); \
maximum deviation: {:.2%}'.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, log10T_AGN, max_error))

# Write worst error to screen
max_errors = np.array(max_errors)
print('Mean error: {:.2%}'.format(max_errors.mean()))
print('Std error: {:.2%}'.format(max_errors.std()))
print('Worst error: {:.2%}'.format(max_errors.max()))