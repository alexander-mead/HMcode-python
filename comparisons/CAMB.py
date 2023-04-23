# Standard imports
import sys

# Third-part imports
import numpy as np

# Project imports
import hmcode
import hmcode.camb_stuff as camb_stuff
import hmcode.utility as util

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
zs = [3., 2., 1., 0.5, 0.]
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
    w0 = rng.uniform(w0_min, w0_max) if vary_w0 else -1.
    while True: # Ensure that dark energy does not dominate the early universe
        wa = rng.uniform(wa_min, wa_max) if vary_wa else 0.
        if w0+wa < 0.: break
    m_nu = rng.uniform(m_nu_min, m_nu_max) if vary_m_nu else 0.

    # Get stuff from CAMB
    _, results, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa)
    Pk_nonlin_interp = results.get_matter_power_interpolator(nonlinear=True).P

    # Arrays for CAMB non-linear spectrum
    Pk_CAMB = np.zeros((len(zs), len(k)))
    for iz, z in enumerate(zs):
        Pk_CAMB[iz, :] = Pk_nonlin_interp(z, k)

    # Get the new pyHMcode spectrum
    Pk_HMcode = hmcode.power(k, zs, results, verbose=verbose)

    # Calculate maximum deviation between pyHMcode and the version in CAMB
    max_error = np.max(np.abs(-1.+Pk_HMcode/Pk_CAMB))
    max_errors.append(max_error)
    print('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu) = \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}); \
maximum deviation: {:.2%}'.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, max_error))

# Write worst error to screen
max_errors = np.array(max_errors)
print('Mean error: {:.2%}'.format(max_errors.mean()))
print('Std error: {:.2%}'.format(max_errors.std()))
print('Worst error: {:.2%}'.format(max_errors.max()))