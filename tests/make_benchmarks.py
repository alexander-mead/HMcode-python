# Third-party imports
import numpy as np

# Project imports
import hmcode
import hmcode.utility as util
import hmcode.camb_stuff as camb_stuff

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

# Wavenumbers [h/Mpc]
kmin, kmax = 1e-3, 1e1
nk = 128
k = util.logspace(kmin, kmax, nk)

# Redshifts
zs = [3., 2., 1., 0.5, 0.]
zs = np.array(zs)

# Number of cosmologies to generate
ncos = 25

# Seed random number generator
rng = np.random.default_rng(seed=42)

# Loop over cosmologies
cosmologies = []
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

    print('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu) = \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2})'
.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu))
    cosmologies.append([Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu])

    # Get stuff from CAMB
    _, results, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa)

    # Get the pyHMcode spectrum
    Pk_HMcode = hmcode.power(k, zs, results)
    data = np.vstack((k, Pk_HMcode))
    outfile = f'benchmarks/cosmology_{icos}.dat'
    with open(outfile, 'x') as f:
        np.savetxt(f, data, header='k [h/Mpc]; P_mm(k) [(Mpc/h)^3] at z = [3, 2, 1, 0.5, 0]')

outfile = 'benchmarks/cosmologies.txt'
with open(outfile, 'x') as f:
    np.savetxt(f, cosmologies, header='Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu')
