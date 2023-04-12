# Standard imports
import numpy as np
import unittest
import sys
sys.path.append("../")

# Project imports
import hmcode
import hmcode.camb_stuff as camb_stuff

### Test data ###

# Read cosmologies
cosmologies_file = 'benchmarks_baryons/cosmologies.txt'
cosmologies = np.loadtxt(cosmologies_file)
ncos = len(cosmologies)

# Redshifts
zs = np.array([3., 2., 1., 0.5, 0.])

# Loop over cosmologies
benchmarks = []; data = []
for icos in range(ncos):

    # Read benchmark
    infile = f'benchmarks_baryons/cosmology_{icos}.dat'
    benchmark = np.loadtxt(infile)
    k, Supp_bench = benchmark[0, :], benchmark[1:, :]
    benchmarks.append(Supp_bench)

    # Get cosmological parameters
    Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, T_AGN = cosmologies[icos]
    
    # Get stuff from CAMB
    _, results, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa)

    # Get the pyHMcode spectrum
    Supp_HMcode = hmcode.get_Baryon_Suppression(k, zs, results, T_AGN)
    data.append(Supp_HMcode)

    # Write cosmological parameters and  to screen
    max_deviation = np.max(np.abs(-1.+Supp_HMcode/Supp_bench))
    print('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu, T_AGN) = \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2e}); \
maximum deviation: {:.4%}'.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, T_AGN, max_deviation))



