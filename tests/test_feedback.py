# Standard imports
import unittest

# Third-party imports
import numpy as np

# Project imports
import hmcode
import hmcode.camb_stuff as camb_stuff

### Test data ###

# Read cosmologies
cosmologies_file = 'benchmarks_feedback/cosmologies.txt'
cosmologies = np.loadtxt(cosmologies_file)
ncos = len(cosmologies)

# Redshifts
zs = np.array([3., 2., 1., 0.5, 0.])

# Loop over cosmologies
benchmarks = []; data = []
for icos in range(ncos):

    # Read benchmark
    infile = f'benchmarks_feedback/cosmology_{icos}.dat'
    benchmark = np.loadtxt(infile)
    k, Supp_bench = benchmark[0, :], benchmark[1:, :]
    benchmarks.append(Supp_bench)

    # Get cosmological parameters
    Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, T_AGN = cosmologies[icos]
    
    # Get stuff from CAMB
    _, results, _, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, m_nu, w0, wa)

    # Get the pyHMcode spectrum
    Pk_feedback = hmcode.power(k, zs, results, T_AGN=T_AGN, verbose=False)
    Pk_gravity = hmcode.power(k, zs, results, T_AGN=None)
    Supp_HMcode = Pk_feedback/Pk_gravity
    data.append(Supp_HMcode)

    # Write cosmological parameters and  to screen
    max_deviation = np.max(np.abs(-1.+Supp_HMcode/Supp_bench))
    print('Cosmology: {:d}; (Om_c, Om_b, Om_k, h, ns, sig8, w0, wa, m_nu, log10(T_AGN/K)) = \
({:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}); \
maximum deviation: {:.4%}'.format(icos, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, w0, wa, m_nu, np.log10(T_AGN), max_deviation))
    
### Tests ###

class TestPower(unittest.TestCase):

    @staticmethod
    def test():
        for benchmark, datum in zip(benchmarks, data):
            np.testing.assert_array_almost_equal(datum/benchmark, 1., decimal=5)

## ###

## Unittest ###

if __name__ == '__main__':
    unittest.main()

## ###



