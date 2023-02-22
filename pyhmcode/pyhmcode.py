# Standard imports
import numpy as np

# Third-party imports
import camb
import pyhalomodel as halo

# Project imports
from . import utility as util
from . import cosmology
from . import linear_growth

'''
To whom it may concern,

I coded this Python version of HMcode up quite quickly before leaving academia. 
It is written in pure Python and doesn't use any of the original Fortran code whatsoever.
There is something amazing/dispiriting about coding something up in 3 days that previously took 5 years.
A tragic last hoorah! At least I switched to Python eventually...

I tested it against the CAMB HMcode version for 100 random sets of cosmological parameters (k < 10 h/Mpc; z < 3). 
The accuracy is as follows:
    - LCDM: Mean error: 0.10%; Std error: 0.03%; Worst error; 0.21%
    - k-LCDM: Mean error: 0.11%; Std error: 0.03%; Worst error; 0.23%
    - w-CDM: Mean error: 0.10%; Std error: 0.03%; Worst error; 0.20%
    - w(a)-CDM: Mean error: 0.13%; Std error: 0.06%; Worst error; 0.48%
    - nu-LCDM: Mean error: 0.47%; Std error: 0.44%; Worst error; 2.01% (larger errors strongly correlated with neutrino mass)
    - nu-k-w(a)CDM: Mean error: 0.42%; Std error: 0.43%; Worst error; 2.02% (larger errors strongly correlated with neutrino mass)
These tests can be reproduced using the tests/test_HMcode.py script.

Note that the quoted accuracy of HMcode relative to simulations is RMS ~2.5%.
Note also that the accuracy is anti-correlated with neutrino masses (cf. Fig. 2 of https://arxiv.org/abs/2009.01858).
The larger discrepancies for massive neutrinos (2% for ~1eV) may seem worrisome, but here are some reasons that I am not that worried:
    - Here neutrinos are treated as completely cold matter when calculating the linear growth factors
    - In CAMB-HMcode the transition from behaving like radiation to behaving like matter is accounted for in the linear growth
    - Here the cold matter power spectrum is taken directly from CAMB.
    - In CAMB-HMcode the *cold* matter power spectrum is calculated approximately using Eisenstein & Hu (1999)
Using the actual cold matter spectrum is definitely a (small) improvement.
Ignoring the actual energy-density scaling of massive neutrinos might seem to be a (small) problem,
but keep in mind the comments below regarding the linear growth factor.

I think any residual differences must stem from:
    - The BAO de-wiggling process
    - The accuracy of the sigmaV numerical integration
    - The accuracy of the sigmaR numerical integration (using CAMB here; done internally in CAMB-HMcode)
    - The accuracy of the linear growth ODE solutions
    - Root finding for the halo-collapse redshift and for Rnl
But I didn't have time to investigate these differences more thoroughly.
Note that there are accuracy parameters in CAMB-HMcode fixed at the 1e-4 level, so you would never expect better than 0.01% agreement.
Although given that HMcode is only accurate at the ~2.5% level compared to simulations the level of agreement seems okay.

While writing this code I had a few ideas for future improvements:
    - Add the baryon-feedback model; this would not be too hard for the enthusiastic student/postdoc.
    - The predictions are a bit sensitive to the smoothing sigma used for the dewiggling. This should probably be a fitted parameter.
    - It's annoying having to calculate linear growth functions (all, LCDM), especially since the linear growth doesn't really exist. 
One should probably should use the P(k) amplitude evolution over time at some cleverly chosen scale instead,
or instead the evolution of sigma(R) over time at some pertinent R value.
Note that the growth factors are *only* used to calculate the Dolag correction and Mead (2017) delta_c, Delta_v.
    - I never liked the halo bloating parameter, it's hard to understand the effect of modifying halo profiles in Fourier space.
Someone should get rid of this (maybe modify the mass function instead?).
    - Redshift 'infinity' for the Dolag correction is actually z = 10. Predictions *are* sensitive to this (particularly w(a)CDM)
Either the redshift 'infinity' should be fitted or the halo-concentration model for beyond-LCDM should be improved somehow.
    - The massive neutrino correction for the Mead (2017) dc, Dv formula (appendix A of https://arxiv.org/abs/2009.01858) is crude.
This should be improved somehow, I guess using the intuition that hot neutrinos are ~smoothly distributed on halo scales.
Currently neutrinos are treated as cold matter in the linear/accumulated growth calculation (used by Mead 2017), which sounds a bit wrong.
    - I haven't checked how fast this code is, but there are a couple of TODO in the code that might improve speed if necessary.
    - The choices regarding how to account for massive neutrinos could usefully be revisited. This whole subject is a bit confusing
and the code doesn't help to alleviate the confusion. Choices like what to use for: dc; Dv; sigma; Rnl; neff; c(M).
    - Refit model (including sigma for BAO and zc for Dolag) to new emulator(s) (e.g., https://arxiv.org/abs/2207.12345).
    - Don't be under any illusions that the HMcode parameters, or the forms of their dependence on the underlying power spectrum,
are special in any particular way. A lot of experimentation went into finding these, but it was by no means exhaustive.
Obviously these parameters should only depend on the underlying spectrum though (rather than being random functions of z or whatever).

Have fun,
Alexander Mead (2023/02/21)
'''

def hmcode(k:np.array, zs:np.array, CAMB_results:camb.CAMBdata, 
           Mmin=1e0, Mmax=1e18, nM=256, verbose=False) -> np.ndarray:
    '''
    Calculates the HMcode matter-matter power spectrum
    Args:
        k: Array of comoving wavenumbers [h/Mpc]
        zs: Array of redshifts
        CAMB_results: CAMBdata structure
    Returns:
        Array of matter power spectra: Pk[z, k]
    '''

    # Checks
    if not util.is_array_monotonic(-zs):
        raise ValueError('Redshift must be monotonically decreasing')

    # Halo mass range
    M = util.logspace(Mmin, Mmax, nM)
    zc = 10. # Redshift 'infinity' for the Dolay correction
    ac = cosmology.scalefactor_from_redshift(zc)

    # Background cosmology at z=0
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c+Om_b+Om_nu
    f_nu = Om_nu/Om_m

    # Linear growth functions (ignoring any radiation contribution at early times)
    growth = linear_growth.get_growth_interpolator(CAMB_results)                 # Standard
    growth_LCDM = linear_growth.get_growth_interpolator(CAMB_results, LCDM=True) # Flatness and w = -1 imposed

    # Useful information
    if verbose:
        print('HMcode parameters')
        print('Halo mass range: 10^{:.1f} -> 10^{:.1f} Msun/h'.format(np.log10(Mmin), np.log10(Mmax)))
        print('Omega_m: {:.3}'.format(Om_m))
        print('Neutrino mass fraction: {:.2%}'.format(f_nu))
        print('Linear growth at z=0: {:.3}'.format(growth(1.)))
        print()

    # Linear power interpolator
    Pk_lin_interp = CAMB_results.get_matter_power_interpolator(nonlinear=False).P

    # Loop over redshift
    Pk_HMcode = np.zeros((len(zs), len(k)))
    for iz, z in enumerate(zs):

        # Background cosmology calculations at this redshift
        # Only used in spherical-collapse calculations for Mead (2017) formulae
        a = cosmology.scalefactor_from_redshift(z)        # Scale factor
        Om_cz = CAMB_results.get_Omega(var='cdm', z=z)    # CDM
        Om_bz = CAMB_results.get_Omega(var='baryon', z=z) # Baryons
        Om_nuz = CAMB_results.get_Omega(var='nu', z=z)    # Massive neutrinos
        Om_mz = Om_cz+Om_bz+Om_nuz                        # Total matter
        if verbose:
            print('Redshift: {:.3}'.format(z))
            print('Scale factor: {:.3}'.format(a))
            print('Omega_m(z): {:.3}'.format(Om_mz))

        # Linear growth and spherical-collapse parameters
        # Only used in spherical-collapse calculations for Mead (2017) formulae
        g = growth(a)                                       # Linear growth
        G = linear_growth.get_accumulated_growth(a, growth) # Accumulated growth
        dc = cosmology.dc_Mead(a, Om_mz, f_nu, g, G)        # Linear collapse threshold
        Dv = cosmology.Dv_Mead(a, Om_mz, f_nu, g, G)        # Halo virial overdensity
        if verbose:
            print('Un-normalisaed growth (= a for a << 1): {:.3}'.format(g))
            print('Normalised growth (= 1 at z = 0): {:.3}'.format(g/growth(1.)))
            print('Accumulated growth (= a for a << 1): {:.3}'.format(G))
            print('Linear collapse threshold: {:.4}'.format(dc))
            print('Virial halo overdensity: {:.4}'.format(Dv))

        # Initialise halo model
        hmod = halo.model(z, Om_m, name='Sheth & Tormen (1999)', Dv=Dv, dc=dc)

        # Linear power and associated quantities
        # Note that the cold matter spectrum is defined via 1+delta_c = rho_c/\bar{rho}_c
        # Where \bar{rho}_c is the mean background *cold* matter density
        # This means that the *cold* linear spectrum and the *cold* field variance are
        # *greater* than the corresponding quantities for the total matter field on scales where
        # massive neutrinos are smoothly distributed
        Pk_lin = Pk_lin_interp(z, k)                         # Linear power spectrum
        R = hmod.Lagrangian_radius(M)                        # Lagrangian radii
        sigmaM = _get_sigmaR(R, iz, CAMB_results, cold=True) # Variance in cold matter field
        nu = hmod._peak_height(M, sigmaM)                    # Halo peak height
        if verbose:
            print('Lagrangian radius range: {:.4} -> {:.4} Mpc/h'.format(R[0], R[-1]))
            print('RMS in matter field range: {:.4} -> {:.4}'.format(sigmaM[0], sigmaM[-1]))
            print('Peak height range: {:.4} -> {:.4}'.format(nu[0], nu[-1]))

        # Parameters of the linear spectrum pertaining to non-linear growth
        Rnl = _get_nonlinear_radius(R[0], R[-1], dc, iz, CAMB_results, cold=True) # Non-linear Lagrangian radius
        sigma8 = _get_sigmaR(8., iz, CAMB_results, cold=True)                     # RMS in the linear cold matter field at 8 Mpc/h
        sigmaV = cosmology.sigmaV(Pk=lambda k: Pk_lin_interp(z, k))               # RMS in the linear displacement field
        neff = _get_effective_index(Rnl, R, sigmaM)                               # Effective index of spectrum at collapse scale
        if verbose:
            print('Non-linear Lagrangian radius: {:.4} Mpc/h'.format(Rnl))
            print('RMS in matter field at 8 Mpc/h: {:.4}'.format(sigma8))
            print('RMS in matter displacement field: {:.4} Mpc/h'.format(sigmaV))
            print('Effective index at collapse scale: {:.4}'.format(neff))

        # HMcode parameters (Table 2 of https://arxiv.org/pdf/2009.01858.pdf)
        kd = 0.05699*sigma8**-1.089  # Two-halo damping wavenumber; equation (16)
        f = 0.2696*sigma8**0.9403    # Two-halo fractional damping; equation (16)
        nd = 2.853                   # Two-halo damping power; equation (16)
        ks = 0.05618*sigma8**-1.013  # One-halo damping wavenumber; equation (17)
        eta = 0.1281*sigma8**-0.3644 # Halo bloating parameter; equation (19)
        B = 5.196                    # Minimum halo concentration; equation (20)
        alpha = 1.875*(1.603)**neff  # Transition smoothing; equation (23)
        if verbose:
           print('Two-halo damping wavenumber: {:.4} h/Mpc'.format(kd))
           print('Two-halo fractional damping: {:.4}'.format(f))
           print('Two-halo damping power: {:.4}'.format(nd))
           print('One-halo damping wavenumber: {:.4} h/Mpc'.format(ks))
           print('Halo bloating: {:.4}'.format(eta))
           print('Minimum halo concentration: {:.4}'.format(B))#, B/4.)
           print('Transition smoothing: {:.4}'.format(alpha))
           print()

        # Halo concentration
        zf = _get_halo_collapse_redshifts(M, z, iz, dc, growth, CAMB_results, cold=True) # Halo formation redshift
        c = B*(1+zf)/(1.+z)                                                              # Halo concentration; equation (20)
        c *= (growth(ac)/growth_LCDM(ac))*(growth_LCDM(a)/growth(a))                     # Dolag correction; equation (22)

        # Halo profile
        # Note the correction for neutrino mass in the profile amplitude here
        rv = hmod.virial_radius(M) 
        Uk = np.ones((len(k), len(M)))
        for iM, (_rv, _c, _nu) in enumerate(zip(rv, c, nu)): # TODO: Remove loop for speed?
            Uk[:, iM] = halo._win_NFW(k*(_nu**eta), _rv, _c)[:, 0]
        profile = halo.profile.Fourier(k, M, Uk, amplitude=M*(1.-f_nu)/hmod.rhom, mass_tracer=True) # NOTE: Factor of 1-f_nu

        # Vanilla power spectrum calculation
        # TODO: Wasteful as this calculate the standard two-halo term unnecessarily
        _, Pk_1h, _ = hmod.power_spectrum(k, Pk_lin, M, sigmaM, {'m': profile})

        # HMcode tweaks
        P_wig = _get_Pk_wiggle(k, Pk_lin, CAMB_results)   # Isolate spectral wiggle; footnote 7
        Pk_dwl = Pk_lin-(1.-np.exp(-(k*sigmaV)**2))*P_wig # Constuct linear spectrum with smoothed wiggle; equation (15)
        Pk_2h = Pk_dwl*(1.-f*(k/kd)**nd/(1.+(k/kd)**nd))  # Two-halo term; equation (16)
        Pk_1h = (k/ks)**4/(1.+(k/ks)**4)*Pk_1h['m-m']     # One-halo term; equation (17)
        Pk_hm = (Pk_2h**alpha+Pk_1h**alpha)**(1./alpha)   # Total prediction via smoothed sum; equation (23)
        Pk_HMcode[iz, :] = Pk_hm

    # Finish
    return Pk_HMcode


def _get_Pk_wiggle(k:np.ndarray, Pk_lin:np.ndarray, CAMB_results:camb.CAMBdata, sigma_dlnk=0.25) -> np.ndarray:
    '''
    Extract the wiggle from the linear power spectrum
    TODO: Should get to work for uneven log(k) spacing
    NOTE: https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
    '''
    from scipy.ndimage import gaussian_filter1d
    if not util.is_array_linear(k): raise ValueError('Dewiggle only works with linearly-spaced k array')
    dlnk = np.log(k[1]/k[0])
    sigma = sigma_dlnk/dlnk
    h = CAMB_results.Params.H0/100.
    omega_m = CAMB_results.Params.omch2+CAMB_results.Params.ombh2+CAMB_results.Params.omnuh2
    omega_b = CAMB_results.Params.ombh2
    T_CMB = CAMB_results.Params.TCMB
    ns = CAMB_results.Params.InitPower.ns
    Pk_nowiggle = (k**ns)*cosmology.Tk_EH_nowiggle(k, h, omega_m, omega_b, T_CMB)**2
    Pk_ratio = Pk_lin/Pk_nowiggle
    Pk_ratio = gaussian_filter1d(Pk_ratio, sigma)
    Pk_smooth = Pk_ratio*Pk_nowiggle
    Pk_wiggle = Pk_lin-Pk_smooth
    return Pk_wiggle


def _get_sigmaR(R:np.ndarray, iz:int, CAMB_results:camb.CAMBdata, cold=False) -> np.ndarray:
    var='delta_nonu' if cold else 'delta_tot'
    sigmaR = CAMB_results.get_sigmaR(R, z_indices=[iz], var1=var, var2=var)[0]
    return sigmaR


def _get_nonlinear_radius(Rmin:float, Rmax:float, dc:float, z_index:int, CAMB_results:camb.CAMBdata, cold=False) -> float:
    from scipy.optimize import root_scalar
    Rnl_root = lambda R: _get_sigmaR(R, z_index, CAMB_results, cold=cold)-dc
    Rnl = root_scalar(Rnl_root, bracket=(Rmin, Rmax)).root
    return Rnl


def _get_effective_index(Rnl:float, R:np.ndarray, sigmaR:np.ndarray) -> float:
    neff = -3.-2.*util.derivative_from_samples(np.log(Rnl), np.log(R), np.log(sigmaR))
    return neff


def _get_halo_collapse_redshifts(M:np.ndarray, z:float, iz:int, dc:float, g:callable,
                                 CAMB_results:camb.CAMBdata, cold=False) -> np.ndarray:
    '''
    Calculate halo collapse redshifts according to the Bullock et al. (2001) prescription
    '''
    from scipy.optimize import root_scalar
    gamma = 0.01
    a = cosmology.scalefactor_from_redshift(z)
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c+Om_b+Om_nu
    zf = np.zeros_like(M)
    for iM, _M in enumerate(M):
        Mc = gamma*_M
        Rc = cosmology.Lagrangian_radius(Mc, Om_m)
        sigma = _get_sigmaR(Rc, iz, CAMB_results, cold=cold)
        fac = g(a)*dc/sigma
        if fac >= g(a):
            af = a # These haloes formed 'in the future'
        else:
            af_root = lambda af: g(af)-fac
            af = root_scalar(af_root, bracket=(1e-3, 1.)).root
        zf[iM] = cosmology.redshift_from_scalefactor(af)
    return zf