# Standard imports
from time import time

# Third-party imports
import numpy as np
import camb
import pyhalomodel as halo

# Project imports
from . import utility as util
from . import cosmology
from . import linear_growth

# Parameters
Pk_lin_extrap_kmax = 1e10 # NOTE: This interplays with the sigmaV integration in a disconcerting way
sigma_cold_approx = False # Should the Eisenstein & Hu (1999) approximation be used for the cold transfer function?

def power(k:np.array, zs:np.array, CAMB_results:camb.CAMBdata, T_AGN=None,
          Mmin=1e0, Mmax=1e18, nM=256, tweaks=True, verbose=False) -> np.ndarray:
    '''
    Calculates the HMcode matter-matter power spectrum
    Args:
        k: Array of comoving wavenumbers [h/Mpc]
        zs: Array of redshifts (ordered from high to low)
        CAMB_results: CAMBdata structure
        T_AGN: AGN feedback temperature [K] (None to disable)
        Mmin: Minimum mass for the halo-model calculation [Msun/h]
        Mmax: Maximum mass for the halo-model calculation [Msun/h]
        nM: Number of mass bins for the halo-model calculation
        tweaks: Use the changes to the vanilla halo model from HMCode2020 if true
    Returns:
        Array of matter power spectra: Pk[z, k]
    '''

    # Checks
    if verbose: t_start = time()
    if not util.is_array_monotonic(-np.array(zs)):
        raise ValueError('Redshifts must be monotonically decreasing')

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

    # Baryonic feedback
    if T_AGN:
        feedback_params = _get_feedback_parameters(T_AGN)
        if verbose:
            print('Using baryonic feedback model from HMCode2020')
            print('log_10(T_AGN/K): {:.2f}'.format(np.log10(T_AGN)))
            print('B_0: {:.4f}'.format(feedback_params['B0']))
            print('B_z: {:.4f}'.format(feedback_params['Bz']))
            print('Mb_0: {:.1e}'.format(feedback_params['Mb0']))
            print('Mb_z: {:.4f}'.format(feedback_params['Mbz']))
            print('f*_0: {:.4f}'.format(feedback_params['f0']))
            print('f*_z: {:.4f}'.format(feedback_params['fz']))
            print()

    # Linear power interpolator
    interp, _, k_interp = CAMB_results.get_matter_power_interpolator(nonlinear=False, 
                                                                     return_z_k=True,
                                                                     extrap_kmax=Pk_lin_extrap_kmax)
    Pk_lin_interp = interp.P
    kmin = k_interp[0] # Minimum wavenumber used for the CAMB interpolator [h/Mpc]

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
        Pk_lin = Pk_lin_interp(z, k)  # Linear power spectrum
        R = hmod.Lagrangian_radius(M) # Lagrangian radii
        if sigma_cold_approx: # Variance in cold matter field
            sigmaM = _get_sigmaR_approx(R, g, lambda k: Pk_lin_interp(z, k), CAMB_results, cold=True)
        else:
            sigmaM = _get_sigmaR(R, iz, CAMB_results, cold=True) 
        nu = hmod._peak_height(M, sigmaM) # Halo peak height
        if verbose:
            print('Lagrangian radius range: {:.4} -> {:.4} Mpc/h'.format(R[0], R[-1]))
            print('RMS in matter field range: {:.4} -> {:.4}'.format(sigmaM[0], sigmaM[-1]))
            print('Peak height range: {:.4} -> {:.4}'.format(nu[0], nu[-1]))

        # Parameters of the linear spectrum pertaining to non-linear growth
        # TODO: I think the sigmaV integral is quite time-consuming for some reason
        Rnl = _get_nonlinear_radius(R[0], R[-1], dc, iz, CAMB_results, cold=True) # Non-linear Lagrangian radius
        sigma8 = _get_sigmaR(8., iz, CAMB_results, cold=True)                     # RMS in the linear cold matter field at 8 Mpc/h
        sigmaV = cosmology.sigmaV(0., lambda k: Pk_lin_interp(z, k), kmin=kmin)   # RMS in the linear displacement field
        neff = _get_effective_index(Rnl, R, sigmaM)                               # Effective index of spectrum at collapse scale
        if verbose:
            print('Non-linear Lagrangian radius: {:.4} Mpc/h'.format(Rnl))
            print('RMS in matter field at 8 Mpc/h: {:.4}'.format(sigma8))
            print('RMS in matter displacement field: {:.4} Mpc/h'.format(sigmaV))
            print('Effective index at collapse scale: {:.4}'.format(neff))

        # HMcode parameters (Table 2 of https://arxiv.org/pdf/2009.01858.pdf)
        ks = 0.05618*sigma8**-1.013  # One-halo damping wavenumber; equation (17)
        if tweaks: 
            kd = 0.05699*sigma8**-1.089  # Two-halo damping wavenumber; equation (16)
            f = 0.2696*sigma8**0.9403    # Two-halo fractional damping; equation (16)
            nd = 2.853                   # Two-halo damping power; equation (16)
            ks = 0.05618*sigma8**-1.013  # One-halo damping wavenumber; equation (17)
            eta = 0.1281*sigma8**-0.3644 # Halo bloating parameter; equation (19)
            B = 5.196                  # Minimum halo concentration; equation (20)
            alpha = 1.875*(1.603)**neff  # Transition smoothing; equation (23)
        else: # Use vanilla-ish halo model if no HMcode tweaks used (still uses one-halo term suppression)
            eta = 0.
            B = 4.

        if T_AGN and not tweaks:
            B = feedback_params['B0']*np.power(10, z*feedback_params['Bz'])
            Mb = feedback_params['Mb0']*np.power(10, z*feedback_params['Mbz'])
            fstar = feedback_params['f0']*np.power(10, z*feedback_params['fz'])

        if verbose:
            if tweaks:
                print('Two-halo damping wavenumber; kd: {:.4} h/Mpc'.format(kd))
                print('Two-halo fractional damping; f: {:.4}'.format(f))
                print('Two-halo damping power; nd: {:.4}'.format(nd))
                print('One-halo damping wavenumber; k*: {:.4} h/Mpc'.format(ks))
                print('Halo bloating; eta: {:.4}'.format(eta))
                print('Minimum halo concentration; B: {:.4}'.format(B))#, B/4.)
                print('Transition smoothing; alpha: {:.4}'.format(alpha))
            if T_AGN and not tweaks:
                print('Gas-loss halo-mass parameter Mb: {:1e}'.format(Mb))
                print('Effective halo stellar-mass fraction f*: {:.4f}'.format(f))
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
            if T_AGN and not tweaks:
                Uk[:, iM] = _win_NFW_baryons(k*(_nu**eta), _rv, _c, M[iM], Mb, fstar, Om_m, Om_c, Om_b)[:, 0]
            else:
                Uk[:, iM] = _win_NFW(k*(_nu**eta), _rv, _c)[:, 0]
        if T_AGN and not tweaks: # NOTE: No Factor of 1-f_nu for baryonic, because this effect is already included!
            profile = halo.profile.Fourier(k, M, Uk, amplitude=M/hmod.rhom, mass_tracer=True) 
        else: # NOTE: Factor of 1-f_nu in profile amplitude
            profile = halo.profile.Fourier(k, M, Uk, amplitude=M*(1.-f_nu)/hmod.rhom, mass_tracer=True) 

        # Vanilla power spectrum calculation
        _, _Pk_1h, _ = hmod.power_spectrum(k, Pk_lin, M, sigmaM, {'m': profile}, simple_twohalo=True)

        # HMcode tweaks
        # Still uses one-halo term dampening even if tweaks=False
        Pk_1h = (k/ks)**4/(1.+(k/ks)**4)*_Pk_1h['m-m'] # One-halo term; equation (17)
        if tweaks:
            Pk_wig = _get_Pk_wiggle(k, Pk_lin, CAMB_results)   # Isolate spectral wiggle; footnote 7
            Pk_dwl = Pk_lin-(1.-np.exp(-(k*sigmaV)**2))*Pk_wig # Constuct linear spectrum with smoothed wiggle; equation (15)
            Pk_2h = Pk_dwl*(1.-f*(k/kd)**nd/(1.+(k/kd)**nd))   # Two-halo term; equation (16)
            Pk_hm = (Pk_2h**alpha+Pk_1h**alpha)**(1./alpha)    # Total prediction via smoothed sum; equation (23)
        else: 
            Pk_hm = Pk_lin+Pk_1h
        Pk_HMcode[iz, :] = Pk_hm

    if T_AGN and tweaks:
        suppression = _get_feedback_suppression(k, zs, CAMB_results, T_AGN, Mmin=Mmin, Mmax=Mmax, nM=nM, verbose=False)
        Pk_HMcode *= suppression

    # Finish
    if verbose:
        t_finish = time()
        print('HMcode predictions complete for {:} redshifts'.format(len(zs)))
        print('Total HMcode run time: {:.3f}s'.format(t_finish-t_start))
        print()
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


def _get_sigmaR_approx(R, g, Pk_lin_interp, CAMB_results:camb.CAMBdata, cold=False):
    if cold:
        wm = CAMB_results.Params.omch2+CAMB_results.Params.ombh2+CAMB_results.Params.omnuh2
        h = CAMB_results.Params.h
        f_nu = CAMB_results.Params.omnuh2/wm
        N_nu = CAMB_results.Params.num_nu_massive
        T_CMB = CAMB_results.Params.TCMB
        Pk_interp = lambda k: Pk_lin_interp(k)*cosmology.Tk_cold_ratio(k, g, wm, h, f_nu, N_nu, T_CMB)**2
    else:
        Pk_interp = Pk_lin_interp
    sigmaR = cosmology.sigmaR(R, Pk_interp, transform_integrand=False)
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


def _win_NFW(k:np.ndarray, rv:np.ndarray, c:np.ndarray) -> np.ndarray:
    '''
    Normalised Fourier transform for an NFW profile
    '''
    from scipy.special import sici
    rs = rv/c
    kv = np.outer(k, rv)
    ks = np.outer(k, rs)
    Sisv, Cisv = sici(ks+kv)
    Sis, Cis = sici(ks)
    f1 = np.cos(ks)*(Cisv-Cis)
    f2 = np.sin(ks)*(Sisv-Sis)
    f3 = np.sin(kv)/(ks+kv)
    f4 = np.log(1.+c)-c/(1.+c)
    Wk = (f1+f2-f3)/f4
    return Wk


def _win_NFW_baryons(k:np.ndarray, rv:np.ndarray, c:np.ndarray, 
                     M:np.ndarray, Mb:float, fstar:float, 
                     Om_m:float, Om_c:float, Om_b:float) -> np.ndarray:
    '''
    Normalised Fourier transform for NFW profile, including baryonic effects
    Equation (25) from Mead et al. (2021)
    Calls _win_NFW
    '''
    Wk = _win_NFW(k, rv, c)
    fg = (Om_b/Om_m-fstar)*(M/Mb)**2/(1.+(M/Mb)**2) # Gas content (Eq. 24 with beta=2)
    Wk = (Om_c/Om_m+fg)*Wk
    Wk += fstar
    return Wk


def _get_feedback_parameters(T_AGN:float) -> dict:
    '''
    Maps one-Param baryon feedback model from HMCode2020 to 6 baryonic parameters
    Uses parameters from Table 5 in Mead et al. (2021)
    This fit was obtained using the vanilla halo model! 
    If the hmcode tweaks are used, different values are likely needed.
    '''
    theta = np.log10(T_AGN/np.power(10, 7.8))
    params = {
        'B0': 3.44-0.496*theta,
        'Bz': -0.0671-0.0371*theta,
        'Mb0': np.power(10, 13.87+1.81*theta), # [Msun/h]
        'Mbz': -0.108+0.195*theta,
        'f0': (2.01-0.3*theta)*1e-2,
        'fz': 0.409+0.0224*theta,
    }
    return params


def _get_feedback_suppression(k:np.array, zs:np.array, CAMB_results:camb.CAMBdata, T_AGN:float, 
                              Mmin=1e0, Mmax=1e18, nM=256, verbose=False) -> np.ndarray:
    '''
    Calculates the ratio of the powerspectrum with baryonic effects to that of dark-matter-only
    Assumes the one-parameter T_AGN model from HMCode2020
    Warning: Since the fit for the baryonic effects was obtained with the vanilla halo model, 
    it is not safe to set tweaks=True below
    '''
    Pk_gravity = power(k, zs, CAMB_results, T_AGN=None, Mmin=Mmin, Mmax=Mmax, nM=nM, 
                       tweaks=False, verbose=verbose)
    Pk_feedback = power(k, zs, CAMB_results, T_AGN=T_AGN, Mmin=Mmin, Mmax=Mmax, nM=nM, 
                        tweaks=False, verbose=verbose)
    return Pk_feedback/Pk_gravity