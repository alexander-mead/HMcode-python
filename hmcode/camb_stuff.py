# Third-party imports
import camb

def run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, 
    m_nu=0., w=-1., wa=0., log10_T_AGN=None,
    As=2e-9, norm_sigma8=True, kmax_CAMB=200., verbose=False):

    # Sets cosmological parameters in camb to calculate the linear power spectrum
    # TODO: Setting 'WantCls=False' in CAMBparams spoils the agreement... why?
    if log10_T_AGN:
        non_linear_model = camb.nonlinear.Halofit(halofit_version="mead2020_feedback", 
                                                  HMCode_logT_AGN=log10_T_AGN)
    else:
        non_linear_model = camb.nonlinear.Halofit(halofit_version="mead2020")
    pars = camb.CAMBparams(NonLinearModel=non_linear_model) 
    wb, wc = Omega_b*h**2, Omega_c*h**2

    # This function sets standard and helium set using BBN consistency
    pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
    pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf')
    pars.InitPower.set_params(As=As, ns=ns, r=0.)
    pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Linear matter power spectrum
    Omega_m = pars.omegam # Extract the matter density

    # Scale 'As' to be correct for the desired 'sigma_8' value if necessary
    if norm_sigma8:
        results = camb.get_results(pars)
        sigma_8_init = results.get_sigma8_0()
        if verbose:
            print('Running CAMB')
            print('Initial sigma_8:', sigma_8_init)
            print('Desired sigma_8:', sigma_8)
        scaling = (sigma_8/sigma_8_init)**2
        As *= scaling
        pars.InitPower.set_params(As=As, ns=ns, r=0.)

    # Run
    results = camb.get_results(pars)
    Pk = results.get_matter_power_interpolator(nonlinear=False).P
    sigma_8 = results.get_sigma8_0()
    if verbose:
        print('Final sigma_8:', sigma_8)
        print()

    # Return
    return Pk, results, Omega_m, sigma_8, As