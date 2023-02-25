# Standard imports
import numpy as np
import scipy.integrate as integrate

# Project imports
from . import constants as const

# Constants
Dv0 = 18.*np.pi**2                  # Delta_v = ~178, EdS halo virial overdensity
dc0 = (3./20.)*(12.*np.pi)**(2./3.) # delta_c = ~1.686' EdS linear collapse threshold

# Parameters
xmin_Tk = 1e-5    # Scale at which to switch to Taylor expansion approximation in tophat Fourier functions
eps_sigmaV = 1e-3 # Accuracy of the sigmaV integration NOTE: Seems to fail with higher accuracy

### Backgroud ###

def redshift_from_scalefactor(a):
    return -1.+1./a


def scalefactor_from_redshift(z):
    return 1./(1.+z)


def comoving_matter_density(Om_m:float) -> float:
    '''
    Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
    args:
        Om_m: Cosmological matter density (at z=0)
    '''
    return const.rho_critical*Om_m

### ###

### Linear perturbations ###

def Tk_EH_nowiggle(k:np.ndarray, h:float, wm:float, wb:float, T_CMB=2.725) -> np.ndarray:
    '''
    No-wiggle transfer function from astro-ph:9709112
    '''
    # These only needs to be calculated once
    rb = wb/wm     # Baryon ratio
    e = np.exp(1.) # e
    s = 44.5*np.log(9.83/wm)/np.sqrt(1.+10.*wb**0.75)              # Equation (26)
    alpha = 1.-0.328*np.log(431.*wm)*rb+0.38*np.log(22.3*wm)*rb**2 # Equation (31)

    # Functions of k
    Gamma = (wm/h)*(alpha+(1.-alpha)/(1.+(0.43*k*s*h)**4)) # Equation (30)
    q = k*(T_CMB/2.7)**2/Gamma # Equation (28)
    L = np.log(2.*e+1.8*q)     # Equation (29)
    C = 14.2+731./(1.+62.5*q)  # Equation (29)
    Tk_nw = L/(L+C*q**2)       # Equation (29)
    return Tk_nw


def _Tophat_k(x:np.ndarray) -> np.ndarray:
    '''
    Fourier transform of a tophat function.
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))


def _dTophat_k(x:np.ndarray) -> np.ndarray:
    '''
    Derivative of the tophat Fourier transform function
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, -x/5.+x**3/70., (3./x**4)*((x**2-3.)*np.sin(x)+3.*x*np.cos(x)))


def _sigmaR_integrand(k:np.array, R:float, Pk:callable) -> np.ndarray:
    '''
    Integrand for calculating sigma(R)
    Note that k can be a float or an arraay here
    args:
        k: Fourier wavenumber (or array of these) [h/Mpc]
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    return Pk(k)*(k**2)*_Tophat_k(k*R)**2
 

def _sigmaR_quad(R:float, Pk:callable) -> float:
    '''
    Quad integration
    args:
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    def sigmaR_vec(R:float, Pk:callable):
        kmin, kmax = 0., np.inf
        sigma_squared, _ = integrate.quad(lambda k: _sigmaR_integrand(k, R, Pk), kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma
    sigma_func = np.vectorize(sigmaR_vec, excluded=['Pk'])
    return sigma_func(R, Pk)


def _sigmaV_integrand(k:float, Pk:callable):
    return Pk(k)
# def _sigmaV_integrand(t, Pk, alpha=3.):
#     k = (-1.+1./t)**alpha
#     return Pk(k)*k*alpha/(t*(1.-t))


def sigmaV(Pk:callable, eps=eps_sigmaV) -> float:
    '''
    Quad integration; R=0
    TODO: This generates a warning sometimes, there must be a cleverer way to integrate here.
    Unless eps_sigmaV > 1e-3 the integration fails for z=0 sometimes, but not after being called for
    z > 0. I really don't understaand this, but it's annoying and should be fixed.
    I should look at how CAMB deals with these type of integrals (e.g., sigmaR).
    args:
        Pk: Function of k to evaluate the linear power spectrum
        eps: Integration accuracy
    '''
    kmin, kmax = 0., np.inf
    #sigmaV_squared, _ = integrate.quad(Pk, kmin, kmax, epsrel=eps, epsabs=eps)
    sigmaV_squared, _ = integrate.quad(lambda k: _sigmaV_integrand(k, Pk),  kmin, kmax, epsrel=eps, epsabs=eps)
    #sigmaV_squared, _ = integrate.quad(_sigmaV_integrand, 0., 1., args=(Pk,), epsabs=eps, epsrel=eps)
    sigmaV = np.sqrt(sigmaV_squared/(2.*np.pi**2))
    sigmaV /= np.sqrt(3.) # Convert from 3D displacement to 1D displacement
    return sigmaV


def _dsigmaR_integrand(k:float, R:float, Pk:callable) -> float:
    return Pk(k)*(k**3)*_Tophat_k(k*R)*_dTophat_k(k*R)


def dlnsigma2_dlnR(R:float, Pk:callable) -> float:
    '''
    Calculates d(ln sigma^2)/d(ln R) by integration
    3+neff = -d(ln sigma^2) / dR
    '''
    # def dsigmaR_vec(R, Pk):
    #     kmin, kmax = 0., np.inf # Evaluate the integral and convert to a nicer form
    #     dsigma, _ = integrate.quad(lambda k: _dsigmaR_integrand(k, R, Pk), kmin, kmax)
    #     dsigma = R*dsigma/(np.pi*_sigmaR_quad(R, Pk))**2
    #     return dsigma
    # dsigma_func = np.vectorize(dsigmaR_vec, excluded=['Pk'])
    # return dsigma_func(R, Pk)
    kmin, kmax = 0., np.inf # Evaluate the integral and convert to a nicer form
    dsigma, _ = integrate.quad(lambda k: _dsigmaR_integrand(k, R, Pk), kmin, kmax)
    dsigma = R*dsigma/(np.pi*_sigmaR_quad(R, Pk))**2
    return dsigma


def neff(R:float, Pk:callable) -> float:
    '''
    Effective index of the power spectrum at scale 'R'
    '''
    return -3.-dlnsigma2_dlnR(R, Pk)

### ###

### Haloes ###

def Lagrangian_radius(M:float, Om_m:float) -> float:
    '''
    Radius [Mpc/h] of a sphere containing mass M in a homogeneous universe
    args:
        M: Halo mass [Msun/h]
        Om_m: Cosmological matter density (at z=0)
    '''
    return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))


def mass(R:float, Om_m:float) -> float:
    '''
    Mass [Msun/h] contained within a sphere of radius 'R' [Mpc/h] in a homogeneous universe
    '''
    return (4./3.)*np.pi*R**3*comoving_matter_density(Om_m)

### ###

### Spherical collapse ###

def dc_NakamuraSuto(Om_mz:float) -> float:
    '''
    LCDM fitting function for the critical linear collapse density from Nakamura & Suto
    (1997; https://arxiv.org/abs/astro-ph/9612074)
    Cosmology dependence is very weak
    '''
    return dc0*(1.+0.012299*np.log10(Om_mz))


def Dv_BryanNorman(Om_mz:float) -> float:
    '''
    LCDM fitting function for virial overdensity from Bryan & Norman
    (1998; https://arxiv.org/abs/astro-ph/9710107)
    Note that here Dv is defined relative to background matter density,
    whereas in paper it is relative to critical density
    For Omega_m = 0.3 LCDM Dv ~ 330.
    '''
    x = Om_mz-1.
    Dv = Dv0+82.*x-39.*x**2
    return Dv/Om_mz


def _f_Mead(x:float, y:float, p0:float, p1:float, p2:float, p3:float) -> float:
    return p0+p1*(1.-x)+p2*(1.-x)**2+p3*(1.-y)


def dc_Mead(a:float, Om_m:float, f_nu:float, g:float, G:float) -> float:
    '''
    delta_c fitting function from Mead (2017; 1606.05345)
    All input parameters should be evaluated as functions of a/z
    '''
    # See Appendix A of Mead (2017) for naming convention
    p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
    p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
    a1, _ = 1, 0
 
    # Linear collapse threshold
    dc_Mead = 1.
    dc_Mead = dc_Mead+_f_Mead(g/a, G/a, p10, p11, p12, p13)*np.log10(Om_m)**a1
    dc_Mead = dc_Mead+_f_Mead(g/a, G/a, p20, p21, p22, p23)
    dc_Mead = dc_Mead*dc0*(1.-0.041*f_nu)
    return dc_Mead


def Dv_Mead(a:float, Om_m:float, f_nu:float, g:float, G:float) -> float:
    '''
    Delta_v fitting function from Mead (2017; 1606.05345)
    All input parameters should be evaluated as functions of a/z
    '''
    # See Appendix A of Mead (2017) for naming convention
    p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
    p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
    a3, a4 = 1, 2

    # Halo virial overdensity
    Dv_Mead = 1.
    Dv_Mead = Dv_Mead+_f_Mead(g/a, G/a, p30, p31, p32, p33)*np.log10(Om_m)**a3
    Dv_Mead = Dv_Mead+_f_Mead(g/a, G/a, p40, p41, p42, p43)*np.log10(Om_m)**a4
    Dv_Mead = Dv_Mead*Dv0*(1.+0.763*f_nu)
    return Dv_Mead

### ###