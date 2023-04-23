# Third-party imports
import numpy as np
from camb import CAMBdata

# Parameters
a_init = 1e-4 # Initial scale-factor for growth ODE integration

def _w(a:float, CAMB_results:CAMBdata, LCDM=False) -> float:
    '''
    Dark energy equation of state for w0, wa models
    '''
    w0, wa = (-1., 0.) if LCDM else (CAMB_results.Params.DarkEnergy.w, CAMB_results.Params.DarkEnergy.wa)
    return w0+(1.-a)*wa


def _X_w(a:float, CAMB_results:CAMBdata, LCDM=False) -> float:
    '''
    Cosmological dark energy density for w0, wa models
    '''
    w0, wa = (-1., 0.) if LCDM else (CAMB_results.Params.DarkEnergy.w, CAMB_results.Params.DarkEnergy.wa)
    return a**(-3.*(1.+w0+wa))*np.exp(-3.*wa*(1.-a))


def _Omega_m(a:float, CAMB_results:CAMBdata, LCDM=False) -> float:
    '''
    Evolution of Omgea_m with scale-factor ignoring radiation
    '''
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c+Om_b+Om_nu
    return Om_m*a**-3/_Hubble2(a, CAMB_results, LCDM)


def _Hubble2(a:float, CAMB_results:CAMBdata, LCDM=False) -> float:
    '''
    Squared Hubble parameter ignoring radiation
    Massive neutrinos are counted as 'matter'
    '''
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c+Om_b+Om_nu
    Om_w = 1.-Om_m if LCDM else CAMB_results.get_Omega(var='de', z=0.)
    Om = 1. if LCDM else 1.-CAMB_results.get_Omega(var='K', z=0.)
    H2 = Om_m*a**-3+Om_w*_X_w(a, CAMB_results, LCDM)+(1.-Om)*a**-2
    return H2


def _AH(a:float, CAMB_results:CAMBdata, LCDM=False) -> float:
    '''
    Acceleration parameter ignoring radiation
    Massive neutrinos are counted as 'matter'
    '''
    Om_c = CAMB_results.get_Omega(var='cdm', z=0.)
    Om_b = CAMB_results.get_Omega(var='baryon', z=0.)
    Om_nu = CAMB_results.get_Omega(var='nu', z=0.)
    Om_m = Om_c+Om_b+Om_nu
    Om_w = 1.-Om_m if LCDM else CAMB_results.get_Omega(var='de', z=0.)
    AH = -0.5*(Om_m*a**-3+(1.+3.*_w(a, CAMB_results, LCDM))*Om_w*_X_w(a, CAMB_results, LCDM))
    return AH


def get_growth_interpolator(CAMB_results:CAMBdata, LCDM=False) -> callable:
    '''
    Solve the linear growth ODE and returns an interpolating function for the solution
    LCDM = True forces w = -1 and imposes flatness by modifying the dark-energy density
    TODO: w dependence for initial conditions; f here is correct for w=0 only
    TODO: Could use d_init = a(1+(w-1)/(w(6w-5))*(Om_w/Om_m)*a**-3w) at early times with w = w(a<<1)
    '''
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d as interp
    na = 129 # Number of scale factors used to construct interpolator
    a = np.linspace(a_init, 1., na)
    f = 1.-_Omega_m(a_init, CAMB_results, LCDM=LCDM) # Early mass density
    d_init = a_init**(1.-3.*f/5.)            # Initial condition (~ a_init; but f factor accounts for EDE-ish)
    v_init = (1.-3.*f/5.)*a_init**(-3.*f/5.) # Initial condition (~ 1; but f factor accounts for EDE-ish)
    y0 = (d_init, v_init)
    def fun(a, y):
        d, v = y[0], y[1]
        dxda = v
        fv = -(2.+_AH(a, CAMB_results, LCDM=LCDM)/_Hubble2(a, CAMB_results, LCDM=LCDM))*v/a
        fd = 1.5*_Omega_m(a, CAMB_results, LCDM=LCDM)*d/a**2
        dvda = fv+fd
        return dxda, dvda
    g = solve_ivp(fun, (a[0], a[-1]), y0, t_eval=a).y[0]
    g_interp = interp(a, g, kind='cubic', assume_sorted=True)
    return g_interp


def get_accumulated_growth(a:float, g:callable) -> float:
    '''
    Calculates the accumulated growth at scale factor 'a'
    '''
    from scipy.integrate import quad
    missing = g(a_init) # Integeral from 0 to ai of g(ai)/ai ~ g(ai) for ai << 1
    G, _ = quad(lambda a: g(a)/a, a_init, a)+missing
    return G