# Third-party imports
import numpy as np

def derivative_from_samples(x:float, xs:np.array, fs:np.array) -> float:
    '''
    Calculates the derivative of the function f(x) which is sampled as fs at values xs
    Approximates the function as quadratic using the samples and Legendre polynomials
    Args:
        x: Point at which to calculate the derivative
        xs: Sample locations
        fs: Value of function at sample locations
    '''
    from scipy.interpolate import lagrange
    ix, _ = find_closest_index_value(x, xs)
    if ix == 0:
        imin, imax = (0, 1) if x < xs[0] else (0, 2) # Tuple braces seem to be necessarry here
    elif ix == len(xs)-1:
        nx = len(xs)
        imin, imax = (nx-2, nx-1) if x > xs[-1] else (nx-3, nx-1) # Tuple braces seem to be necessarry here
    else:
        imin, imax = ix-1, ix+1
    poly = lagrange(xs[imin:imax+1], fs[imin:imax+1])
    return poly.deriv()(x)


def logspace(xmin:float, xmax:float, nx:int) -> np.ndarray:
    '''
    Return a logarithmically spaced range of numbers
    '''
    return np.logspace(np.log10(xmin), np.log10(xmax), nx)


def find_closest_index_value(x:float, xs:np.array) -> tuple:
    '''
    Find the index, value pair of the closest values in array 'arr' to value 'x'
    '''
    idx = (np.abs(xs-x)).argmin()
    return idx, xs[idx]


def is_array_monotonic(x:np.array) -> bool:
    '''
    Returns True iff the array contains monotonically increasing values
    '''
    return np.all(np.diff(x) > 0.)


def is_array_linear(x:np.array, atol=1e-8) -> bool:
    '''
    Returns True iff the array is linearly spaced
    '''
    return np.isclose(np.all(np.diff(x)-np.diff(x)[0]), 0., atol=atol)