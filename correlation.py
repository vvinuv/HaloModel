import numpy as np
from scipy import special
import scipy import integrate


def correlation_integral(ell, cl_ky, theta):
    '''
    Eq. 105 fof Schneider book
    '''
    return ell * special.j0(ell * theta) * cl_ky
    

def correlation():
    '''
    Eq. 105 fof Schneider book
    '''
    xi = integrate.quad(correlation_integral, ell=0, ell=1e4)

    return xi 
