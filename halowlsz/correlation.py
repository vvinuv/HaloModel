import numpy as np
from numba import jit
from scipy import special
from scipy import integrate
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
import config
from halomodel_cl_WL_tSZ import cl_WL_tSZ
import pylab as pl

@jit(nopython=True)
def integrate_rad(theta_radian, xi, ell, dt):
    cl = 0.0
    for ri, t in enumerate(theta_radian):
        cl += (t*np.sin(t*ell)*xi[ri]/t/ell)
    return cl*dt*2.*np.pi

@jit(nopython=True)
def integrate_ell(larr, cl, theta_radian, dl):
    '''
    Eq. 105 fof Schneider book
    '''
    xi = 0.0
    for i, l in enumerate(larr):
        xi += (l*np.sin(l*theta_radian)*cl[i]/l/theta_radian)
    return xi * dl / 2. /np.pi


def correlation_integral(ell, cl_ky, theta):
    '''
    Eq. 105 fof Schneider book
    '''
    return ell * special.j0(ell * theta) * cl_ky
    

def correlation():
    '''
    Eq. 105 fof Schneider book
    '''
    xi = integrate.quad(correlation_integral, 0, 1e4)

    return xi 


def xi_wl_tsz(rarcmin):
    '''
    Given the radius array in arcmin it will return the halomodel
    '''
    rradian = rarcmin / 60. * np.pi / 180.
    ellarr, cl1h, cl2h, cl = cl_WL_tSZ(config.fwhm, config.kk, config.yy, config.ky, config.zsfile)
    #pl.loglog(ellarr, cl, c='k', label='original')
    clspl = InterpolatedUnivariateSpline(ellarr, cl, k=3)

    ellarr = np.linspace(ellarr.min(), ellarr.max(), 10000)
    dl = ellarr[1] - ellarr[0]
    cl = clspl(ellarr)
    #pl.loglog(ellarr, cl, c='r', label='Spline')
    xi = np.array([integrate_ell(ellarr, cl, r, dl) for r in rradian])
    return xi

if __name__=='__main__':
    rarcmin = np.linspace(1e-2, 100, 100) #arcmin
    xi = xi_wl_tsz(rarcmin)
    pl.plot(rarcmin, xi)
    pl.xlabel('r (arcmin)')
    pl.ylabel(r'$\xi(r)_{y\kappa}$')
    pl.savefig('figs/xi.png', bbox_inches='tight')
    pl.show()


