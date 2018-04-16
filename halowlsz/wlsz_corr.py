import os
import numpy as np
from numba import jit
from scipy import special
from scipy import integrate
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from configparser import ConfigParser
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


def xi_wl_tsz(config_file='wlsz.ini', cosmology='battaglia', 
              rmin=1e-2, rmax=150, rbin=100, fwhm_k=1, fwhm_y=10, 
              kk=False, yy=False, ky=True, zsfile='source_distribution.txt',
              P01=None, P02=None, P03=None,
              xc1=None, xc2=None, xc3=None,
              beta1=None, beta2=None, beta3=None, 
              omega_m0=None, sigma_8=None, 
              odir='../data', oclfile='cltest.dat', oxifile='xitest.dat'):
    '''
    Given the radius array in arcmin it will return the halomodel
    '''
    config = ConfigParser()
    config.read(config_file)
    savefile = config.getboolean('halomodel', 'savefile')

    rarcmin = np.linspace(rmin, rmax, rbin) #arcmin
    rradian = rarcmin / 60. * np.pi / 180.
    ellarr, cl1h, cl2h, cl = cl_WL_tSZ(config_file, cosmology, fwhm_k, 
                                       fwhm_y, kk, yy, ky, zsfile, 
                                       P01, P02, P03, xc1, xc2, xc3,
                                       beta1, beta2, beta3, omega_m0,
                                       sigma_8, odir, oclfile)
    #pl.loglog(ellarr, cl, c='k', label='original')
    clspl = InterpolatedUnivariateSpline(ellarr, cl, k=3)
    cl1hspl = InterpolatedUnivariateSpline(ellarr, cl1h, k=3)
    cl2hspl = InterpolatedUnivariateSpline(ellarr, cl2h, k=3)

    ellarr = np.linspace(ellarr.min(), ellarr.max(), 10000)
    dl = ellarr[1] - ellarr[0]
    cl = clspl(ellarr)
    cl1h = cl1hspl(ellarr)
    cl2h = cl2hspl(ellarr)
    #pl.loglog(ellarr, cl, c='r', label='Spline')
    xi1h = np.array([integrate_ell(ellarr, cl1h, r, dl) for r in rradian])
    xi2h = np.array([integrate_ell(ellarr, cl2h, r, dl) for r in rradian])
    xi = np.array([integrate_ell(ellarr, cl, r, dl) for r in rradian])
    if savefile:
        np.savetxt(os.path.join(odir, oxifile), np.transpose((rarcmin, xi1h, xi2h, xi)), fmt='%.2f %.3e %.3e %.3e', header='theta_arcmin xi1h xi2h xi')

    return rarcmin, xi1h, xi2h, xi

if __name__=='__main__':
    ofile = 'xi_ky.dat'
    zsfile = 'source_distribution_new_z0p4.txt'
    rarcmin, xi1h, xi2h, xi = xi_wl_tsz(1e-2, 150, 100, 
                                        fwhm_k=1, fwhm_y=10., 
                                        kk=False, yy=False, ky=True,
                                        zsfile=zsfile, ofile=ofile)
    pl.plot(rarcmin, xi1h, label='1- halo model')
    pl.plot(rarcmin, xi2h, label='2- halo model')
    pl.plot(rarcmin, xi, label='Halo model')
    pl.xlabel('r (arcmin)')
    pl.ylabel(r'$\xi(r)_{y\kappa}$')
    pl.legend(loc=0)
    pl.savefig('../figs/xi.png', bbox_inches='tight')
    pl.show()


