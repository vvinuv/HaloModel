import os
import numpy as np
from numba import jit
from scipy import special
from scipy import integrate
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from halomodel_cl_WL_tSZ import cl_WL_tSZ
import pylab as pl

import configparser
config = configparser.ConfigParser()

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


def xi_wl_tsz(rmin=1e-2, rmax=150, rbin=100, fwhm_k=1, fwhm_y=10, 
              kk=False, yy=False, ky=True, rarcmin=None, 
              zsfile='source_distribution.txt', omega_m0=0.25, sigma_8=0.8, 
              P01=18.1, P02=0.154, P03=-0.758, xc1=0.497, xc2=-0.00865, 
              xc3=0.731, beta1=4.35, beta2=0.0393, beta3=0.415, 
              default_pp=False, doPrintCl=True,
              paramsfile='wlxtsz.ini', odir='../data', ofile='test.dat'):
    '''
    Given the radius array in arcmin it will return the halomodel
    '''
    config.read(paramsfile)
    if rarcmin is None:
        rarcmin = np.linspace(rmin, rmax, rbin) #arcmin
    rradian = rarcmin / 60. * np.pi / 180.
    ellarr, cl1h, cl2h, cl = cl_WL_tSZ(paramsfile, fwhm_k, fwhm_y, kk, 
                                       yy, ky, zsfile, 
                                       omega_m0=omega_m0, sigma_8=sigma_8, 
                                       P01=P01, P02=P02, P03=P03, 
                                       xc1=xc1, xc2=xc2, xc3=xc3, 
                                       beta1=beta1, beta2=beta2, beta3=beta3, 
                                       default_pp=default_pp, odir=odir,
                                       doPrintCl=doPrintCl)
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
    if config['haloparams']['savefile']:
        if ky:
            np.savetxt(os.path.join(odir, ofile), np.transpose((rarcmin, xi1h, xi2h, xi)), fmt='%.2f %.3e %.3e %.3e', header='theta_arcmin xi1h xi2h xi')
        if kk:
            np.savetxt(os.path.join(odir, ofile), np.transpose((rarcmin, xi1h, xi2h, xi)), fmt='%.2f %.3e %.3e %.3e', header='theta_arcmin xi1h xi2h xi')
        if yy:
            np.savetxt(os.path.join(odir, ofile), np.transpose((rarcmin, xi1h, xi2h, xi)), fmt='%.2f %.3e %.3e %.3e', header='theta_arcmin xi1h xi2h xi')

    return rarcmin, xi1h, xi2h, xi

if __name__=='__main__':
    ofile = 'xi_ky.dat'
    zsfile = 'source_distribution_new_z0p4.txt'
    rarcmin, xi1h, xi2h, xi = xi_wl_tsz(rmin=1e-2, rmax=150, rbin=100, 
                                        fwhm_k=1, fwhm_y=10,
                                        kk=False, yy=False, ky=True,
                                        zsfile='source_distribution.txt', 
                                        omega_m0=0.25, sigma_8=0.8,
                                        P01=18.1, P02=0.154, P03=-0.758, 
                                        xc1=0.497, xc2=-0.00865,
                                        xc3=0.731, beta1=4.35, beta2=0.0393, 
                                        beta3=0.415, default_pp=False,
                                        paramsfile='wlxtsz.ini', 
                                        odir='../data', ofile=ofile, 
                                        doPrintCl=True)
    pl.plot(rarcmin, xi1h, label='1- halo model')
    pl.plot(rarcmin, xi2h, label='2- halo model')
    pl.plot(rarcmin, xi, label='Halo model')
    pl.xlabel('r (arcmin)')
    pl.ylabel(r'$\xi(r)_{y\kappa}$')
    pl.legend(loc=0)
    pl.savefig('../figs/xi.png', bbox_inches='tight')
    pl.show()


