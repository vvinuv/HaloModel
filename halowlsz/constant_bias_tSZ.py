import os
import sys
import config
import numpy as np
from numpy import vectorize
from scipy import interpolate, integrate
from scipy import special
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter
import pylab as pl
from numba import double, float64, float32
from numba import jit
import numba as nb
import timeit
#from mytools import constants
#import fastcorr
from CosmologyFunctions import CosmologyFunctions
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

@jit(nopython=True)
def integrate_rad(theta_radian, xi, ell, dlnt):
    cl = 0.0
    for ri, t in enumerate(theta_radian):
        cl += (t*t*np.sin(t*ell)*xi[ri]/t/ell)
    return cl*dlnt*2.*np.pi

@jit(nopython=True)
def integrate_ell(larr, cl, theta_rad, dlnl):
    xi = 0.0
    for i, l in enumerate(larr):
        xi += (l*l*np.sin(l*theta_rad)*cl[i]/l/theta_rad)
    return xi *dlnl / 2 /np.pi

@jit(nopython=True)
def integrate_splrad(theta_radian_spl, xi_spl, ell, dt):
    cl = 0.0
    for ri, t in enumerate(theta_radian_spl):
        cl += (t*np.sin(t*ell)*xi_spl[ri]/t/ell)
    return cl*dt*2.*np.pi

@jit(nopython=True)
def integrate_splell(larr_spl, cl_spl, theta_rad, dl):
    xi = 0.0
    for i, l in enumerate(larr_spl):
        xi += (l*np.sin(l*theta_rad)*cl_spl[i]/l/theta_rad)
    return xi *dl / 2 /np.pi


@jit(nopython=True)
def integrate_zdist(gangsarr, gchisarr, angl, Ns):
    #chi - in Eq. 2 of Waerbeke
    #fchi - chi in photometric redshift distribution 
    #angarr = np.linspace(chi, angH, 100)
    gint = 0.0
    i = 0
    for N in Ns:
        gint += ((gangsarr[i] - angl) * N / gangsarr[i]) 
        i += 1
    #gint *= (gangarr[1] - gangarr[0])
    gint *= (gchisarr[1] - gchisarr[0])
    return gint

@jit(nopython=True)
#def integrate_chi(zarr, chiarr, angarr, pkarr, Darr, constk, chiH, Ns):
def integrate_chi(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns):
    aarr = 1. / (1. + zlarr)
    Wk = constk * anglarr / aarr
    cl = 0.0
    for i, angl in enumerate(anglarr):
        gangsarr = angsarr[i+1:]
        gchisarr = chisarr[i+1:]
        gw = integrate_zdist(gangsarr, gchisarr, angl, Ns[i+1:])
        if gw <= 0:
            gw = 0.
        cl += (Wk[i] * Wk[i] *  gw * gw * pkarr[i] * Darr[i] * Darr[i] / angl / angl)
        #print Wk[i], gw, pkarr[i],Darr[i]
    cl *= (chilarr[1] - chilarr[0])
    #Wy = consty * bg * Te * ne
    return cl

if __name__=='__main__':
    #Write variables
    zmax_l = 1.0
    omega_m = 0.264
    h = 0.71
    H0 = h * 100 #(km/s)/Mpc
    #constants = constants()
    #c = constants.c_km_s
    c = 2.998e5 #km/s
    mpctocm = 3.085677581e24
    kB_kev_K = 8.617330e-8 #keV k^-1
    sigma_t_cm = 6.6524e-25 #cm^2
    rest_electron_kev = 511 #keV
    constk = 3. * omega_m * (H0 / c)**2. / 2. #Mpc^-2
    print constk
    consty = kB_kev_K  * sigma_t_cm / rest_electron_kev #cm^-2 
    const = constk * consty

    #Source redshift distribution
    f = np.genfromtxt('source_distribution.txt')
    zsarr = f[:,0][1:]
    Ns = f[:,1][1:]
    
    minz = zsarr.min()
    maxz = zsarr.max()
    zsspl = InterpolatedUnivariateSpline(zsarr, Ns, k=1)


    #These parameters will do nothing at this moment 
    compute = 1 #Whether the profile should be computed 
    fwhm = 10 #arcmin Doesn't work now
    rmin = 1 #Inner radius in arcmin 
    rmax = 1e2 #Outer radius in arcmin
    space = 10 #linear space between two points
    #Stop

    kmin = 1e-5
    kmax = 1e4
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
    karr = np.exp(lnkarr).astype(np.float64)

    chisarr, Darr = [], []
    for zi in zsarr:
        cosmo = CosmologyFunctions(zi)
        chisarr.append(cosmo.comoving_distance())
    chisarr = np.array(chisarr) / h #Mpc
    #pl.scatter(zarr, chiarr, c='k')

    chizspl = InterpolatedUnivariateSpline(chisarr, zsarr, k=1)
    chisarr = np.linspace(chisarr.min(), chisarr.max(), 1000)
    zsarr = chizspl(chisarr)
    conl = (zsarr > 0.0) & (zsarr < zmax_l)
    zlarr = zsarr[conl]
    chilarr = chisarr[conl]

    anglarr = chilarr / (1. + zlarr)
    angsarr = chisarr / (1. + zsarr)

    #pl.scatter(zarr, chiarr, c='r')
    #pl.show()

    #Interpolating source at individual redshift points
    Ns = zsspl(zsarr)

    for zi in zlarr:
        cosmo = CosmologyFunctions(zi)
        Darr.append(cosmo._growth)
    Darr = np.array(Darr)


	#pk_arr = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
        #pl.loglog(karr, pk_arr)
    #pl.show()
    #sys.exit() 
    #zint normalization of Eq 2 of Waerbeke
    ##zint = np.sum(Ns * (angarr[1] - angarr[0]))
    zint = np.sum((Ns[:-1] + Ns[1:]) * (chisarr[1] - chisarr[0])) / 2.
    Ns /= zint
    #pl.plot(zarr, Ns)
    #pl.yscale('log')
    #pl.show()

    cosmo = CosmologyFunctions(0.)

    #No little h
    pk_arr_z0 = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
    pkspl_z0 = InterpolatedUnivariateSpline(karr/cosmo._h, pk_arr_z0)

    fpk = np.genfromtxt('/media/luna1/vinu/software/FrankenEmu/output_pk/0_0.005.dat')
    karr, pk_arr_z0 = fpk[:,0], fpk[:,1]
    pkspl_z0 = InterpolatedUnivariateSpline(karr, pk_arr_z0)
    #pl.savetxt('data/pk_z0.dat', np.transpose((karr, pk_arr_z0)))
    #pl.loglog(karr, pk_arr_z0)
    #pl.show()
    #larr = np.linspace(1, 3000, 50)
    larr = np.arange(1, 4000, 10)
    larr = np.logspace(np.log10(1), np.log10(3000), 50)
    print larr 

    angH = angsarr.max()
    angL = angsarr.min()
    chiH = chisarr.max()
    chiL = chisarr.min()
 
    print angL, angH, const, zint
    cl_ky = [] 
    cl_kk = [] 
    for l in larr:
        #print l/angarr
        pkarr = pkspl_z0(l/anglarr)
        #pl.loglog(angarr, pkarr)
        #pl.show()
        #sys.exit()
        cl_kk.append(integrate_chi(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns))
    cl_kk = np.array(cl_kk)
    print cl_kk #* larr * (1+larr) / 2. / np.pi 
    np.savetxt('data/kk_power_consta_bias.txt', np.transpose((larr, cl_kk)))
    #np.savetxt('data/pk_%.1f.txt'%redshift, np.transpose((karr, pk_arr))) 
    pl.loglog(larr, cl_kk)
    pl.show()
    sys.exit()

