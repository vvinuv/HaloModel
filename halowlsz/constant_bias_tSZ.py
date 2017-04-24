import os
import sys
import config
import numpy as np
from numpy import vectorize
from scipy import interpolate, integrate
from scipy import special
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter
import pylab as pl
from numba import double, float64, float32
from numba import jit
import numba as nb
import timeit
import mytools
#import fastcorr
from CosmologyFunctions import CosmologyFunctions
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

@jit(nopython=True)
def integrate_lnrad(theta_radian, xi, ell, dlnt):
    cl = 0.0
    for ri, t in enumerate(theta_radian):
        cl += (t*t*np.sin(t*ell)*xi[ri]/t/ell)
    return cl*dlnt*2.*np.pi

@jit(nopython=True)
def integrate_lnell(larr, cl, theta_rad, dlnl):
    xi = 0.0
    for i, l in enumerate(larr):
        xi += (l*l*np.sin(l*theta_rad)*cl[i]/l/theta_rad)
    return xi *dlnl / 2 /np.pi

@jit(nopython=True)
def integrate_rad(theta_radian, xi, ell, dt):
    cl = 0.0
    for ri, t in enumerate(theta_radian):
        cl += (t*np.sin(t*ell)*xi[ri]/t/ell)
    return cl*dt*2.*np.pi

@jit(nopython=True)
def integrate_ell(larr, cl, theta_rad, dl):
    xi = 0.0
    for i, l in enumerate(larr):
        xi += (l*np.sin(l*theta_rad)*cl[i]/l/theta_rad)
    return xi * dl / 2. /np.pi


@jit(nopython=True)
def integrate_zdist(gangsarr, gchisarr, angl, Ns):
    '''This uses exactly Eq. 2 of Waerbeke'''
    gint = 0.0
    for i, N in enumerate(Ns):
        gint += ((gangsarr[i] - angl) * N / gangsarr[i]) 
    gint *= (gchisarr[1] - gchisarr[0])
    return gint

@jit(nopython=True)
def integrate_kk(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns):
    '''
    Uses Eq. 4 of Waerbeke 2014 and I think the power spectrum is non-linear
    '''
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

@jit(nopython=True)
def integrate_ky(bgTene, zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, consty, chiH, Ns):
    '''
    Uses Eq. 4 of Waerbeke 2014 and I think the power spectrum is non-linear
    '''
    aarr = 1. / (1. + zlarr)
    Wk = constk * anglarr / aarr
    #Wsz = (kB / m_e /c^2) ne * sigma_T * Te * b_gas
    #Wsz = 0.00196 * ne * sigma_T * Te * b_gas 
    #mpctocm = 3.085677581e24
    #mpctom = 3.085677581e22
    #bgTene = 6 * 1e7 * 0.25 * 3.085677581e22
    #bgTene = 1.5 * 3.085677581e29
    #(m^2/K) * K / m^3 = 1/m = 3.085e22/Mpc
    bgTene *= 3.085677581e29
    Wsz = consty * bgTene * aarr 
    cl = 0.0
    for i, angl in enumerate(anglarr):
        gangsarr = angsarr[i+1:]
        gchisarr = chisarr[i+1:]
        gw = integrate_zdist(gangsarr, gchisarr, angl, Ns[i+1:])
        if gw <= 0:
            gw = 0.
        cl += (Wk[i] * Wsz[i] *  gw * gw * pkarr[i] * Darr[i] * Darr[i] / angl / angl)
        #print Wk[i], gw, pkarr[i],Darr[i]
    cl *= (chilarr[1] - chilarr[0])
    #Wy = consty * bg * Te * ne
    return cl


if __name__=='__main__':
    #Write variables
    maxzl = 1.0
    omega_m = 0.264
    h = 0.70
    H0 = h * 100 #(km/s)/Mpc

    #Source redshift distribution
    fsource = 'source_distribution_new_z0p4.txt'

    #Data 
    #dfile = '/media/luna1/vinu/Lensing/DES/Kappa_map/SVA1/des_sz/kappa_y_y1_im3shape_milca_0_0.40.npz'
    dfile = '/media/luna1/vinu/Lensing/DES/Kappa_map/SVA1/des_sz/kappa_y_y1_mcal_milca_0_11.00.npz'

    fwhm_k = 1 #arcmin for kappa
    fwhm_y = 10 #arcmin for sz map
    rmin = 1 #Inner radius in arcmin 
    rmax = 1e2 #Outer radius in arcmin
    space = 10 #linear space between two points
    #Stop

    sigma_k = fwhm_k * np.pi / 2.355 / 60. /180
    sigma_y = fwhm_y * np.pi / 2.355 / 60. /180
    sigmasq = sigma_k * sigma_y 

    rarr = np.arange(1, 100, 2)
    rradian = rarr / 60. * np.pi / 180. 
    mlarr = np.linspace(1, 10000, 1000) #larr for interpolate
    dl = mlarr[1] - mlarr[0]


    light_speed = 2.998e5 #km/s
    kB_kev_K = 8.617330e-8 #keV/K
    sigma_t_m = 6.6524e-29 #m^2
    rest_electron_kev = 511 #keV
    constk = 3. * omega_m * (H0 / light_speed)**2. / 2. #Mpc^-2
    consty = kB_kev_K  * sigma_t_m / rest_electron_kev #m^2/K 
    const = constk * consty
    #print 'constk=%.2e consty=%.2e'%(constk, consty)

    f = np.genfromtxt(fsource)
    zsarr = f[:,0][1:]
    Ns = f[:,1][1:]
    minzs = zsarr.min()
    maxzs = zsarr.max()
    zsspl = interp1d(zsarr, Ns, kind='slinear')
 
    zlarr = np.arange(0, maxzs, 0.05)    

    kmin = 1e-5
    kmax = 1e4
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
    karr = np.exp(lnkarr).astype(np.float64)

    chilarr, Darr = [], []
    for zi in zlarr:
        cosmo = CosmologyFunctions(zi)
        chilarr.append(cosmo.comoving_distance())
        Darr.append(cosmo._growth)

    chilarr = np.array(chilarr) / h #Mpc
    Darr = np.array(Darr)
    #pl.scatter(zsarr, chiarr, c='k')
    #pl.scatter(zsarr, Darr, c='r')
    chizspl = interp1d(chilarr, zlarr, kind='slinear')
    zDarrspl = interp1d(zlarr, Darr, kind='slinear')
    chilarr = np.linspace(chilarr.min(), chilarr.max(), 1000)
    zlarr = chizspl(chilarr)
    Darr = zDarrspl(zlarr)
    #pl.plot(zsarr, Darr, c='g')
    #pl.show()
    cons = (zlarr > minzs) & (zlarr < maxzs)
    zsarr = zlarr[cons]
    chisarr = chilarr[cons]

    conl = (zlarr > 0) & (zlarr < maxzl)
    zlarr = zlarr[conl]
    chilarr = chilarr[conl]
    Darr = Darr[conl]

    anglarr = chilarr / (1. + zlarr)
    angsarr = chisarr / (1. + zsarr)

    #pl.scatter(zarr, chiarr, c='r')
    #pl.show()

    #Interpolating source at individual redshift points
    Ns = zsspl(zsarr)

    #zint normalization of Eq 2 of Waerbeke
    zint = np.sum((Ns[:-1] + Ns[1:]) * (chisarr[1] - chisarr[0])) / 2.
    print 'zint ', zint
    Ns /= zint
    
    #pl.plot(zsarr, Ns)
    #pl.yscale('log')
    #pl.show()

    #cosmo = CosmologyFunctions(0.)
    #No little h
    #pk_arr_z0 = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
    #pkspl_z0 = interp1d(karr/cosmo._h, pk_arr_z0)
    #pl.loglog(karr, pk_arr_z0, label='Linear')

    #Non-linear power spectrum at z~0
    fpk = np.genfromtxt('/media/luna1/vinu/software/FrankenEmu/output_pk/pk_nonlin_z0.dat') #0_0.005.dat')
    karr, pk_arr_z0 = fpk[:,0], fpk[:,1]
    dlnk = np.log(karr[1]/karr[0])
    #pl.savetxt('data/pk_z0.dat', np.transpose((karr, pk_arr_z0)))
    kl = np.logspace(-6, np.log10(karr[0]),100)
    sl = (np.log(pk_arr_z0[1]) - np.log(pk_arr_z0[0]))/(np.log(karr[1]) - np.log(karr[0]))
    pkl = np.exp(np.log(pk_arr_z0[0]) - sl * (np.log(karr[0])-np.log(kl)))
    ku = np.logspace(np.log10(karr[-1]), 4, 100)
    su = (np.log(pk_arr_z0[-3]) - np.log(pk_arr_z0[-1]))/(np.log(karr[-3]) - np.log(karr[-1]))
    pku = np.exp(np.log(pk_arr_z0[-1]) - su * (np.log(karr[-1])-np.log(ku)))

    karr = np.hstack((kl[:-1], karr, ku[1:]))
    pk_arr_z0 = np.hstack((pkl[:-1], pk_arr_z0, pku[1:]))
    pkspl_z0 = interp1d(karr, pk_arr_z0)
    karr = np.logspace(np.log10(karr.min()), np.log10(karr.max()), 100)
    pk_arr_z0 = pkspl_z0(karr)
    #pl.loglog(karr, pk_arr_z0, label='Emulator')
    #pl.loglog(kl, pkl, label='Lower Extrapolation')
    #pl.loglog(ku, pku, label='Upper Extrapolation')
    #pl.legend(loc=0)
    #pl.show()
    #sys.exit()
    larr = np.logspace(np.log10(1), np.log10(10000), 500)
    Bl = np.exp(-1 * sigmasq * larr * larr)

    angH = angsarr.max()
    angL = angsarr.min()
    chiH = chisarr.max()
    chiL = chisarr.min()
 
    cl_kk = [] 
    for l in larr:
        pkarr = pkspl_z0(l/anglarr)
        cl_kk.append(integrate_kk(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns))
    cl_kk = np.array(cl_kk)
    cl_kk_sm = cl_kk * Bl
    #sys.exit()
    splcl_kk = interp1d(larr, cl_kk)
    splcl_kk_sm = interp1d(larr, cl_kk_sm)
    cl_kk = splcl_kk(mlarr)
    cl_kk_sm = splcl_kk_sm(mlarr)
    xi_kk = np.array([integrate_ell(mlarr, cl_kk, r, dl) for r in rradian])
    xi_kk_sm = np.array([integrate_ell(mlarr, cl_kk_sm, r, dl) for r in rradian])
    np.savetxt('../data/kk_power_const_bias_z1.dat', np.transpose((mlarr, cl_kk, cl_kk_sm)), fmt='%.2f %.3e %.3e', header='l Cl_kk Cl_kk_smoothed')
    np.savetxt('../data/kk_xi_const_bias_z1.dat', np.transpose((rarr, xi_kk, xi_kk_sm)), fmt='%.2f %.3e %.3e', header='arcmin xi xi_smoothed')

    cl_ky = [] 
    for l in larr:
        pkarr = pkspl_z0(l/anglarr)
        cl_ky.append(integrate_ky(0.33, zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, consty, chiH, Ns))
    cl_ky = np.array(cl_ky)
    cl_ky_sm = cl_ky * Bl


    splcl_ky = interp1d(larr, cl_ky)
    splcl_ky_sm = interp1d(larr, cl_ky_sm)
    cl_ky = splcl_ky(mlarr)
    cl_ky_sm = splcl_ky_sm(mlarr)
    xi_ky = np.array([integrate_ell(mlarr, cl_ky, r, dl) for r in rradian])
    xi_ky_sm = np.array([integrate_ell(mlarr, cl_ky_sm, r, dl) for r in rradian])


    np.savetxt('../data/ky_power_const_bias_z1.dat', np.transpose((mlarr, cl_ky, cl_ky_sm)), fmt='%.2f %.3e %.3e', header='l Cl_kk Cl_kk_smoothed')
    np.savetxt('../data/ky_xi_const_bias_z1.dat', np.transpose((rarr, xi_ky, xi_ky_sm)), fmt='%.2f %.3e %.3e', header='arcmin xi xi_smoothed')
    #np.savetxt('data/pk_%.1f.txt'%redshift, np.transpose((karr, pk_arr))) 

    dfile = np.load(dfile)
    theta_arcmin = dfile['theta_arcmin']
    rradian = theta_arcmin / 60. * np.pi / 180.
    ey = dfile['ey']
    ey_cov = dfile['ey_cov']
    ey_cov = np.diagonal(ey_cov)
    print ey, ey_cov
    by = dfile['by']
    by_cov = dfile['by_cov']
    by_err = np.sqrt(np.diagonal(by_cov))
    if 0:
        chi2 = [] 
        barr = np.linspace(0.1, 0.4, 10)
        for bi in barr:
            cl_ky = []
            for l in larr:
                pkarr = pkspl_z0(l/anglarr)
                cl_ky.append(integrate_ky(bi, zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, consty, chiH, Ns))
            cl_ky = np.array(cl_ky)
            cl_ky_sm = cl_ky * Bl

            splcl_ky_sm = interp1d(larr, cl_ky_sm)
            cl_ky_sm = splcl_ky_sm(mlarr)
            xi_ky_sm = np.array([integrate_ell(mlarr, cl_ky_sm, r, dl) for r in rradian])
            print xi_ky_sm
            chi2.append(((ey-xi_ky_sm)**2/ey_cov).sum())
        np.savetxt('chi2_const_bias.dat', np.transpose((barr, chi2)))


    if 1:
        mytools.matrc_small()
        pl.figure(1)

        f = np.genfromtxt('../data/kk_power_const_bias_z1.dat')
        larr = f[:,0] 
        cl = f[:,1]
        clsm = f[:,2]
        pl.loglog(mlarr, cl_kk, label=r'$C_\ell^{\kappa \kappa}$')
 
        f = np.genfromtxt('../data/ky_power_const_bias_z1.dat')
        larr = f[:,0] 
        cl = f[:,1]
        clsm = f[:,2]
        pl.loglog(mlarr, cl_ky, label=r'$C_\ell^{\kappa y}$')

        pl.legend(loc=0)
        pl.xlabel(r'$\ell$')
        pl.ylabel(r'$C_\ell$')
        pl.show()

        f = np.genfromtxt('../data/ky_xi_const_bias_z1.dat')
        rarr = f[:,0]
        xi = f[:,1]
        xism = f[:,2]
        pl.errorbar(theta_arcmin, ey, np.sqrt(ey_cov), marker='o', ls='', ms=8)
        pl.plot(rarr, xi, label='Unsmoothed')
        pl.plot(rarr, xism, label='Smoothed')
        pl.legend(loc=0)
        pl.xlim([0, 100])
        pl.show()


