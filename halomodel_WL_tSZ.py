import os
import sys
import time
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
#import fastcorr
from CosmologyFunctions import CosmologyFunctions
from mass_function import bias_mass_func_tinker, bias_mass_func_st, halo_bias_st
from convert_NFW_RadMass import MfracToMvir, MvirToMRfrac, MfracToMfrac
import pressure_profiles as pprof
import pygsl.integrate as intgsl
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

@jit(nopython=True)
def battaglia_profile_2d(x, y, Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    mtype: Definition of mass provided. mtype=vir or frac
    x = r/Rs where r and Rs in angular diameter distance 
    if you input r in comoving unit then you need to change R200 *= (1+z) 
    in the code and inclde z in P200. Currently it is 1. + 0 and it should be
    changed to 1. + z 
    Retrun: 
        Pressure profile in eV/cm^3 at radius r in angular comoving distance
    
    This result is confirmed by using Adam's code
    '''
    #It seems R200 is in the physical distance, i.e. proper distance
    #Need to multiplied by (1+z) to get the comoving unit as I am giving r in
    #comoving unit.
    #R200 *= (1. + z) #Comoving radius 
    #r = x * (1. + z) * Rs
    x = np.sqrt(x**2. + y**2)
    r = x * Rs
    x = r / R200
    #print Mvir, M200, R200
    msolar = 1.9889e30 #kg
    mpc2cm = 3.0856e24 #cm 
    G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * omega_b0 * G * M200 / omega_m0 / 2. / (R200 / (1. + 0)) #Msun km^2 / Mpc^3 / s^2

    #Delta=200
    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415)
    #Delta=500
    #P0 = 7.49 * ((M200 / 1e14)**0.226 * (1. + z)**-0.957)
    #xc = 0.710 * ((M200 / 1e14)**-0.0833 * (1. + z)**0.853)
    #beta = 4.19 * ((M200 / 1e14)**0.0480 * (1. + z)**0.615)
    #Shock Delta=500
    #P0 = 20.7 * ((M200 / 1e14)**-0.074 * (1. + z)**-0.743)
    #xc = 0.438 * ((M200 / 1e14)**0.011 * (1. + z)**1.01)
    #beta = 3.82 * ((M200 / 1e14)**0.0375 * (1. + z)**0.535)
 

    #print P0, xc, beta
    #print (P200*msolar * 6.24e18 * 1e3 / mpc2cm**3), P0, xc, beta
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta) #(km/s)^2 M_sun / Mpc^3

    #Joule = kg m^2 / s^2, Joule = 6.24e18 eV = 6.24e15 keV
    pth *= (msolar * 6.24e15 * 1e6 / mpc2cm**3) #keV/cm^3. 1e6 implies that I have converted km to m
    p_e = pth * 0.518 #For Y=0.24, Vikram, Lidz & Jain
    return p_e

@jit(nopython=True)
def battaglia_profile_proj(x, Rs, M200, R200, zi, rho_crit, xmax, omega_b0, omega_m0, cosmo_h):
    M = np.sqrt(xmax**2 - x**2)
    N = int(M / 0.1)
    if N == 0:
        return 2. * battaglia_profile_2d(x, 0, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h)
    else:
        xx = np.linspace(0, M, N)
        f = 0.0
        for x1 in xx:
            f += battaglia_profile_2d(x, x1, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h)
        #print xx
        f *= (2 * (xx[1] - xx[0]))
        return f


@jit(nopython=True)
def yintegral(x, ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h):
    '''Eq. 3.3 in Ma et al'''
    return x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_2d(x, 0, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h)

@jit(nopython=True)
def yintegration(xmax, ell, ells, Rs, M200, R200, z, rho_crit, omega_b0, omega_m0, cosmo_h):
    xarr = np.linspace(1e-5, xmax, 100)
    yint = 0.
    for x in xarr:
        if x == 0:
            continue
        yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_2d(x, 0., Rs, M200, R200, z, rho_crit, omega_b0, omega_m0, cosmo_h))
        #yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_proj(x, Rs, M200, R200, z, rho_crit, xmax, omega_b0, omega_m0, cosmo_h))
    yint *= (4 * np.pi * Rs * (xarr[1] - xarr[0]) / ells / ells)
    return yint

def mintegral(mi, mspl, z, BD, rho_crit, chi, dVdzdOm, ell, omega_b0, omega_m0, cosmo_h, consty):
    Mvir, Rvir, M200, R200, rho_s, Rs = MvirToMRfrac(mi, z, BD, rho_crit, cosmo_h, frac=200.0)
    #Eq. 3.3 Ma et al
    xmax = 4. * Rvir / Rs
    ells = chi / (1.+z) / Rs
    
    #yint = integrate.quad(yintegral, 0, xmax, args=(ell, ells, Rs, M200, R200, z, rho_crit, omega_b0, omega_m0, cosmo_h), epsabs=1.49e-22, epsrel=1.49e-22, limit=100)[0] * (4 * np.pi * Rs / ells / ells)

    yint = yintegration(xmax, ell, ells, Rs, M200, R200, z, rho_crit, omega_b0, omega_m0, cosmo_h)
    if mspl(M200) < 0 or mspl(M200) > 1e-11:
        mint = 0.
        print 'mint ', M200, mspl(M200)
    else:
        mint = mspl(M200)
    return mint * yint * yint * consty * consty

def integrate_batt_yyhalo_quad(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    cl1h = 0.0
    cl2h = 0.0
    for i, lnzi in enumerate(lnzarr[:]):
        zi = np.exp(lnzi) - 1.
        zp = 1. + zi
        #mm = np.logspace(11, 16, 1000)
        #if zi < 0.9:
        #    ms = 50
        #    maxim = 5e15
        #elif zi >= 0.9 and zi < 1.7:
        #    ms = 50
        #    maxim = 8e14
        #elif zi >= 1.7 and zi < 2.5:
        #    ms = 50
        #    maxim = 1e14
        #elif zi >= 2.5 and zi < 4.0:
        #    ms = 50
        #    maxim = 5e13
        #elif zi >= 4.:
        #    ms = 50
        #    maxim = 1e13
        tmarr, tmf, a, b = bias_mass_func_tinker(zi, 1e10, 1e16, 100, bias=False, mtune=True)
        #print tmarr.max(), tmarr.min()
        if zi > 1.1:
            mspl = InterpolatedUnivariateSpline(tmarr/cosmo_h, tmf*cosmo_h/tmarr, k=1)
        else:
            mspl = InterpolatedUnivariateSpline(tmarr/cosmo_h, tmf*cosmo_h/tmarr, k=2)
        #pl.loglog(tmarr, tmf*cosmo_h/tmarr, label='large')
        #pl.loglog(mm, mspl(np.log10(mm)), label='spl-large') 
        #print mspl(np.log10(8.5e15))
        #tmarr, tmf, a, b = bias_mass_func_tinker(zi, 1e10, 1e17, 50, bias=False)
        #mspl = InterpolatedUnivariateSpline(np.log10(tmarr), tmf*cosmo_h/tmarr, k=1)
        #pl.loglog(tmarr, tmf*cosmo_h/tmarr, label='small')
        #pl.loglog(mm, mspl(np.log10(mm)), label='spl-small')
        #pl.legend(loc=0) 
        #pl.show()
        #sys.exit()
        mint1 = integrate.quad(mintegral, 1e11, 5e15, args=(mspl, zi, BDarr[i], rho_crit_arr[i], chiarr[i], dVdzdOm[i], ell, omega_b0, omega_m0, cosmo_h, consty), epsabs=1.49e-22, epsrel=1.49e-22, limit=100)
        print 'Quad ', ell, zi, mint1
        mint = mint1[0]
        my2 = 0.0
        cl1h += (dVdzdOm[i] * mint * zp)
        cl2h += (dVdzdOm[i] * pk[i] * Darr[i] * Darr[i] * consty * consty * my2 * my2)
    cl1h *= dlnz
    cl2h *= dlnz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 

@jit(nopython=True) 
def integrate_batt_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    cl1h = 0.0
    cl2h = 0.0
    #mmin = 1e11
    #mmax = 5e15
    #mspace = 50

    print dlnm
    jj = 0
    for i, lnzi in enumerate(lnzarr[:]):
        zi = np.exp(lnzi) - 1.
        zp = 1. + zi
        #print  zi, Wk(zi, chiarr[i], zsarr, angsarr, Ns, constk)
        #marr, mf, a, b = bias_mass_func_tinker(zi, mmin, mmax, mspace, bias=False)
        #print marr.min(), marr.max()
        mint = 0.0
        mintq = 0.0
        for j, mi in enumerate(marr[:]): 
            #Mvir, Rvir, M200, R200, rho_s, Rs = MfracToMvir(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h, frac=200.0)   
            Mvir, Rvir, M200, R200, rho_s, Rs = MvirToMRfrac(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h, frac=200.0)   
            xmax = 4. * Rvir / Rs
            ells = chiarr[i] / zp / Rs


            xarr = np.linspace(1e-5, xmax, 100)
            #p = [battaglia_profile_proj(x, Rs, M200, R200, zi, rho_crit_arr[i], xmax, omega_b0, omega_m0, cosmo_h) for x in xarr]
            #pl.loglog(xarr, p, label='proj')
            #p = [battaglia_profile_2d(x, 0., Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h) for x in xarr]
            #pl.loglog(xarr, p, label='r')
            #pl.legend(loc=0)
            #pl.show()

            yint = 0.
            for x in xarr:
                if x == 0:
                    continue
                yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_2d(x, 0., Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h))
                #yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_proj(x, Rs, M200, R200, zi, rho_crit_arr[i], xmax, omega_b0, omega_m0, cosmo_h))

            yint *= (4 * np.pi * Rs * (xarr[1] - xarr[0]) / ells / ells)

            #yintq = integrate.quad(yintegral, 0, xmax, args=(ell, ells, Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h), epsabs=1.49e-22, epsrel=1.49e-22, limit=1000)[0] * (4 * np.pi * Rs / ells / ells)
            #print yint, mf[jj]
            #print mi, integrate.quad(yintegral, 0, xmax, args=(ell, ells, Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h), epsabs=1.49e-22, epsrel=1.49e-22, limit=1000)
            #ygsl = intgsl.gsl_function(yintegral_gsl, [ell, ells, Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h])
            #ylist = intgsl.qags(ygsl, 1e-3, xmax, 1.49e-22, 1.49e-22, 1000, w)
            #print ylist
            #yint = ylist[1] * (4 * np.pi * Rs  / ells / ells)
            mint += (dlnm * mf[jj] * yint * yint)
            #if zi > 0.48 and zi < 0.5:
            #    print mf[jj]
            #mintq += (dlnm * mf[jj] * yintq * yintq)
            jj += 1
        #print ell, zi, mint * consty * consty #, mintq * consty * consty
        my2 = 0.0
        cl1h += (dVdzdOm[i] * consty * consty * mint * zp)
        #if zi > 0.48 and zi < 0.5:
        #    print zi, dVdzdOm[i]
        #cl1h += (dVdzdOm[i] * mint * zp)
        #cl1h += (mint)
        cl2h += (dVdzdOm[i] * pk[i] * Darr[i] * Darr[i] * consty * consty * my2 * my2)
    cl1h *= dlnz
    cl2h *= dlnz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 

def wl_tsz_model(compute, fwhm, zsfile='source_distribution.txt', kk=False, yy=False, ky=True, MF='ST99'):
    '''
    Compute tSZ halomodel from the given mass and redshift
    MF : ST99 or T08
    '''
    fwhm = fwhm * np.pi / 2.355 / 60. /180. #angle in radian
    fwhmsq = fwhm * fwhm
    
    cosmo0 = CosmologyFunctions(0)
    omega_b0 = cosmo0._omega_b0
    omega_m0 = cosmo0._omega_m0
    cosmo_h = cosmo0._h

    light_speed = 2.998e5 #km/s
    mpctocm = 3.085677581e24
    kB_kev_K = 8.617330e-8 #keV k^-1
    sigma_t_cm = 6.6524e-25 #cm^2
    rest_electron_kev = 511. #keV
    constk = 3. * omega_m0 * (cosmo_h * 100. / light_speed)**2. / 2. #Mpc^-2
    consty = mpctocm * sigma_t_cm / rest_electron_kev 

    fz= np.genfromtxt(zsfile)
    zsarr = fz[:,0]
    Ns = fz[:,1]
    zint = np.sum(Ns) * (zsarr[1] - zsarr[0])
    Ns /= zint

    kmin = 1e-4 #1/Mpc
    kmax = 1e4
    kspace = 100.
    #If ST MF is used then mmin, mmax are considered as the virial mass
    #If Tinker MF is used mmin, mmax are considered as the M200 where 200 is 
    #matter density (i.e. critical density times Omega_m (Tinker et al, 2008))
    mmin = 1e11
    mmax = 5e15
    mspace = 30

    dlnk = np.float64(np.log(kmax/kmin) / kspace)
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), kspace)
    karr = np.exp(lnkarr).astype(np.float64)
    #No little h
    #Input Mpc/h to power spectra and get Mpc^3/h^3
    pk_arr = np.array([cosmo0.linear_power(k/cosmo0._h) for k in karr]).astype(np.float64) / cosmo0._h / cosmo0._h / cosmo0._h
    pkspl = InterpolatedUnivariateSpline(karr/cosmo0._h, pk_arr, k=2) 
    #pl.loglog(karr, pk_arr)
    #pl.show()

    dlnm = np.float64(np.log(mmax/mmin) / mspace)
    lnmarr = np.linspace(np.log(mmin), np.log(mmax), mspace)
    lnmarr = (lnmarr[1:]+lnmarr[:-1])/2.
    marr = np.exp(lnmarr).astype(np.float64)
    mspace = marr.size
    #bias_mass_func(1e13, cosmo, ST99=True)
    #sys.exit()

    #No little h
    #Need to give mass * h and get the sigma without little h
    #The following lines are used only used for ST MF and ST bias
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo0._h) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0, k=3)


    zmin = 1e-5
    zmax = 2
    zspace = 21
    lnzarr = np.linspace(np.log(1.+zmin), np.log(1.+zmax), zspace)
    dlnz = np.float64(np.log((1.+zmax)/(1.+zmin)) / zspace)

    print 'dlnk, dlnm dlnz ', dlnk, dlnm, dlnz

    hzarr, BDarr, rhobarr, chiarr, dVdzdOm, rho_crit_arr = [], [], [], [], [], []
    bias, Darr = [], []
    mf, mfdm = [], []
    zarr = np.exp(lnzarr) - 1.0

    for lnzi in lnzarr:
        zi = np.exp(lnzi) - 1.
        cosmo = CosmologyFunctions(zi)
        BDarr.append(cosmo.BryanDelta()) #OK
        rhobarr.append(cosmo.rho_bar() * cosmo._h * cosmo._h)
        chiarr.append(cosmo.comoving_distance() / cosmo._h)
        hzarr.append(cosmo.E0(zi))
        #print zi, cosmo.comoving_distance() / cosmo._h
        #Number of Msun objects/Mpc^3 (i.e. unit is 1/Mpc^3)
        if MF == '1ST99':
            mf.append(bias_mass_func_st(zi, mmin, mmax, mspace, bias=False)[1])
        elif MF=='T08':
            m200arr = []
            for m in marr:
                m200arr.append(MvirToMRfrac(m, zi, cosmo.BryanDelta(), cosmo.rho_bar() * cosmo._h * cosmo._h, cosmo_h, frac=200.0)[2])
            m200arr = np.array(m200arr)
            m200arr_w_h = m200arr*cosmo._h
            #m200arr = marr.copy()
            #print bias_mass_func_tinker(zi, m200arr_w_h.min(), m200arr_w_h.max(), mspace, bias=False)[1]
            tm = bias_mass_func_tinker(zi, m200arr_w_h.min(), m200arr_w_h.max(), mspace, bias=False)[0]
            print dlnm, np.log(tm.max()/tm.min())/tm.size
          
            #print dlnm, np.float64(np.log(m200arr.max()/m200arr.min()) / mspace), np.float64(np.log(m200arr_w_h.max()/m200arr_w_h.min()) / mspace)

            #sys.exit()
            #mf.append(tmf[1])
            #mfdm.append(tmf[1]/marr)
            mf.append(bias_mass_func_tinker(zi, m200arr_w_h.min(), m200arr_w_h.max(), mspace, bias=False)[1])
            #mf.append((tmf[1:] + tmf[:-1])/2.)
        #marr = m200arr.copy()
        #print zi, marr, mfdm 
        #sys.exit()
        #bias.append(np.array([halo_bias_st(cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(m)) / lnMassSigmaSpl(np.log(m))) for m in marr]))
        rho_crit_arr.append(cosmo.rho_crit() * cosmo._h * cosmo._h) #OK
        dVdzdOm.append(cosmo.E(zi) / cosmo._h) #Mpc/h, It should have (km/s/Mpc)^-1 but in the cosmology code the speed of light is removed  
        Darr.append(cosmo._growth)

    #print marr
    #print zarr
    #print mf
    #print np.array(mf).flatten()
    #sys.exit()
    #mf = []
    #bias = []
    hzarr = np.array(hzarr)
    BDarr = np.array(BDarr)
    rhobarr = np.array(rhobarr)
    chiarr = np.array(chiarr)
    dVdzdOm = np.array(dVdzdOm) * chiarr * chiarr
    #for i, j in zip(zarr, dVdzdOm):
    #    print i, j
    #sys.exit()
    rho_crit_arr = np.array(rho_crit_arr)
    mf = np.array(mf).flatten()
    zchispl = InterpolatedUnivariateSpline(zarr, chiarr, k=2)
    chisarr = zchispl(zsarr)
    bias = np.array(bias).flatten()
    Darr = np.array(Darr)
    #zarr = (zarr[1:] + zarr[:-1]) / 2.
    #BDarr = (BDarr[1:] + BDarr[:-1]) / 2.
    #rhobarr = (rhobarr[1:] + rhobarr[:-1]) / 2.
    #chiarr = (chiarr[1:] + chiarr[:-1]) / 2.
    #dVdzdOm = (dVdzdOm[1:] + dVdzdOm[:-1]) / 2.
    print len(lnzarr) * len(marr), mf.shape


    #marr /= 0.71
    #marr = (marr[1:]+marr[:-1])/2.
    ellarr = np.linspace(11, 10001, 5)
    ellarr = np.array([200, 1000, 2000, 3000, 4000, 5000, 6000, 7000])
    #ellarr = np.logspace(0, np.log10(5000), 20)
    #ellarr = np.logspace(1, np.log10(10001), 50)
    #ellarr = [10., 15.84, 25.11, 39.81, 63.09, 100., 158.49, 251.19, 398.11, 630.96, 1000., 1584.89, 2511.89, 3981.07, 6309.57, 10000.]
    cl_arr, cl1h_arr, cl2h_arr = [], [], []
    for ell in ellarr:
        pk = pkspl(ell/chiarr)
        if ky: 
            cl1h, cl2h, cl = integrate_halo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        if kk:
            cl1h, cl2h, cl = integrate_kkhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        if yy:
            cl1h, cl2h, cl = integrate_batt_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
            #cl1h, cl2h, cl = integrate_batt_yyhalo_quad(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
            #cl1h, cl2h, cl = integrate_batt_yyhalo_scipy_quad_quad(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
            #cl1h, cl2h, cl = integrate_batt_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, pk, zsarr, chisarr, Ns, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
            #cl1h, cl2h, cl = integrate_ks_yyhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        cl_arr.append(cl)
        cl1h_arr.append(cl1h)
        cl2h_arr.append(cl2h)
        print ell, cl1h, cl2h, cl

    convolve = np.exp(-1 * fwhmsq * ellarr * ellarr)# i.e. the output is Cl by convolving by exp(-sigma^2 l^2)
    #print convolve
    cl = np.array(cl_arr) * convolve
    cl1h = np.array(cl1h_arr) * convolve
    cl2h = np.array(cl2h_arr) * convolve
    
    if ky:
        np.savetxt('cl_ky.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')
    if kk:
        np.savetxt('cl_kk.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')
    if yy:
        np.savetxt('cl_yy.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')


    if yy:
        gx = 7.2786
        #Convert y to \delta_T using 147 GHz. (g(x) TCMB)^2 = 7.2786
        cl *= gx
        cl1h *= gx
        cl2h *= gx
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi, label='Cl')
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
        pl.xlabel(r'$\ell$')
        pl.ylabel(r'$C_\ell \ell (\ell + 1)/2/\pi \mu K^2$')
        pl.legend(loc=0)
    else:
        pl.plot(ellarr, ellarr * (ellarr+1) * cl / 2. / np.pi, label='Cl')
        pl.plot(ellarr, ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
        pl.plot(ellarr, ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
        pl.xlabel(r'$\ell$')
        pl.ylabel(r'$C_\ell \ell (\ell + 1)/2/\pi$')
        pl.legend(loc=0)

    pl.show()
    #No little h
    #Mass_sqnu = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(Mass)) / lnMassSigmaSpl(np.log(Mass))
    #hb = np.float64(halo_bias_st(Mass_sqnu))

    #bmf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=True) for m in marr]).astype(np.float64)

    #sys.exit()

    #integrate_2halo(1, pk_arr, marr, karr, bmf, dlnk, dlnm, hb, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
    #sys.exit()
    #print rarr, cosmo.comoving_distance()
    #smooth_r = (fwhm/60.) * np.pi / 180 / np.sqrt(8 * np.log(2)) * cosmo.comoving_distance() #angle = arc/radius
    #print smooth_r
    #sys.exit()
    #Bk = np.exp(-karr*karr*smooth_r*smooth_r/2.)
    #if logr:
    #    rarr = np.logspace(np.log10(rmin), np.log10(rmax), space).astype(np.float64)
    #else:
    #    rarr = np.linspace(rmin, rmax, space).astype(np.float64)
    #print rarr


if __name__=='__main__':
    #Write variables
    compute = 0 #Whether the profile should be computed 
    fwhm = 0 #arcmin Doesn't work now
    rmin = 1e-2 #Inner radius of pressure profile 
    rmax = 1e2 #Outer radius of pressure profile
    space = 50 #logarithmic space between two points
    #Stop

    if 1:
        wl_tsz_model(compute, fwhm, zsfile='source_distribution.txt', kk=0, yy=1, ky=0, MF='T08')
    
    if 0:
        mpctocm = 3.085677581e24
        sigma_t_cm = 6.6524e-25 #cm^2
        rest_electron_kev = 511. #keV
        consty = mpctocm * sigma_t_cm / rest_electron_kev


        Rvir = 0.45
        xmax = 21.01
        ell = 3000
        ells = 14669.3
        Rs = 0.09
        M200 = 1.00e+13
        R200 = 0.37
        zi = 0.50
        rho_crit = 2.26e+11
        omega_b0 = 0.04
        omega_m0 = 0.26
        cosmo_h = 0.71
        yint = 1.06623e-10

        
        inte = integrate.quad(yintegral, 0., xmax, args=(ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h))
        print inte
        #print consty * inte[0] * (4 * np.pi * Rs  / ells / ells)     

        ini = time.time()  
        print 'quad ', integrate.quad(yintegral, 0., xmax, args=(ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h))[0] * (4 * np.pi * Rs  / ells / ells) * consty
        print time.time() - ini


        ini = time.time()  
        print 'romberg ', consty * integrate.romberg(yintegral, 1e-11, xmax, args=(ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h)) * (4 * np.pi * Rs  / ells / ells)
        print time.time() - ini

        yint = 0.0
        xarr = np.linspace(0., xmax, 10.) #/ Rs
        #print xarr
        p = []
        for x in xarr:
            if x == 0:
                continue
            yint += yintegral(x, ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h)
            p.append(yintegral(x, ell, ells, Rs, M200, R200, zi, rho_crit, omega_b0, omega_m0, cosmo_h))
        yint *= (xarr[1] - xarr[0])
        print 'trapizodal ', consty * yint * (4 * np.pi * Rs  / ells / ells)

        print 'simp ', consty * integrate.simps(np.array(p), x=xarr[1:], even='last') * (4 * np.pi * Rs  / ells / ells)


    if 0:
        marr, mf, a, b = bias_mass_func_tinker(0., 1e11, 5e15, 100, bias=False)
        print marr
        MMFspl = InterpolatedUnivariateSpline(marr, mf/marr, k=3)
        print integrate.quad(MMFspl, 1e11, 5e15)
    if 0:
        z = 0.01
        Mvir = 1e15
        cosmo = CosmologyFunctions(0)
        omega_b = cosmo._omega_b0
        omega_m = cosmo._omega_m0
        cosmo_h = cosmo._h
        BryanDelta = cosmo.BryanDelta() 
        rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h
        Mfrac, Rfrac, rho_s, Rs, Rvir = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
        rp = np.linspace(0, 5 * Rvir, 100)
        xp = rp / Rs / (1 + z) #Ma et al paper says that Eq. 3.3 convergence by r=5 rvir 
        Pe = []
        for x in xp:
            Pe.append(battaglia_profile(x, Rs, Mvir, z, BryanDelta, rho_critical, omega_b, omega_m, cosmo_h))
        Pe = np.array(Pe) * rp * rp * 1e3 #keV/cm^3 to ev/cm^3
        pl.semilogy(rp, Pe)
        pl.show() 
