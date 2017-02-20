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
from mass_function import bias_mass_func_tinker
from convert_NFW_RadMass import MfracToMvir, MvirToMRfrac, MfracToMfrac, MvirTomMRfrac, MfracTomMFrac, dlnMdensitydlnMcritOR200
 
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
    #It seems R200 is in the physical distance
    x = np.sqrt(x**2. + y**2)
    r = x * Rs
    x = r / R200
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


@jit(nopython=True) 
def integrate_batt_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, dlnmdlnm, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    cl1h = 0.0
    cl2h = 0.0
    jj = 0
    for i, lnzi in enumerate(lnzarr[:]):
        zi = np.exp(lnzi) - 1.
        zp = 1. + zi
        mint = 0.0
        for j, mi in enumerate(marr[:]): 
            Mvir, Rvir, M200, R200, rho_s, Rs = MfracToMvir(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h, frac=200.0)   
            #Mvir, Rvir, M200, R200, rho_s, Rs = MvirToMRfrac(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h, frac=200.0)   
            xmax = 4. * Rvir / Rs
            ells = chiarr[i] / zp / Rs

            xarr = np.linspace(1e-5, xmax, 100)

            yint = 0.
            for x in xarr:
                if x == 0:
                    continue
                yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile_2d(x, 0., Rs, M200, R200, zi, rho_crit_arr[i], omega_b0, omega_m0, cosmo_h))
            yint *= (4 * np.pi * Rs * (xarr[1] - xarr[0]) / ells / ells)

            mint += (dlnm * mf[jj] * dlnmdlnm[jj] * yint * yint)
            jj += 1
        my2 = 0.0
        cl1h += (dVdzdOm[i] * consty * consty * mint * zp)
    cl1h *= dlnz
    cl2h *= dlnz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 

if __name__=='__main__':
    '''
    Compute tSZ halomodel from the given mass and redshift
    MF : ST99 or T08
    '''
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
    consty = mpctocm * sigma_t_cm / rest_electron_kev #This is to convert preesure to tSZ

    mmin = 1e11
    mmax = 5e15
    mspace = 50

    dlnm = np.float64(np.log(mmax/mmin) / mspace)
    lnmarr = np.linspace(np.log(mmin), np.log(mmax), mspace)
    marr = np.exp(lnmarr).astype(np.float64)

    zmin = 1e-5
    zmax = 5
    zspace = 51
    lnzarr = np.linspace(np.log(1.+zmin), np.log(1.+zmax), zspace)
    dlnz = np.float64(np.log((1.+zmax)/(1.+zmin)) / zspace)

    print 'dlnm dlnz ', dlnm, dlnz

    hzarr, BDarr, rhobarr, chiarr, dVdzdOm, rho_crit_arr = [], [], [], [], [], []
    bias, Darr = [], []
    mf, dlnmdlnm = [], []
    zarr = np.exp(lnzarr) - 1.0

    for lnzi in lnzarr:
        zi = np.exp(lnzi) - 1.
        cosmo = CosmologyFunctions(zi)
        BDarr.append(cosmo.BryanDelta()) #OK
        rhobarr.append(cosmo.rho_bar() * cosmo._h * cosmo._h)
        chiarr.append(cosmo.comoving_distance() / cosmo._h)
        hzarr.append(cosmo.E0(zi))
        #Number of Msun objects/Mpc^3 (i.e. unit is 1/Mpc^3)
        m200arr = []
        for m in marr:
            #print MfracTomMFrac(m, zi, 200, cosmo.rho_crit() * cosmo._h * cosmo._h, cosmo.rho_bar() * cosmo._h * cosmo._h, cosmo_h, frac=200.0)
            #print '%.2e %.2e'%(cosmo.rho_crit(), cosmo.rho_bar())
            #m200arr.append(MvirTomMRfrac(m, zi, cosmo.BryanDelta(), cosmo.rho_crit() * cosmo._h * cosmo._h, cosmo.rho_bar() * cosmo._h * cosmo._h, cosmo_h, frac=200.0)[2])
            Ma = MfracTomMFrac(m, zi, 200, cosmo.rho_crit() * cosmo._h * cosmo._h, cosmo.rho_bar() * cosmo._h * cosmo._h, cosmo_h, frac=200.0)[2]
            m200arr.append(Ma)
            print '%.2e %.2e '%(m, Ma), dlnMdensitydlnMcritOR200(200. * cosmo.omega_m(), 200., Ma, m, zi, cosmo_h)
            dlnmdlnm.append(dlnMdensitydlnMcritOR200(200. * cosmo.omega_m() * cosmo._h * cosmo._h, 200. * cosmo._h * cosmo._h, Ma, m, zi, cosmo_h))
            #print '%.2e %.2e'%(m, MvirToMRfrac_m(m, zi, cosmo.BryanDelta(), cosmo.rho_crit() * cosmo._h * cosmo._h, cosmo.rho_bar() * cosmo._h * cosmo._h, cosmo_h, frac=200.0)[2])
            sys.exit()
        m200arr = np.array(m200arr)
        m200arr_w_h = m200arr*cosmo._h
        #print np.log(m200arr_w_h[1:]/m200arr_w_h[:-1])
        mf.append(bias_mass_func_tinker(zi, m200arr_w_h.min(), m200arr_w_h.max(), mspace, bias=False, marr=m200arr_w_h)[1])
        rho_crit_arr.append(cosmo.rho_crit() * cosmo._h * cosmo._h) #OK
        dVdzdOm.append(cosmo.E(zi) / cosmo._h) #Mpc/h, It should have (km/s/Mpc)^-1 but in the cosmology code the speed of light is removed  
        Darr.append(cosmo._growth)
    hzarr = np.array(hzarr)
    BDarr = np.array(BDarr)
    rhobarr = np.array(rhobarr)
    chiarr = np.array(chiarr)
    dVdzdOm = np.array(dVdzdOm) * chiarr * chiarr
    rho_crit_arr = np.array(rho_crit_arr)
    mf = np.array(mf).flatten()
    Darr = np.array(Darr)
    dlnmdlnm = abs(np.array(dlnmdlnm))
    bias = np.array(bias).flatten()

    ellarr = np.array([200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1e4])
    cl_arr, cl1h_arr, cl2h_arr = [], [], []
    for ell in ellarr:
        cl1h, cl2h, cl = integrate_batt_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, dlnmdlnm, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        cl_arr.append(cl)
        cl1h_arr.append(cl1h)
        cl2h_arr.append(cl2h)
        print ell, cl1h, cl2h, cl

    
    np.savetxt('cl_yy.dat', np.transpose((ellarr, cl1h_arr, cl2h_arr, cl_arr)), fmt='%.3e')


    #Convert y to \delta_T using 150 GHz. (g(x) TCMB)^2 = 6.7354
    gx = 6.7354
    cl1h = np.array(cl1h_arr) * gx
    pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='This code')
    a = np.genfromtxt('battaglia_analytical.csv', delimiter=',')
    pl.plot(a[:,0], a[:,1], label='Battaglia Analytical')
    pl.xlabel(r'$\ell$')
    pl.ylabel(r'$C_\ell \ell (\ell + 1)/2\pi \mu K^2$')
    pl.legend(loc=0)
    pl.show()


