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
from mass_function import bias_mass_func_tinker, bias_mass_func_bocquet
from convert_NFW_RadMass import MfracToMvir, MvirToMRfrac, MfracToMfrac, MvirTomMRfrac, MfracTomMFrac, dlnMdensitydlnMcritOR200, HuKravtsov
from pressure_profiles import battaglia_profile_2d

__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

@jit(nopython=True) 
def integrate_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, dlnmdlnm, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    dlnmdlnm /= dlnmdlnm
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

    #z= 1.
    #cosmo = CosmologyFunctions(z)
    #print(MvirTomMRfrac(1e14, z, cosmo.BryanDelta(), cosmo.rho_crit() * cosmo0._h * cosmo0._h, cosmo.rho_bar() * cosmo0._h * cosmo0._h, cosmo_h, frac=200.))
    #print(cosmo.BryanDelta(), 200*cosmo.omega_m()) 
    #print(HuKravtsov(z, 1e14, cosmo.rho_crit() * cosmo0._h * cosmo0._h, cosmo0.rho_bar() * cosmo0._h * cosmo0._h, cosmo.BryanDelta(), 200*cosmo.omega_m(), cosmo_h, 1))
    #print(cosmo.BryanDelta(), 200) 
    #print(MvirToMRfrac(1e14, z, cosmo.BryanDelta(), cosmo.rho_crit() * cosmo0._h * cosmo0._h, cosmo_h, frac=200.0))
    #print(HuKravtsov(z, 1e14, cosmo.rho_crit() * cosmo0._h * cosmo0._h, cosmo.rho_crit() * cosmo0._h * cosmo0._h, cosmo.BryanDelta(), 200, cosmo_h, 1))
    #m2m = MvirTomMRfrac(1e11, 0., cosmo0.BryanDelta(), cosmo0.rho_crit() * cosmo0._h * cosmo0._h, cosmo0.rho_bar() * cosmo0._h * cosmo0._h, cosmo_h, frac=200.)[2]
    #print(dlnMdensitydlnMcritOR200(200. * cosmo0.omega_m(), cosmo0.BryanDelta(), m2m, 1e11, 0, cosmo_h))
    #sys.exit()
    light_speed = 2.998e5 #km/s
    mpctocm = 3.085677581e24
    kB_kev_K = 8.617330e-8 #keV k^-1
    sigma_t_cm = 6.6524e-25 #cm^2
    rest_electron_kev = 511. #keV
    constk = 3. * omega_m0 * (cosmo_h * 100. / light_speed)**2. / 2. #Mpc^-2
    consty = mpctocm * sigma_t_cm / rest_electron_kev #This is to convert preesure to tSZ

    mmin = 1e11
    mmax = 5e15
    mspace = 21

    dlnm = np.float64(np.log(mmax/mmin) / mspace)
    lnmarr = np.linspace(np.log(mmin), np.log(mmax), mspace)
    marr = np.exp(lnmarr).astype(np.float64)

    zmin = 0.07
    zmax = 5
    zspace = 31
    lnzarr = np.linspace(np.log(1.+zmin), np.log(1.+zmax), zspace)
    dlnz = np.float64(np.log((1.+zmax)/(1.+zmin)) / zspace)

    print('dlnm dlnz ', dlnm, dlnz)

    hzarr, BDarr, rhobarr, chiarr, dVdzdOm, rho_crit_arr = [], [], [], [], [], []
    bias, Darr = [], []
    mf, dlnmdlnm = [], []
    zarr = np.exp(lnzarr) - 1.0

    for lnzi in lnzarr:
        zi = np.exp(lnzi) - 1.
        cosmo = CosmologyFunctions(zi)
        rcrit = cosmo.rho_crit() * cosmo._h * cosmo._h
        rbar = cosmo.rho_bar() * cosmo._h * cosmo._h
        bn = cosmo.BryanDelta()
        BDarr.append(bn) #OK
        rho_crit_arr.append(rcrit) #OK
        rhobarr.append(rbar)
        chiarr.append(cosmo.comoving_distance() / cosmo._h)
        hzarr.append(cosmo.E0(zi))
        #Number of Msun objects/Mpc^3 (i.e. unit is 1/Mpc^3)
        if config.MF =='Tinker':
            m200m = np.array([MvirTomMRfrac(mv, zi, bn, rcrit, rbar, cosmo_h, frac=200.)[2] for mv in marr]) * cosmo_h
            mf.append(bias_mass_func_tinker(zi, m200m.min(), m200m.max(), mspace, bias=False, Delta=400, marr=m200m)[1])
            #pl.loglog(marr, mf[0])
            for mv,m2m in zip(marr, m200m):
                dlnmdlnm.append(dlnMdensitydlnMcritOR200(200. * cosmo.omega_m(), bn, m2m/cosmo_h, mv, zi, cosmo_h))
            #m400m = np.array([MvirTomMRfrac(mv, zi, bn, rcrit, rbar, cosmo_h, frac=400.)[2] for mv in marr]) * cosmo_h
            #mf.append(bias_mass_func_tinker(zi, m400m.min(), m400m.max(), mspace, bias=False, Delta=400, marr=m400m)[1])
            #pl.loglog(marr, mf[1])
            #pl.show()
            #for mv,m4m in zip(marr, m400m):
            #    dlnmdlnm.append(dlnMdensitydlnMcritOR200(400. * cosmo.omega_m(), bn, m4m/cosmo_h, mv, zi, cosmo_h))
            #print dlnmdlnm
        elif config.MF == 'Bocquet':
            if config.MassToIntegrate == 'virial':
                m200 = np.array([HuKravtsov(zi, mv, rcrit, rcrit, bn, 200, cosmo_h, 1)[2] for mv in marr])
                mf.append(bias_mass_func_bocquet(zi, m200.min(), m200.max(), mspace, bias=False, marr=m200)[1])
                for mv,m2 in zip(marr, m200):
                    dlnmdlnm.append(dlnMdensitydlnMcritOR200(200., bn, m2, mv, zi, cosmo_h))
            elif config.MassToIntegrate == 'm200':
                tmf = bias_mass_func_bocquet(zi, marr.min(), marr.max(), mspace, bias=False, marr=marr)[1]
                mf.append(tmf)
                dlnmdlnm.append(np.ones(len(tmf)))
        dVdzdOm.append(cosmo.E(zi) / cosmo._h) #Mpc/h, It should have (km/s/Mpc)^-1 but in the cosmology code the speed of light is removed  
        Darr.append(cosmo._growth)
        #sys.exit()
    hzarr = np.array(hzarr)
    BDarr = np.array(BDarr)
    rhobarr = np.array(rhobarr)
    chiarr = np.array(chiarr)
    dVdzdOm = np.array(dVdzdOm) * chiarr * chiarr
    rho_crit_arr = np.array(rho_crit_arr)
    mf = np.array(mf).flatten()  * np.array(dlnmdlnm).flatten()
    Darr = np.array(Darr)
    dlnmdlnm = np.ones(mf.size)
    bias = np.array(bias).flatten()

    ellarr = np.array([200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 1e4])
    #ellarr = np.logspace(1, 4, 50)
    cl_arr, cl1h_arr, cl2h_arr = [], [], []
    for ell in ellarr:
        cl1h, cl2h, cl = integrate_yyhalo(ell, lnzarr, chiarr, dVdzdOm, marr, mf, dlnmdlnm, BDarr, rhobarr, rho_crit_arr, bias, Darr, hzarr, dlnz, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        cl_arr.append(cl)
        cl1h_arr.append(cl1h)
        cl2h_arr.append(cl2h)
        print(ell, cl1h, cl2h, cl)

    
    np.savetxt('data/cl_yy.dat', np.transpose((ellarr, cl1h_arr, cl2h_arr, cl_arr)), fmt='%.3e')


    #Convert y to \delta_T using 150 GHz. (g(x) TCMB)^2 = 6.7354
    gx = 6.7354
    cl1h = np.array(cl1h_arr) * gx
    pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='This code')
    a = np.genfromtxt('data/battaglia_analytical.csv', delimiter=',')
    pl.plot(a[:,0], a[:,1], label='Battaglia Analytical')
    pl.xlabel(r'$\ell$')
    pl.ylabel(r'$C_\ell \ell (\ell + 1)/2\pi \mu K^2$')
    pl.legend(loc=0)
    pl.show()


