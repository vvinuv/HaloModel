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
import cosmology_vinu as cosmology
#import fastcorr
from halomodel_tSZ import CosmologyFunctions
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

def halo_bias_st(sqnu):
    '''
    Eq. 8 in Sheth et al 2001
    '''
    common = 1./np.sqrt(0.707)/1.686
    fterm = np.sqrt(0.707) * 0.707 * sqnu #First term
    sterm = np.sqrt(0.707) * 0.5 * (0.707 * sqnu)**(1.-0.6) #second term
    ttermn = (0.707 * sqnu)**0.6 #numerator of third term
    ttermd = (0.707 * sqnu)**0.6 + 0.5 * (1.-0.6) * (1.-0.6/2.) #demoninator of third term
    tterm = ttermn / ttermd #third term
    blag = common * (fterm + sterm - tterm) #b_lag
    return 1+blag #b_eul
 

def bias_mass_func(redshift, lM200, uM200, mspace, bias=True):
    '''
    Wrote on Jan 26, 2017

    redshift : Redshift of mass function
    lM200 : Lower limit of M200
    uM200 : Upper limit of M200
    mspace : mass space
    bias : if weighted by ST bias (Doesn't work now)

    mass function in dn/dlnM200 in 1/Mpc^3

    M200 -solar unit
    '''

    cosmo0 = CosmologyFunctions(0)
    cosmo_h = cosmo0._h

    dlnm = np.float64(np.log(uM200/lM200) / mspace)
    lnmarr = np.linspace(np.log(lM200), np.log(uM200), mspace)
    marr = np.exp(lnmarr).astype(np.float64)
    #print 'dlnm ', dlnm

    #No little h
    #Need to give mass * h and get the sigma without little h
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo0._h) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    #print marr, sigma_m0

    cosmo = CosmologyFunctions(redshift)
    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0*cosmo._growth, k=3)
    A = 0.186 * (1. + cosmo.redshift())**-0.14 
    a = 1.47 * (1. + cosmo.redshift())**-0.06
    alpha = 10**(-(0.75/np.log10(200./75.))**1.2)
    b = 2.57   * (1. + cosmo.redshift())**-alpha
    c = 1.19

    mf,sarr,fsarr = [],[],[]
    for M200 in marr:
        mlow = M200 * 0.99
        mhigh = M200 * 1.01
        slow = lnMassSigmaSpl(np.log(mlow))
        shigh = lnMassSigmaSpl(np.log(mhigh))
        ds_dm = (shigh - slow) / (mhigh - mlow)
        sigma = lnMassSigmaSpl(np.log(M200))
        #print '%.2e %.2f %.2e'%(M200, sigma, ds_dm)
     
        fsigma = A * np.exp(-c / sigma**2.) * ((sigma/b)**-a + 1.)
        #print '%.2e %.2e %.2f %.2f %.2f %.2f %.2f'%(M200, fsigma, A, a, b, c, sigma)
        mf.append(-1 * fsigma * rho_norm0 * cosmo._h * cosmo._h * ds_dm / sigma)
        sarr.append(sigma)
        fsarr.append(fsigma)
        
    mf = np.array(mf) 
    sarr = np.array(sarr) 
    fsarr = np.array(fsarr) 
    if 0:
        return mass_function * halo_bias_st(delta_c_sigma_m2), sigma, fsigma
    else:
        return marr, mf, sarr, fsarr

if __name__=='__main__':
    '''
    Compute mass function
    '''
    redshift = 1
    marr, mf, S, fs = bias_mass_func(redshift, 1e8, 1e16, 100, bias=False)
    #pl.plot(np.log10(1/S), np.log10(fs))
    #pl.show()
    #pl.loglog(marr*0.7, mf*0.71**3, label='Vinu')
    pl.loglog(marr, mf, label='Vinu')
    f = np.genfromtxt('hmf/mVector_PLANCK-SMT z: 1.0.txt')
    pl.loglog(f[:,0]/0.71, f[:,6]*0.71**3, label='HMF')
    #com = np.genfromtxt('/media/luna1/vinu/software/cosmosis/mft_output/mass_function/m_h.txt')
    #comf = np.genfromtxt('/media/luna1/vinu/software/cosmosis/mft_output/mass_function/dndlnmh.txt')[9]

    #pl.loglog(com*0.71**3/0.71, comf*0.71**3, label='COSMOSIS')
    #kf = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/massfunction/tinkerredshift/mf_tinker_redshift/Mh_dndlnMh.txt')
    #pl.loglog(kf[:,0]/0.71, kf[:,1]*0.71**3, label='Komastu')
    pl.legend(loc=0)
    pl.axis([1e12, 1e16, 1e-12, 1e1])
    pl.show()
