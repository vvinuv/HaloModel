import os
import sys
import config
import numpy as np
from scipy import interpolate, integrate
from scipy.interpolate import UnivariateSpline, InterpolatedUnivariateSpline
from scipy.ndimage.filters import gaussian_filter
import pylab as pl
from numba import double, float64, float32
from numba import jit
import numba as nb
import timeit
from cosmology_vinu import CosmologyFunctions

@jit((nb.float64)(nb.float64, nb.float64, nb.float64),nopython=True)
def concentration(Mvir, redshift, cosmo_h):
    '''Duffy 2008
    '''
    Mvir = Mvir * cosmo_h
    conc = (5.72 / (1.+redshift)**0.71) * (Mvir/1e14)**-0.081
    return conc

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def f_Rfrac(Rfrac, rho_s, Rs, rho_critical, fraction):
    return (fraction * rho_critical * Rfrac**3. / 3.) - (rho_s * Rs**3) * (np.log((Rs + Rfrac) / Rs) - Rfrac / (Rs + Rfrac))

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def df_Rfrac(Rfrac, rho_s, Rs, rho_critical, fraction):
    return (fraction * rho_critical * Rfrac**2.) - (rho_s * Rs**3) * (Rfrac / (Rs + Rfrac)**2.)

@jit(nb.typeof((1.0,1.0))(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h, fraction):
    '''Convert Mvir in solar mass to Rvir in Mpc, to the fraction of critical
       density which compute M_fraction in solar mass and R_fraction in Mpc
    '''
    conc = concentration(Mvir, z, cosmo_h)
    #print Mvir, conc
    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rvir / conc

    tolerance = 1e-6

    # Using Newton - Raphson method. x1 = x0 - f(x0) / f'(x0) where x0 is
    # the initial guess, f and f' are the function and derivative

    # Intial guess is Rvir / 2. 
    x0 = Rvir / 2.0
    tol = Rvir * tolerance #tolerance
    x1 = tol * 10**6
    #print 1, x0, x1
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, fraction) / df_Rfrac(x0, rho_s, Rs, rho_critical, fraction)
        #print x0, x1
    Rfrac = x1
    Mfrac = (4. / 3.) * np.pi * Rfrac**3 * fraction * rho_critical
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mfrac, Rfrac


@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def battaglia_profile(r, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    Retrun: 
        Pressure profile in keV/cm^3 at radius r
    '''
    M200, R200 = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h, 200)
    #It seems R200 is in the physical distance, i.e. proper distance
    #Need to multiplied by (1+z) to get the comoving unit as I am giving r in
    #comoving unit.
    R200 *= (1. + z) #Comoving radius 
    x = r / R200
    #print Mvir, M200, R200
    msolar = 1.9889e30 #kg
    mpc2cm = 3.0856e24 #cm 
    G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * omega_b0 * G * M200 / omega_m0 / 2. / (R200 / (1. + z)) #Msun km^2 / Mpc^3 / s^2

    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415)
    #print P0, xc, beta
    #print (P200*msolar * 6.24e18 * 1e3 / mpc2cm**3), P0, xc, beta
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta) #(km/s)^2 M_sun / Mpc^3

    #Joule = kg m^2 / s^2, Joule = 6.24e18 eV = 6.24e15 keV
    pth *= (msolar * 6.24e15 * 1e6 / mpc2cm**3) #keV/cm^3. 1e6 implies that I have converted km to m
    p_e = pth * 0.518 #For Y=0.24, Vikram, Lidz & Jain
    return p_e


if __name__=='__main__':
    redshift = 0.
    Mvir = 1e15
    cosmo = CosmologyFunctions(redshift)
    BryanDelta = cosmo.BryanDelta() #OK
    #Msun/Mpc^3 
    rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h #OK
    #print BryanDelta, rho_critical
    omega_b0 = cosmo._omega_b0
    omega_m0 = cosmo._omega_m0
    cosmo_h = cosmo._h
    rarr = np.logspace(np.log10(0.1), np.log10(10), 10)

    pressure = np.array([battaglia_profile(r, Mvir, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h) for r in rarr])
    pl.loglog(rarr, pressure)
    pl.xlabel('Radius (comoving Mpc)')
    pl.ylabel('keV/cm^3')
    pl.show()
