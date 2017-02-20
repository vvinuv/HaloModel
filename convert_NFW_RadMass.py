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

@jit(nopython=True)
def concentration_duffy(Mvir, z, cosmo_h):
    '''Duffy 2008'''
    conc = (7.85 / (1. + z)**0.71) * (Mvir * cosmo_h / 2e12)**-0.081
    return conc

@jit(nopython=True)
def concentration_duffy_200(M200, z, cosmo_h):
    '''Duffy 2008'''
    conc200 = (5.71 / (1. + z)**0.47) * (M200 * cosmo_h / 2e12)**-0.084
    return conc200

@jit(nopython=True)
def f_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**3. / 3.) - (rho_s * Rs**3) * (np.log((Rs + Rfrac) / Rs) - Rfrac / (Rs + Rfrac))

@jit(nopython=True)
def df_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**2.) - (rho_s * Rs**3) * (Rfrac / (Rs + Rfrac)**2.)

@jit(nopython=True)
def dlnMdensitydlnMcritOR200(delta, delta1, M, M1, z, cosmo_h): 
    '''
    delta - density at which mass function is calculated. i.e. 
    the mean density (omega_m(z) * critical density)

    M is corresponds to delta definition 

    delta1 - standard mass definition. i.e. the mass corresponds 
    to either virial or 200 times critical density

    M1 is corresponds to delta1 definition 
    '''
    print delta, delta1, M, M1,z
    a1 = 0.5116
    a2 = -0.4283
    a3 = -3.13e-3
    a4 = -3.52e-5
    conc = concentration_duffy_200(M1, z, cosmo_h)
    A = np.log(1.+conc) - conc / (1. + conc)
    f = (delta / delta1) * (1./conc**3) * A
    p = a2 + a3 * np.log(f) + a4 * np.log(f)**2
    x = (a1 * f**(2.*p) + 0.75**2.)**(-0.5) + 2. * f
    B = -0.084
    t1 = (delta / delta1) * (1./x/conc)**3.
    t21 = 2. - a1 * p * f**(2*p-1) * (a1 * f**(2*p) + 0.75**2)**-1.5
    t22 = (-3. * f / conc) + f * (1./(1.+conc) - 1./(1.+conc)**2) / A
    t2 = (delta / delta1) * B * conc * (-3. / (conc*x)**4) * (x + conc * t21 * t22)
    #print conc, f, p, x, t1, t2
    ##t21 = 1./conc**3 * (2./(1. + conc) - conc / (1. + conc)**2.)
    ##t22 = (3. / conc**4) * (np.log(1. + conc) + conc / (1. + conc))
    ##t2 = t2 * (t21 - t22)
    ##t21 = -3 * f / conc + f / A * (1./(1.+conc) - 1./(1.+conc)**2.)
    #print t2, t21, t22
    dlnMdlnM1 = (M1/M) * (t1 + t2)
    return dlnMdlnM1

@jit(nopython=True)
def dMdensitydMcritOR200(M1, z, omegam): 
    '''
    delta - density at which mass function is calculated. i.e. 
    the mean density (omega_m(z) * critical density)

    M is corresponds to delta definition 

    delta1 - standard mass definition. i.e. the mass corresponds 
    to either virial or 200 times critical density

    M1 is corresponds to delta1 definition 
    '''
    g0 = 3.54e-2 + omegam**0.09
    g1 = 4.56e-2 + 2.68e-2/omegam
    g2 = 0.721 + 3.50e-2/omegam
    g3 = 0.628 + 0.164/omegam
    d0 = -1.67e-2 + 2.18e-2 * omegam
    d1 = 6.52e-3 - 6.86e-3 * omegam
    d = d0 + d1 * z
    g = g0 + g1 * np.exp(-1. * ((g2-z)/g3)**2.)
    MM1 = g + d * np.log(M1)
    return MM1


@jit(nopython=True)
def MvirTomMRfrac(Mvir, z, BryanDelta, rho_critical, rho_bar, cosmo_h, frac=200.0):
    '''Convert Mvir in solar mass to Rvir in Mpc, M200 in solar mass 
       R200 in Mpc
    '''
    conc = concentration_duffy(Mvir, z, cosmo_h)
    #print Mvir, conc
    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rvir / conc

    tolerance = 1e-6
    # Intial guess is Rvir / 2. 
    x0 = Rvir / 2.0
    tol = Rvir * tolerance #tolerance
    x1 = 2 * Rvir #tol * 10**6
    #print 1, x0, x1
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_bar, frac) / df_Rfrac(x0, rho_s, Rs, rho_bar, frac)
        #print x0, x1
    Rfrac = x1
    Mfrac = (4. / 3.) * np.pi * Rfrac**3 * frac * rho_bar
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs

@jit(nopython=True)
def MfracTomMFrac(Mfrac, z, frac2, rho_critical, rho_bar, cosmo_h, frac=200.0):
    '''Convert M200 in solar mass to R200 in Mpc, average density 
       ,ie. omega_m(z) * critical density, Mfrac in solar mass 
       Rvir in Mpc
    '''
    conc = concentration_duffy_200(Mfrac, z, cosmo_h)
    #print Mvir, conc
    Rfrac = (Mfrac / ((4 * np.pi / 3.) * frac * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    #rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    rho_s = rho_critical * (frac / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rfrac / conc

    tolerance = 1e-6
    # Intial guess is Rfrac
    x0 = Rfrac/2 
    tol = Rfrac * tolerance #tolerance
    x1 = Rfrac *2 #tol * 10**6
    #print 1, x0, x1, abs(x0 - x1), tol
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_bar, frac2) / df_Rfrac(x0, rho_s, Rs, rho_bar, frac2)
        #print x0, x1
    Rfrac2 = x1
    Mfrac2 = (4. / 3.) * np.pi * Rfrac2**3 * frac2 * rho_bar
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mfrac, Rfrac, Mfrac2, Rfrac2, rho_s, Rs


@jit(nopython=True)
def MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h, frac=200.0):
    '''Convert Mvir in solar mass to Rvir in Mpc, M200 in solar mass 
       R200 in Mpc
    '''
    conc = concentration_duffy(Mvir, z, cosmo_h)
    #print Mvir, conc
    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rvir / conc

    tolerance = 1e-6
    # Intial guess is Rvir / 2. 
    x0 = Rvir / 2.0
    tol = Rvir * tolerance #tolerance
    x1 = tol * 10**6
    #print 1, x0, x1
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, frac) / df_Rfrac(x0, rho_s, Rs, rho_critical, frac)
        #print x0, x1
    Rfrac = x1
    Mfrac = (4. / 3.) * np.pi * Rfrac**3 * frac * rho_critical
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs

@jit(nopython=True)
def MfracToMvir(Mfrac, z, BryanDelta, rho_critical, cosmo_h, frac=200.0):
    '''Convert M200 in solar mass to R200 in Mpc, Mvir in solar mass 
       Rvir in Mpc
    '''
    conc = concentration_duffy_200(Mfrac, z, cosmo_h)
    #print Mvir, conc
    Rfrac = (Mfrac / ((4 * np.pi / 3.) * frac * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    #rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    rho_s = rho_critical * (frac / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rfrac / conc

    tolerance = 1e-6
    # Intial guess is Rfrac
    x0 = Rfrac*2 
    tol = Rfrac * tolerance #tolerance
    x1 = Rfrac #tol * 10**6
    #print 1, x0, x1, abs(x0 - x1), tol
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, BryanDelta) / df_Rfrac(x0, rho_s, Rs, rho_critical, BryanDelta)
        #print x0, x1
    Rvir = x1
    Mvir = (4. / 3.) * np.pi * Rvir**3 * BryanDelta * rho_critical

    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc 
    conc = concentration_duffy(Mvir, z, cosmo_h)
    Rs = Rvir / conc
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3 
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs

#@jit(nopython=True)
def MfracToMfrac(Mfrac, z, BryanDelta, frac2, rho_critical, cosmo_h, frac=200.0):
    '''Convert frac to frac2
       Rvir in Mpc
       Mf, Rf are in frac2 and Mfrac, Rfrac in frac
    '''
    conc = concentration_duffy_200(Mfrac, z, cosmo_h)
    #print Mvir, conc
    Rfrac = (Mfrac / ((4 * np.pi / 3.) * frac * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    #rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    rho_s = rho_critical * (frac / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rfrac / conc

    tolerance = 1e-6
    # Intial guess is Rfrac
    x0 = Rfrac*2 
    tol = Rfrac * tolerance #tolerance
    x1 = Rfrac #tol * 10**6
    #print 1, x0, x1, abs(x0 - x1), tol
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, BryanDelta) / df_Rfrac(x0, rho_s, Rs, rho_critical, BryanDelta)
        #print x0, x1
    Rvir = x1
    Mvir = (4. / 3.) * np.pi * Rvir**3 * BryanDelta * rho_critical

    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc 

    tolerance = 1e-6
    # Intial guess is Rfrac
    x0 = Rfrac/2. 
    tol = Rfrac * tolerance #tolerance
    x1 = Rfrac #tol * 10**6
    #print 1, x0, x1, abs(x0 - x1), tol
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, frac2) / df_Rfrac(x0, rho_s, Rs, rho_critical, frac2)
        #print x0, x1
    Rf = x1
    Mf = (4. / 3.) * np.pi * Rf**3 * frac2 * rho_critical

    Rf = (Mf / ((4 * np.pi / 3.) * frac2 * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc 
    conc = concentration_duffy(Mvir, z, cosmo_h)
    Rs = Rvir / conc
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3 
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mf, Rf, Mfrac, Rfrac, rho_s, Rs




if __name__=='__main__':
    z=1.0
    Mvir = 1e11
    cosmo = CosmologyFunctions(z)
    omega_b = cosmo._omega_b0
    omega_m = cosmo._omega_m0
    cosmo_h = cosmo._h
    BryanDelta = cosmo.BryanDelta() 
    rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h
    print 'Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs'
    Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
    print '%.2e %.2e %.2e %.2e %.2e %.2e'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)
    Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MfracToMvir(Mfrac, z, BryanDelta, rho_critical, cosmo_h, frac=200.0)
    print '%.2e %.2e %.2e %.2e %.2e %.2e'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)

    Mf, Rf, Mfrac, Rfrac, rho_s, Rs = MfracToMfrac(Mfrac, z, BryanDelta, 400, rho_critical, cosmo_h, frac=200.0)
    print '%.2e %.2e %.2e %.2e %.2e %.2e'%(Mf, Rf, Mfrac, Rfrac, rho_s, Rs)

