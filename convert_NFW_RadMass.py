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
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)
    Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MfracToMvir(Mfrac, z, BryanDelta, rho_critical, cosmo_h, frac=200.0)
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)

