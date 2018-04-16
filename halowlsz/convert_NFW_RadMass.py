import os
import sys
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
def dlnMdensitydlnMcritOR200(delta, delta1, M, M1, z, cosmo_h, mvir): 
    '''
    Make sure that delta and delta1 is above critical density. 
    In halomodel_cl_WL_tSZ.py the delta is  
    delta - density at which mass function is calculated. i.e. 
    MassDef * omega_m(z) of critical density. For Tinker mass function this is
    MassDef * mean density (mean density  = omega_m(z) * critical density)
    M is corresponds to delta definition 
    delta1 - standard mass definition. i.e. the mass corresponds 
    to either virial or 200 times critical density
    M1 is corresponds to delta1 definition 
    '''
    #print delta, delta1, M, M1,z
    a1 = 0.5116
    a2 = -0.4283
    a3 = -3.13e-3
    a4 = -3.52e-5
    Delta = delta / delta1
    if mvir: 
        conc = concentration_duffy(M1, z, cosmo_h)
    else:
        conc = concentration_duffy_200(M1, z, cosmo_h)
    A = np.log(1.+conc) - 1. + 1. / (1. + conc)
    f = Delta * (1./conc**3) * A
    p = a2 + a3 * np.log(f) + a4 * np.log(f)**2
    x = (a1 * f**(2.*p) + 0.75**2.)**(-0.5) + 2. * f
    B = -0.081
    t31 = Delta / conc**2 * (1./(1. + conc) - 1./(1. + conc)**2.) - 3. * Delta * A / conc**3.
    t32 = 2. - p * a1 * f**(2*p-1.) * (a1 * f**(2*p) + 0.75**2.)**-1.5
    dMdM1 = Delta / (conc * x)**3 * (1. - 3 * B / x * (x + t31 * t32))
    #print dMdM1
    #print conc, f, p, x, t1, t2
    ##t21 = 1./conc**3 * (2./(1. + conc) - conc / (1. + conc)**2.)
    ##t22 = (3. / conc**4) * (np.log(1. + conc) + conc / (1. + conc))
    ##t2 = t2 * (t21 - t22)
    ##t21 = -3 * f / conc + f / A * (1./(1.+conc) - 1./(1.+conc)**2.)
    #print t2, t21, t22
    dlnMdlnM1 = (M1/M) * dMdM1
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
    x0 = Rs * 1.1
    tol = Rvir * tolerance #tolerance
    x1 = 1 * Rvir #tol * 10**6
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
    x0 = Rfrac/2. 
    tol = Rfrac * tolerance #tolerance
    x1 = Rfrac *2. #tol * 10**6
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


@jit(nopython=True)
def HuKravtsov(z, M, rho, delta, deltao, cosmo_h, mvir):
    '''
    Eq. C10 in Hu&Kravstov to convert virial mass to any mass within some delta
    either both in critical density or mean density. In this function I use
    critical density. deltao is appropriately multiplied by omega(z) to find 
    output mass. i.e. lets say the virial critical mass is mass within a radius 
    which contains the average density is Delta_v*rho_critical and to find the
    mass within the average mean density of 200 mean density, then I use
    200 * (Omega(z) * rho_critical) = 50 * rho_critical. i.e. mass within
    50 times rho_critical 
    Input: 
           z : redshift
           M which either Mvir or M200c solar mass
           rho : Rho used for Mvir or M200c
           delta : fraction of rho corresponds to Mvir or M200c
           deltao : fraction of rho corresponds to output mass 
           mvir : should be 1 
    '''
    a1 = 0.5116
    a2 = -0.4283
    a3 = -3.13e-3
    a4 = -3.52e-5
    Delta = deltao / delta
    if mvir:
        conc = concentration_duffy(M, z, cosmo_h) 
        B = -0.081
    else:
        #raise ValueError('mvir should be 1')
        conc = concentration_duffy_200(M, z, cosmo_h) 
        B = -0.084
    R = (M / ((4 * np.pi / 3.) * delta * rho))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    rho_s = rho * (delta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = R / conc

    A = np.log(1.+conc) - 1. + 1. / (1. + conc)
    f = Delta * (1./conc**3) * A
    p = a2 + a3 * np.log(f) + a4 * np.log(f)**2
    x = (a1 * f**(2.*p) + 0.75**2.)**(-0.5) + 2. * f
    Mfrac = M * Delta * (1./conc/x)**3
    Rfrac = (3. * Mfrac / deltao / rho / 4. / np.pi)**(1./3.)
    return M, R, Mfrac, Rfrac, rho_s, Rs

if __name__=='__main__':
    z=0.07
    Mvir = 1e15
    cosmo = CosmologyFunctions(z, 'wlsz.ini', 'battaglia')
    omega_b = cosmo._omega_b0
    omega_m = cosmo._omega_m0
    cosmo_h = cosmo._h
    BryanDelta = cosmo.BryanDelta() 
    rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h
    rho_bar = cosmo.rho_bar() * cosmo._h * cosmo._h
    for D in np.arange(1, 100, 10):
        M, R, Mfrac, Rfrac, rho_s, Rs = HuKravtsov(z, Mvir, rho_critical, BryanDelta, D*cosmo.omega_m(), cosmo_h, True)
        print '%.2e %.2f %.2e %.2f %.2e %.2f'%(M, R, Mfrac, Rfrac, rho_s, Rs)
    sys.exit()
    print 'rho_critical = %.2e , rho_bar = %.2e'%(rho_critical, rho_bar)
    print 'Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs'
    Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)
    Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MfracToMvir(Mfrac, z, BryanDelta, rho_critical, cosmo_h, frac=200.0)
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)

    Mf, Rf, Mfrac, Rfrac, rho_s, Rs = MfracToMfrac(Mfrac, z, BryanDelta, 400, rho_critical, cosmo_h, frac=200.0)
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mf, Rf, Mfrac, Rfrac, rho_s, Rs)

    Mf, Rf, Mfrac, Rfrac, rho_s, Rs = MfracTomMFrac(Mfrac, z, 200, rho_critical, rho_bar, cosmo_h, frac=200.0)
    print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mf, Rf, Mfrac, Rfrac, rho_s, Rs)
    #Checking where HuKravtsov() & MvirToMRfrac() give the same answer and 
    #those do
    for Mvir in np.logspace(9, 16, 50):
        #Mvir = 1e15
        M, R, Mfrac, Rfrac, rho_s, Rs = HuKravtsov(z, Mvir, rho_critical, BryanDelta, 200*cosmo.omega_m(), cosmo_h, True)
        #print '%.2e %.2f %.2e %.2f %.2e %.2f'%(M, R, Mfrac, Rfrac, rho_s, Rs)
        Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h, frac=200.*cosmo.omega_m())
        #print '%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, Mfrac, Rfrac, rho_s, Rs)

        print dlnMdensitydlnMcritOR200(BryanDelta, 200*cosmo.omega_m(), Mvir, Mfrac, z, cosmo_h)
