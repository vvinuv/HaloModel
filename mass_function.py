import os
import sys
import numpy as np
import pylab as pl
from cosmology_vinu import CosmologyFunctions
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate

class MassFunction:
    '''
    This class consists of mass function using Eqs. 56-59 of Cooray & Sheth 2002
    Input:
        Virial mass Mvir (not with little h)
        Redshift 
        mass function: ST99, CS02, PS74 

    Output:
        Mass function in 1/Mpc^3
    '''
    def __init__(self, Mvir, redshift, mf='ST99'):
        '''
        cosmological parameters 
        '''
        rho_norm, lnMassSigmaSpl = self.RhoNorm(Mvir)
        self.massfunction(Mvir, redshift, lnMassSigmaSpl, rho_norm, mf)
 
    def RhoNorm(self, Mvir):
        '''
        Return the normaliztion for mass function
        ''' 
        lnmarr = np.linspace(np.log(Mvir * 0.99), np.log(Mvir * 1.01), 10)
        marr = np.exp(lnmarr).astype(np.float64)
        cosmo0 = CosmologyFunctions(0)
        #Byt giving m * h to sigm_m gives the sigma_m at z=0 
        sigma_m0 = np.array([cosmo0.sigma_m(m*cosmo0._h) for m in marr])
        rho_norm = cosmo0.rho_bar()
        lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0)
        return rho_norm, lnMassSigmaSpl

    def massfunction(self, Mvir, redshift, lnMassSigmaSpl, rho_norm, mf):
        '''
        p21 and Eqns. 56-59 of CS02     

        m^2 n(m,z) * dm * nu = m * bar(rho) * nu f(nu) * dnu 
        n(m,z) = bar(rho) * nu f(nu) * (dlnnu / dlnm) / m^2
        n(z) = bar(rho) * nu f(nu) * (dlnnu / dlnm) / m

        Mvir -solar unit
        nu, nuf - unitless
        rho_norm - is the rho_crit * Omega_m * h^2 in the unit of solar  Mpc^(-3)
        at redshift of z
        mass_function -  Mpc^(-3)
     
        '''
        cosmo = CosmologyFunctions(redshift)
        mass_array = np.logspace(np.log10(Mvir*0.9999), np.log10(Mvir*1.0001), 2)
        ln_mass_array = np.log(mass_array)
        ln_sigma_m_array = np.log(np.array([lnMassSigmaSpl(np.log(m)) for m in mass_array]))
        #spl = UnivariateSpline(ln_mass_array, ln_nu_array, k=3)
        #print spl.derivatives(np.log(Mvir))[1] 
        #Derivatives of dln_nu/dln_mass at ln_mass
        ln_sigma_m_ln_mass_derivative = abs((ln_sigma_m_array[1] - ln_sigma_m_array[0]) / (ln_mass_array[1] - ln_mass_array[0]))#1 for first derivate
        #print 'ln_sigma_m_ln_mass_derivative ', Mvir, ln_sigma_m_ln_mass_derivative

        if mf=='ST99':
            #This is (delta_c/sigma(m))^2. Here delta_c is slightly dependence on 
            #Omega_m across redshift. cosmo._growth is the growth fucntion
            #This means delta_c increases as a function of redshift
            #lnMassSigmaSpl returns the sigma(m) at z=0 when gives the log(Mvir) 
            delta_c_sigma_m2 = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(Mvir)) / lnMassSigmaSpl(np.log(Mvir))
            nu_d = 0.707 * delta_c_sigma_m2
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (2. * nu_d)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            nuf = 0.322 * nuf_1 * nuf_2 * nuf_3
        if mf=='CS02':
            #This is from CS02 paper not from ST99 paper. The halo mass function implimented in CHOMP is from ST99 paper
            p = 0.3
            A = 0.3222
            q = 0.75
            nuf_1 = A * (1. + (q * self.nu)**-0.3) 
            nuf_2 = (q * self.nu / (2. * np.pi))**0.5
            nuf_3 = np.exp(-q * self.nu / 2.)
            #nuf = nu*f(nu) 
            nuf = nuf_1 * nuf_2 * nuf_3
        if mf=='PS74':
            nuf = np.sqrt(1. / 2. / np.pi / self.nu) * np.exp(-self.nu / 2.)

        #print Mvir, rho_norm * cosmo._h * cosmo._h, cosmo.delta_c()/cosmo._growth, lnMassSigmaSpl(np.log(Mvir)),ln_sigma_m_ln_mass_derivative 
        mass_function = nuf * rho_norm * cosmo._h * cosmo._h * ln_sigma_m_ln_mass_derivative / Mvir
        #print Mvir, halo_bias(np.sqrt(nu)), halo_bias(nu), mass_function
        return mass_function


if __name__=='__main__':
    print MassFunction(1e15, 0, mf='ST99')
