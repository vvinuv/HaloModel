import os
import sys
import numpy as np
import pylab as pl
import cosmology_vinu as cosmology 
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import integrate

class MassFunction:
    '''
    This class consists of mass function using Eqs. 56-59 of Cooray & Sheth 2002
    '''
    def __init__(self, redshift):
        '''
        cosmological parameters 
        '''
        self.redshift = redshift
        self.cosmo = cosmology.SingleEpoch(self.redshift)
        #self.cosmo = cosmology.MultiEpoch(self.redshift)
        self.ShethTormen()
 
    def ShethTormen(self, mass=None, mmin=1e10, mmax=1e16, CS02=0, ST99=1, PS74=0):
        '''
        p21 and Eqns. 56-59 of CS02     

        m^2 n(m,z) * dlnm * nu = bar(rho) * nu f(nu) * dnu 
        n(m,z) = bar(rho) * nu f(nu) * (dnu / dlnm) / m^2 Mpc^(-3) Solar^-1
        Here the mass function returns the following 
        n(m,z) = bar(rho) * [nu f(nu)] * (dlnnu / dlnm) / m Mpc^(-3) 

        '''
        self.mass_array = np.logspace(np.log10(mmin), np.log10(mmax), 50) 
        self.ln_mass_array = np.linspace(np.log(mmin), np.log(mmax), 50) 

        #nu(z) = delta_sc(z)^2/sigma(m)^2 in Eq. 57 of CS02         
        self.nu = np.array([self.cosmo.nu_m(m) for m in self.mass_array])

        #ln_mass_nu_spline = InterpolatedUnivariateSpline(self.ln_mass_array, self.nu)
        #self.ln_mass_nu_derivative = np.array([ln_mass_nu_spline.derivatives(lntmp_mass)[1] for lntmp_mass in self.ln_mass_array])

        #Interpolate ln(mass) with ln(nu)
        ln_nu_ln_mass_spline = InterpolatedUnivariateSpline(self.ln_mass_array, np.log(self.nu))
        #Derivatives of dln_nu/dln_mass at ln_mass
        self.ln_nu_ln_mass_derivative = np.array([ln_nu_ln_mass_spline.derivatives(lntmp_mass)[1] for lntmp_mass in self.ln_mass_array])

        if CS02:
            #This is from CS02 paper not from ST99 paper. The halo mass function implimented in CHOMP is from ST99 paper
            p = 0.3
            A = 0.3222
            q = 0.75
            nuf_1 = A * (1. + (q * self.nu)**-0.3) 
            nuf_2 = (q * self.nu / (2. * np.pi))**0.5
            nuf_3 = np.exp(-q * self.nu / 2.)
            #nuf = nu*f(nu) 
            self.nuf = nuf_1 * nuf_2 * nuf_3
        if ST99:
            nu_d = 0.707 * self.nu
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (nu_d / 2.)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            self.nuf = nuf_1 * nuf_2 * nuf_3
        if PS74:
            self.nuf = np.sqrt(1. / 2. / np.pi / self.nu) * np.exp(-self.nu / 2.)

        #bar(rho) from Eq. 56 on CS02 
        #self.cosmo.rho_bar() gives h^2 M_sun Mpc^(-3)
        rho_bar = self.cosmo.rho_bar() / 0.7 / 0.7
        if 0:
            print 'rho_bar = %.1e'%rho_bar

        #Mass function from Eq. 56, i.e. n(m,z)
        #self.mass_function = nuf_div_nu * rho_bar * abs(self.ln_mass_nu_derivative) / self.mass_array #/ self.mass_array 
        self.mass_function = self.nuf * self.cosmo.rho_bar() * self.ln_nu_ln_mass_derivative / self.mass_array 
       
def growth_function():
    '''
    Find the growth function for the cosmology Schneider book p64, Figure 14
    D+(t) = D+(t) / D(t0) where D(t0) is the sigma_8 for redshift=0 
    '''
    cosmo = cosmology.SingleEpoch(0)
    z = np.linspace(0., 5., 50)
    scale = 1 / (1+ z)
    growth_norm = cosmo.growth_factor_eval(1.0)
    growth = np.array([cosmo.growth_factor_eval(a) for a in scale])
    pl.subplot(121)
    pl.plot(z, growth/growth_norm, color='b', label='z')
    pl.xlabel('Redshift')
    pl.ylabel('Growth')
    pl.subplot(122)
    pl.plot(scale, growth/growth_norm, color='r', label='a')
    pl.xlabel('Scale')
    pl.ylabel('Growth')
    pl.show()
    sys.exit()
#growth_function()

def p_k_redshift():
    '''
    Find P(k) Mpc^3 h^(-3)
    '''
    karr = np.logspace(-3, 1, 50)
    for z in [0, 1, 2, 3, 4]:
        cosmo = cosmology.SingleEpoch(z)
        pk = np.array([cosmo.linear_power_unsigma8(k) for k in karr])
        pl.plot(karr, pk, label='z=%.1f'%z)
    pl.xlabel('k h/Mpc')
    pl.ylabel(r'P(k) $Mpc^3/h^3$')
    pl.yscale('log')
    pl.xscale('log')
    pl.show()
    sys.exit()
#p_k_redshift()

z0 = MassFunction(0)
pl.loglog(z0.mass_array, z0.mass_function, label='z=0')

z1 = MassFunction(1)
pl.loglog(z1.mass_array, z1.mass_function, label='z=1')

#pl.figure(2)
#pl.loglog(z0.nu, z0.nuf, label='z=0')
#pl.loglog(z1.nu, z1.nuf, label='z=1')


#pl.figure(3)
#pl.loglog(z0.mass_array, z0.ln_mass_nu_derivative, label='z=0')
#pl.loglog(z1.mass_array, z1.ln_mass_nu_derivative, label='z=1')

#pl.show()
#sys.exit()
z2 = MassFunction(2)
pl.loglog(z2.mass_array, z2.mass_function, label='z=2')


pl.legend(loc=0)
pl.show()

