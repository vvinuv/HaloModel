import sys
import numpy as np
import cosmology
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import defaults


class kappa_l():
    '''
    Van Waerbeke et 2014
    '''
    def __init__(self, lens_redshift, ell):
        self.lens_redshift = lens_redshift
        self.ell = ell

        self.cosmo_dict = defaults.default_cosmo_dict
        self.cosmo = cosmology.SingleEpoch(self.lens_redshift, cosmo_dict=self.cosmo_dict)

    def Delta_c(self):
        '''
        This is the same function on nfw.py
        '''
        x = self.cosmo.omega_m() - 1.
        if 1:
            #This will return if the omega curvature is zero. It is ~1e-4 so
            #this may work
            return 18 * np.pi * np.pi + 82. * x - 39. * x * x
        elif 0:
            return 18 * np.pi * np.pi + 62. * x - 32. * x * x
        else:
            raise ValueError('Given cosmology is not implemented')

    def concentration(self, Mvir):
        '''Ma & Van Waerbeke Eq 3.4 
           self.conc - unitless. Not that I multiplied with h=0.07 to convert 
           this in unitless
        '''
        return 5.72 / (1. + self.redshift)**0.71 * (Mvir * self.cosmo_dict['h'] / 1e14)**-0.081

    def R_virial(self, Mvir):
        '''
        Eq. 23 of Dan Coe (nfw_profile.pdf)
        '''
        return (3. * Mvir / 4. / np.pi / self.cosmo.rho_crit() / Delta_c())**(1. / 3.)

    def source_dist(self, source_redshifts):
        '''
        Interpolate source redshift distribution
        '''
        ps, zs = np.histogram(source_redshifts, 100, density=True)
        zs= (zs[:-1] + zs[1:) / 2.
        self.ps_zs_interpolate = InterpolatedUnivariateSpline(zs, ps)

    def source_dist_interpolation(self, source_redshift)
        '''
        Return probabilty for a given source redshift
        '''
        return self.ps_zs_interpolate(source_redshift) * 0.001 

    def g(self, source_redshift, lens_angular_diameter_distance):
        '''
        Eq. 94 in Schneider book (In Eq. 1 of Van Waerbeke et 2014 it is 
        written as angular diameter distance but they use 1/a to convert the 
        equation to comoving distance. So Eq. 3 of Van Waerbeke et 2014 and 
        Eq. 93 of Schneider book has same dimensions)

        w = lens_angular_diameter_distance
        w' = source angular 
        g(w) = int_w^wH(p(w') fK(w'-w) / fK(w'))
        '''
        gcosmo_dict = defaults.default_cosmo_dict
        gcosmo = cosmology.SingleEpoch(source_redshift, cosmo_dict=gcosmo_dict)
        return source_dist_interpolation(source_redshift) * (cosmo.angular_diameter_distance(source_redshift) - lens_angular_diameter_distance) / cosmo.angular_diameter_distance(source_redshift)      

    def w_kappa(self):
        '''
        Eq. 1 of Van Waerbeke et 2014
        W(w) = (3/2) Omega_0 * (H0/c)^2 g(w) fK(w) / a 
        a = 1 / (1+z)
        cosmo.H0 is already divided by c and it is in the unit of h Mpc^(-1)

        Unit of W(w) = h Mpc^(-1)
        '''
        w_k = 1.5 * self.cosmo_dict["omega_m0"] * self.cosmo.H0 * self.cosmo.H0 * self.cosmo.angular_diameter_distance(self.lens_redshift) * (1. + lens_redshift) * integrate.quad(g, self.cosmo.angular_diameter_distance(self.lens_redshift), 6000, args=(lens_angular_diameter_distance))
        return w_k

    def nfw_profile(self, r, mass):
        '''
        Coe 2010 Eqs. 33-38
        '''
        Delta_c = self.Delta_c()
        conc = self.concentration(mass)
        Rvir = self.R_virial(mass)
        rs = Rvir / conc
        rho_s = Delta_c * conc**3 / 3. / (np.log(1. + conc) - conc / (1. + conc)) 
        return Delta_c * self.cosmo.rho_crit() / (r/rs) / (1. + r/rs)**2

    def nfw_integrate(self, r, mass):
        return 4 * np.pi * r * r * np.sin(self.ell * r / self.cosmo.comoving_distance(self.lens_redshift)) * self.nfw_profile(r, mass) / (self.ell * r / self.cosmo.comoving_distance(self.lens_redshift))

    def kappa_l(self, mass):
        '''
        Ma, Van Waerbeke 2015 Eq. 3.2
        k_l = (W(z)/Chi^2 rho_bar) int_0_r_vir(dr 4pi r^2 sin(l r/Chi) / (lr/Chi) * rho(r, M,z))
        '''
        Rvir = self.R_virial(mass)
        k_l = self.w_kappa() / self.cosmo.comoving_distance(self.lens_redshift) / self.cosmo.comoving_distance(self.lens_redshift) / self.cosmo.rho_bar(self.lens_redshift) * integrate.quad(self.nfw_integrate, 0, self.r_vir, args=(mass))
        return k_l

class sz_l() 
    '''
    '''
    def __init__(self, lens_redshift, mvir, rs):
        '''
        '''
        self.lens_redshift = lens_redshift
        self.mvir = mvir
        self.rs = rs
        self.sigma_T = 6.65e-29 #m^2
        self.me = 9.12e-31 #kg
        self.csq = 9.e16 #m^2/s^2
        self.msolar = 1.9889e30
        self.constant = self.sigma_T / self.me / self.msolar / self.csq #s^2/solar

        self.cosmo_dict = defaults.default_cosmo_dict
        self.cosmo = cosmology.SingleEpoch(self.lens_redshift, cosmo_dict=self.cosmo_dict)
        self.ells = (self.cosmo.comoving_distance()/(1. + lens_redshif)/rs)
        self.concentration = (5.72 / (1. + lens_redshif)**0.71) * (self.mvir / 1e14)**-0.081
        self.rvir = self.concentration * self.rs

    def up_profile():
        '''
        Universal pressure profile Eqs. 4.1 & 4.2 of Ma, Van Waerbeke 2015
        '''
        alpha_p = 0.12
        P0 = 6.41
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gamma = 0.31
        bga = (beta - gamma) / alpha
        p_x = P0 / (c500 * x)**gamma / (1. + (c500 * x)**alpha)**bga
        return p_x

    def battaglia_P0(self, z):
        return 18.1 * (M200 / 1e14)**0.154 * (1. + z)**-0.758


    def battaglia_xc(self, z):
        return 0.497 * (M200 / 1e14)**-0.00865 * (1. + z)**0.731


    def battaglia_beta(self, z):
        return 4.35 * (M200 / 1e14)**0.0393 * (1. + z)**0.415


    def battaglia_progile(self, x, M200, R200):
        '''
        Using Battaglia et al (2012)
        ''' 
        G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
        alpha = 1.0 
        gamma = -0.3
        P200 = 200. * self.cosmo.rho_crit() * self.cosmo_dict["omega_b0"] / self.cosmo_dict["omega_m0"] * G * M200 / R200
        p_e = P200 * (x / self.battaglia_xc(lens_redshift))**gamma * (1. + (x / self.battaglia_xc(lens_redshift)))**(-1*self.battaglia_beta(lens_redshift))
        return p_e

    def battaglia_integral(self, x, M200, R200):
        '''
        '''
        return x*x*np.sin(self.ell * x / self.ells) / (self.ell * x / self.ells) * battaglia_progile(x, M200, R200)

    def y_l(self, ): 
        '''
        Eq. 3.3 of Ma & Van Waerbeke 
        '''
        
        y_l = self.constants * 4. * np.pi*self.rs/self.ells/self.ells * integrate.quad(self.battaglia_integral, 0, 2, args=(M200, R200))
        return y_l

