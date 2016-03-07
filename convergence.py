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
    def __init__(self, lens_redshift, r_vir, rs, ell):
        self.lens_redshift = lens_redshift
        self.r_vir = r_vir
        self.rs = rs
        self.ell = ell

        self.cosmo_dict = defaults.default_cosmo_dict
        self.cosmo = cosmology.SingleEpoch(self.lens_redshift, cosmo_dict=self.cosmo_dict)

    def source_dist(self, source_redshifts):
        ps, zs = np.histogram(source_redshifts, 100, density=True)
        zs= (zs[:-1] + zs[1:) / 2.
        self.ps_zs_interpolate = InterpolatedUnivariateSpline(zs, ps)

    def source_dist_interpolation(self, source_redshift)
        return self.ps_zs_interpolate(source_redshift) 

    def g(self, source_redshift, lens_angular_diameter_distance):
        '''
        Eq. 94 in Schneider book (In Eq. 1 of Van Waerbeke et 2014 it is written as angular diameter distance but they use 1/a to convert the equation to comoving distance. So Eq. 3 of Van Waerbeke et 2014 and Eq. 93 of Schneider book has same dimensions)
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
        self.cosmo.H0 is already divided by c and it is in the unit of h Mpc^(-1)
        '''
        w_k = 1.5 * self.cosmo_dict["omega_m0"] * self.cosmo.H0 * self.cosmo.H0 * self.cosmo.angular_diameter_distance(self.lens_redshift) * (1. + lens_redshift) * integrate.quad(g, self.cosmo.angular_diameter_distance(self.lens_redshift), 6000)
        return w_k

    def nfw_profile(self, r):
        '''
        Coe 2010 Eq. 33
        '''
        return self.cosmo.delta_c / (r/self.rs) / (1 + r/self.rs)**2

    def nfw_integrate(self, r):
        return 4 * np.pi * r * r * np.sin(self.ell * r / self.cosmo.comoving_distance(self.lens_redshift)) * self.nfw_profile(r, self.rs) / (self.ell * r / self.cosmo.comoving_distance(self.lens_redshift))

    def kappa_l(self):
        '''
        Ma, Van Waerbeke 2015 Eq. 3.2
        k_l = (W(z)/Chi^2 rho_bar) int_0_r_vir(dr 4pi r^2 sin(l r/Chi) / (lr/Chi) * rho(r, M,z))
        '''
        k_l = self.w_kappa() / self.cosmo.comoving_distance(self.lens_redshift) / self.cosmo.comoving_distance(self.lens_redshift) / self.cosmo.rho_bar(self.lens_redshift) * integrate.quad(self.nfw_integrate, 0, self.r_vir)

        return k_l

class y_l() 
    '''
    '''
    def __init__(self):
        '''
        '''
        self.lens_redshift = lens_redshift
        self.rs = rs
        self.sigma_T = 6.65e-29 #m^2
        self.me = 9.12e-31 #kg
        self.csq = 9.e16 #m^2/s^2
        self.msolar = 1.9889e30
        self.constant = self.sigma_T / self.me / self.msolar / self.csq #s^2/solar

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




        




