import sys
import numpy as np
import cosmology_vinu
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import config
from nfw import NFW
import pylab as pl


class wk:
    '''
    Van Waerbeke et 2014
    '''
    def __init__(self, cosmology):
        self.lens_redshift = cosmology.redshift()
        self.cosmology = cosmology
        self.source_dist(source_redshifts=None)
        self.w_kappa()

    def dNdz_func(self, zstar=0.2, alpha=10):
        '''Mandelbaum (2008) Eq. 9 '''
        zs = np.linspace(0, 5, 100)
        dNdz = (zs/zstar)**(alpha - 1.) * np.exp(-0.5 * (zs/zstar)**2.)
        a = (dNdz[1:] + dNdz[:-1])/2.
        a_int = np.sum(a * (zs[1]-zs[0]))
        dNdz /= a_int #Normalizing so that integral(dN/dz)=1
        return dNdz, zs

    def source_dist(self, source_redshifts=None):
        '''
        Interpolate source redshift distribution
        '''
        if source_redshifts is None:
            ps, zs = self.dNdz_func(zstar=0.2, alpha=10)
        else: 
            ps, zs = np.histogram(source_redshifts, 50, range=(0,5), density=True)
            zs= (zs[:-1] + zs[1:]) / 2.
        self.zbin = zs[1] - zs[0]
        gcosmo_dict = config.default_cosmo_dict
        source_ad = np.array([cosmology_vinu.SingleEpoch(zz, cosmo_dict=gcosmo_dict).angular_diameter_distance() for zz in zs])
        source_cd = np.array([cosmology_vinu.SingleEpoch(zz, cosmo_dict=gcosmo_dict).comoving_distance() for zz in zs])

        self.zs_ad_interpolate = InterpolatedUnivariateSpline(zs, source_ad)
        #pl.plot(source_ad, ps)
        #pl.plot(source_cd, ps)
        #pl.show()
        #sys.exit()
        self.zs_ps_interpolate = InterpolatedUnivariateSpline(zs, ps)
        self.cd_ps_interpolate = InterpolatedUnivariateSpline(source_cd, ps)

    def source_dist_interpolation(self, zs):
        '''
        Return probabilty for a given source redshift
        '''
        #print  source_redshift, self.ps_zs_interpolate(source_redshift)
        return self.zs_ps_interpolate(zs) * self.zbin

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
        angular_diameter_distance = self.zs_ad_interpolate(source_redshift)
        return self.source_dist_interpolation(source_redshift) * (angular_diameter_distance - lens_angular_diameter_distance) / angular_diameter_distance     

    def w_kappa(self):
        '''
        Eq. 1 of Van Waerbeke et 2014
        W(w) = (3/2) Omega_0 * (H0/c)^2 g(w) fK(w) / a 
        a = 1 / (1+z)
        cosmo.H0 is already divided by c and it is in the unit of h Mpc^(-1)

        Unit of W(w) = h Mpc^(-1)
        '''
        self.w_k = 1.5 * self.cosmology._omega_m0 * self.cosmology.H0 * self.cosmology.H0 * self.cosmology.angular_diameter_distance() * (1. + self.cosmology.redshift()) * integrate.quad(self.g, self.cosmology.redshift(), 5, args=(self.cosmology.angular_diameter_distance()), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]



class kappa_l:
    '''
    Van Waerbeke et 2014
    '''
    def __init__(self, wk, cosmology, nfw, ell):
        self.wk = wk
        self.lens_redshift = cosmology.redshift()
        self.ell = ell
        self.cosmology = cosmology
        self.nfw = nfw
        self.k_l()

    def nfw_profile(self, r):
        '''
        Coe 2010 Eqs. 33-38
        r - Mpc h^(-1) 
        '''
        return self.nfw.rho_s / (r/self.nfw.Rs) / (1. + r/self.nfw.Rs)**2

    def nfw_integrate(self, r):
        '''
        r - Mpc h^(-1) 
        self.cosmology.comoving_distance() - Mpc h^(-1)
        '''
        return 4 * np.pi * r * r * np.sin(self.ell * r / self.cosmology.comoving_distance()) * self.nfw_profile(r) / (self.ell * r / self.cosmology.comoving_distance())

    def k_l(self):
        '''
        Ma, Van Waerbeke 2015 Eq. 3.2
        k_l = (W(z)/Chi^2 rho_bar) int_0_r_vir(dr 4pi r^2 sin(l r/Chi) / (lr/Chi) * rho(r, M,z))

        Rvir - Mpc h^(-1) 
        Rs - Mpc h^(-1) 
        self.nfw.rho_s - Msun h^3 / Mpc^3
        '''
        Rvir = self.nfw.Rvir
        self.k_l = self.wk / self.cosmology.comoving_distance() / self.cosmology.comoving_distance() / self.cosmology.rho_bar() * integrate.quad(self.nfw_integrate, 0, Rvir, epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]

class sz_l: 
    '''
    Ma & Van Waerbeke 2015 Eq. 3.3
    '''
    def __init__(self, cosmology, nfw, ell):
        '''
        '''
        self.cosmology = cosmology
        self.nfw = nfw
        self.lens_redshift = self.cosmology.redshift()
        self.ell = ell

        self.sigma_T = 6.65e-29 #m^2
        self.me = 9.12e-31 #kg
        self.csq = 9.e16 #m^2/s^2
        self.msolar = 1.9889e30 #kg
        self.constant = self.sigma_T / self.me / self.msolar / self.csq #s^2/solar

    def concentration(self, Mvir):
        '''Ma & Van Waerbeke Eq 3.4 
           self.conc - unitless. Not that I multiplied with h=0.7 to convert 
           this in unitless
        '''
        return 5.72 / (1. + self.lens_redshift)**0.71 * (Mvir / self.cosmology._h / 1e14)**-0.081

    def beta_integral(self, r, beta_y):
        '''
        '''
        return r * r * (1. + (r/self.rs)**2)**(-1.5 * beta_y)

    def beta_norm(self, beta_y):
        '''
        Eq. 6 & 14 of Waizmann & Bartelmann 2009
        '''
        fH = 0.76 #Hydrogen mass fraction
        mp = 1.67e-27 #kg
        fgas = 0.168    
        Ne = (1. + fH) * self.mass * self.msolar * fgas / 2. / mp 

        #Eq. 6 of Waizmann & Bartelmann 2009
        ne0 = Ne / 4. / np.pi / integrate.quad(self.beta_integral, 0, self.nfw.Rvir, args=(beta_y), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0] #Unit is Mpc^(-3)
 

        #Eq. 14 of Waizmann & Bartelmann 2009
        kBTe = (1. + self.lens_redshift) / beta_y * (self.nfw.Delta_c() * self.cosmology._omega_m0 / self.cosmology.rho_crit())**(1/3.) * (self.mass / self.cosmology._h / 1.e15)**(2/3.) #Unit is keV 

        P0 = ne0 * kBTe #unit is keV Mpc^(-3)
        return P0

    def beta_pressure_profile(self, r):
        '''
        Returns beta pressure profile
        I multiplied by 1e3 to make the pressure profile looks like figure 1a
        on MW15. I need to think about that factor 
       
        Unit of pressure is keV cm^{-3} 
        ''' 
        beta_y = 0.86 # Plagge et al 2010
        pe = 1.e3 * self.beta_norm(beta_y) * (1. + (r/self.rs)**2)**(-1.5 * beta_y) / (3.08e24)**3
        #print pe
        return pe

    def beta_profile_integral(self, r):
        '''
        Eq. 3.3 of Ma & Van Waerbeke
        '''
        return r*r*np.sin(self.ell * r / self.ells) / (self.ell * r / self.ells) * self.beta_pressure_profile(r)

    def beta_y_l(self, mass): 
        '''
        Eq. 3.3 of Ma & Van Waerbeke 
        '''
        self.mass = mass
        self.rs = self.nfw.Rvir / self.concentration(mass) 
        self.ells = (self.cosmology.comoving_distance()/(1. + self.lens_redshift)/self.rs)
       
        #Ma&Van says Eq. 3.3 has an upper limit of 5 times virial radius and
        # x = a(z) * r / rs 
        ulim = 1/(1.+self.lens_redshift) * 5 * self.nfw.Rvir / self.rs

        y_l = 4. * np.pi * self.rs * self.constant / self.ells / self.ells * integrate.quad(self.beta_profile_integral, 0, ulim, epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]

        return y_l 


    def universal_pressure_profile(self, r):
        '''
        Universal pressure profile Eqs. 4.1 & 4.2 of Ma, Van Waerbeke 2015
        Eqs. D1 to D3 in Komatsu 2011. However, the constants come from 
        Planck 2013, A&A, 550, 131

        According to Ma, Van Waerbeke 2015 and Komatsu 2011 the unit of pe is
        keV cm^-3
        '''
        alpha_p = 0.12
        P0 = 6.41
        c500 = 1.81
        alpha = 1.33
        beta = 4.13
        gamma = 0.31
        bga = (beta - gamma) / alpha
        h70 = self.cosmology._h / 0.7
        x = r / self.nfw.R500
        px = P0 / (c500 * x)**gamma / (1. + (c500 * x)**alpha)**bga
        pe = 1.65e-3 * self.cosmology.E0(self.lens_redshift)**(16/3.) * h70 * h70 * (self.nfw.M500 / 3e14 / h70)**(2./3. + alpha_p) * px 
        #print pe
        return pe


    def universal_pressure_integral(self, r):
        '''
        Eq. 3.3 of Ma & Van Waerbeke
        '''
        return r*r*np.sin(self.ell * r / self.ells) / (self.ell * r / self.ells) * self.universal_pressure_profile(r)



    def up_y_l(self, mass): 
        '''
        Eq. 3.3 of Ma & Van Waerbeke 
        '''
        self.mass = mass
        self.rs = self.nfw.Rvir / self.concentration(mass) 
        self.ells = (self.cosmology.comoving_distance()/(1. + self.lens_redshift)/self.rs)
       
        #Ma&Van says Eq. 3.3 has an upper limit of 5 times virial radius and
        # x = a(z) * r / rs 
        ulim = 1/(1.+self.lens_redshift) * 5 * self.nfw.Rvir / self.rs
        y_l = 4. * np.pi * self.rs * self.constant / self.ells / self.ells * integrate.quad(self.universal_pressure_integral, 0, ulim, epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]

        return y_l 

    def battaglia_P0(self, z):
        return 18.1 * (M200 / 1e14)**0.154 * (1. + z)**-0.758


    def battaglia_xc(self, z):
        return 0.497 * (M200 / 1e14)**-0.00865 * (1. + z)**0.731


    def battaglia_beta(self, z):
        return 4.35 * (M200 / 1e14)**0.0393 * (1. + z)**0.415


    def battaglia_profile(self, x, M200, R200):
        '''
        Using Battaglia et al (2012)
        ''' 
        G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
        alpha = 1.0 
        gamma = -0.3
        P200 = 200. * self.cosmology.rho_crit() * self.cosmology._omega_b0 / self.cosmology._omega_m0 * G * M200 / R200
        p_e = P200 * (x / self.battaglia_xc(self.lens_redshift))**gamma * (1. + (x / self.battaglia_xc(self.lens_redshift)))**(-1*self.battaglia_beta(self.lens_redshift))
        return p_e

    def battaglia_integral(self, x, M200, R200):
        '''
        '''
        return x*x*np.sin(self.ell * x / self.ells) / (self.ell * x / self.ells) * battaglia_progile(x, M200, R200)


    def battaglia_y_l(self, mass):
         
        y_l = self.constants * 4. * np.pi*self.rs/self.ells/self.ells * integrate.quad(self.battaglia_integral, 0, 2, args=(M200, R200), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]
        return y_l


if __name__=='__main__':
    cosmo_dict = config.default_cosmo_dict
    lens_redshift = 0.05
    ell = 100
    mass = 1e15
    cosmo = cosmology_vinu.SingleEpoch(lens_redshift, cosmo_dict=cosmo_dict)
    nfw = NFW(lens_redshift, mass, NM=False, print_mode=False)

    y = sz_l(cosmo, nfw, ell) 
    print 'Beta '
    print y.beta_y_l(mass)
    print 'UP '
    print y.up_y_l(mass)
    if 0:
        r = np.arange(1e-3, 8, 0.01)
        yr = 1e3 * y.beta_pressure_profile(r)
        pl.plot(r, yr * r * r, label='Beta')
        yr = 1e3 * y.universal_pressure_profile(r)
        pl.plot(r, yr * r * r, label='UP')
        pl.legend(loc=0)
        pl.xlabel('R (Mpc/h)')
        pl.ylabel(r'$P_e r^2 [eV cm^{-3} Mpc^2 h^{-2}]$')
        pl.show()
        sys.exit()

    w_k = wk(cosmo).w_k
    k = kappa_l(w_k, cosmo, nfw, ell)
    print k.k_l
