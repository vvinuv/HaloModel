import numpy as np
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import cosmology
import convergence 
from nfw import NFW
import config

class HaloKappaSZ:
    '''
        Eq. 3.1 of Ma & Van Waerbeke 
    '''
    def __init__(self, ell):
        self.cosmo_dict = config.default_cosmo_dict
        self.ell = ell
        self.light_speed = 3.e5 #km/s

    def total_halo(self):
        self.cl_ky = self.cl_1halo_ky() + self.cl_2halo_ky()

    def concentration(self, Mvir):
        '''Ma & Van Waerbeke Eq 3.4 
           self.conc - unitless. Not that I multiplied with h=0.07 to convert 
           this in unitless
        '''
        return 5.72 / (1. + self.cosmo.redshift())**0.71 * (Mvir * self.cosmo_dict['h'] / 1e14)**-0.081
       
    def mass_func(self, Mvir, ST99=True):
        '''
        p21 and Eqns. 56-59 of CS02     

        m^2 n(m,z) * dlnm * nu = bar(rho) * nu f(nu) * dnu 
        n(m,z) = bar(rho) * nu f(nu) * (dnu / dlnm) / m^2
        n(z) = bar(rho) * nu f(nu) * (dnu / dlnm) / m

        Mvir -solar unit
        nu, nuf - unitless
        rho_bar - h^2 solar  Mpc^(-3)
        mass_function - h^2 Mpc^(-3)
        '''
        mass_array = np.logspace(np.log10(Mvir*0.99), np.log10(Mvir*1.01), 5)
        ln_mass_array = np.log(mass_array)
        nu_array = np.array([self.cosmo.nu_m(m) for m in mass_array]) 
        ln_mass_nu_spline = InterpolatedUnivariateSpline(ln_mass_array, nu_array)
        ln_mass_nu_derivative = ln_mass_nu_spline.derivatives(np.log(Mvir))[1] #1 for first derivate
        nu = self.cosmo.nu_m(Mvir)

        if ST99:
            nu_d = 0.707 * nu
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (nu_d / 2.)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            nuf = nuf_1 * nuf_2 * nuf_3        

        masss_function = nuf / nu * self.cosmo.rho_bar() * abs(ln_mass_nu_derivative) / Mvir
        return masss_function

    def integrate_mass_func_kappa_y(self, m, z):
        '''
        Eq. 3.1 of Ma & Van Waerbeke. Here I took mass of the halo (i.e. m)
        as Mvir 
        and find Rvir and concentration accordingly. I need to ask Salman 
        about this assumption  
        '''
        masss_function = self.mass_func(m)
        kappa_l = convergence.kappa_l(z, self.ell)
        y_l = convergence.sz_l(z, self.ell)
        #print masss_function, kappa_l.k_l(m), y_l.beta_y_l(m)
        return masss_function * kappa_l.k_l(m) * y_l.beta_y_l(m)

    def integrate_redshift(self, z):
        ''' 
        Eq. 3.1 of Ma & Van Waerbeke 
        dV/(dz dOmega) = c Chi^2/H(z)
        E(z) = 1/H(z). Since in the cosmology.py H0 is already divided by 
        speed of light I don't need to do that 

        1/H(z) - unit of  Mpc h^(-1) 
        Chi - unit of Mpc h^(-1)
        The unit of returned quantiy is Mpc^3 h^(-3)
        ''' 
        self.cosmo = cosmology.SingleEpoch(z, cosmo_dict=self.cosmo_dict)
        return self.cosmo.comoving_distance() * self.cosmo.comoving_distance() * self.cosmo.E(z) * integrate.quad(self.integrate_mass_func_kappa_y, 1e12, 1e16, args=(z), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]

    def cl_1halo_ky(self):
        '''
        Eq. 3.1 of Ma & Van Waerbeke 
        '''
        return integrate.quad(self.integrate_redshift, 0, 2, epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0] 

    def halo_bias(self, nu):
        '''
        Eq. 17 of Mo & White 2002
        '''
        delta_c = 1.68 #The critical value at z=0 to become spherical collapse
        return 1. + (nu**2 - 1.) / delta_c
 
    def integrate_mass_func_kappa(self, m, z):
        '''
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        nu = self.cosmo.nu_m(m)
        bias = self.halo_bias(nu)
        mass_function = self.mass_func(m)
        kappa_l = convergence.kappa_l(z, self.ell)
        return mass_function * bias * kappa_l.k_l(m) 

    def integrate_mass_func_y(self, m, z):
        '''
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        nu = self.cosmo.nu_m(m)
        bias = self.halo_bias(nu)
        mass_function = self.mass_func(m)
        y_l = convergence.sz_l(z, self.ell)
        return mass_function * bias * y_l.beta_y_l(m) 

    def integrate_redshift_linear_power(self, z):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        ''' 
        self.cosmo = cosmology.SingleEpoch(z, cosmo_dict=self.cosmo_dict)
        k = self.ell / self.cosmo.comoving_distance()
        return self.cosmo.comoving_distance() * self.cosmo.comoving_distance() * self.cosmo.E(z) * self.cosmo.linear_power(k) * integrate.quad(self.integrate_mass_func_kappa, 1e12, 1e16, args=(z), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0] * integrate.quad(self.integrate_mass_func_y,1e12, 1e16, args=(z), epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]


    def cl_2halo_ky(self):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        return integrate.quad(self.integrate_redshift_linear_power, 0, 2, epsabs=config.default_precision['epsabs'], epsrel=config.default_precision['epsrel'])[0]


if __name__=='__main__':
    ell = 100
    h = HaloKappaSZ(ell)
    print h.cl_1halo_ky()
    print h.cl_2halo_ky()
 
