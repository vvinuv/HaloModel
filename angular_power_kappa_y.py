import numpy as np
import scipy import integrate
import cosmology
import mass_function 
import convergence 


class onehalo:
    '''
        Eq. 3.1 of Ma & Van Waerbeke 
    '''
    def __init__(self, redshift):
        self.redshift = redshift
        self.cosmo_dict = defaults.default_cosmo_dict

    def mass_func(self, mass):
        '''
        Eq. 3.1 of Ma & Van Waerbeke 
        '''
        mass_array = np.array([mass*0.99, mass, mass*1.01])
        ln_mass_array = np.log(mass_array)
        nu_array = np.array([self.cosmo.nu_m(m) for m in mass_array]) 
        ln_mass_nu_spline = InterpolatedUnivariateSpline(ln_mass_array, nu_array)
        ln_mass_nu_derivative = ln_mass_nu_spline.derivatives(mass)[1]
        nu = self.cosmo.nu_m(mass)

        if ST99:
            nu_d = 0.707 * nu
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (nu_d / 2.)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            nuf = nuf_1 * nuf_2 * nuf_3        

        nuf_div_nu = self.nuf / self.nu
        rho_bar = self.cosmo.rho_bar() / 0.7 / 0.7
        masss_function = nuf_div_nu * rho_bar * abs(self.ln_mass_nu_derivative) / mass
        return masss_function

    def integrate_mass_func_kappa_y(self, mass):
        '''
        Eq. 3.1 of Ma & Van Waerbeke 
        '''
        masss_function = self.mass_func(mass)
        kappa_l = convergence.kappa_l(mass, z)
        y_l = convergence.sz_l(mass, z)
        return masss_function * kappa_l * y_l

    def integrate_redshift(self, z):
        ''' 
        Eq. 3.1 of Ma & Van Waerbeke 
        ''' 
        self.cosmo = cosmology.SingleEpoch(z, cosmo_dict=self.cosmo_dict)
        return self.cosmo.comoving_distance() * self.cosmo.comoving_distance() * self.cosmo.E(z) * 3e5 * integrate.quad(integrate_mass_func_kappa_y, 1e12, 1e16)

    def cl_1halo_ky(self):
        '''
        Eq. 3.1 of Ma & Van Waerbeke 
        '''
        return integrate.quad(integrate_redshift, 0, 2) 

    def integrate_mass_func_kappa(self, mass):
        '''
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        masss_function = self.mass_func(mass)
        kappa_l = convergence.kappa_l(mass, z)
        return masss_function * kappa_l 

    def integrate_mass_func_y(self, mass):
        '''
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        masss_function = self.mass_func(mass)
        y_l = convergence.sz_l(mass, z)
        return masss_function * y_l 

    def integrate_redshift_linear_power(self, z):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        ''' 
        self.cosmo = cosmology.SingleEpoch(z, cosmo_dict=self.cosmo_dict)
        k = ell / self.cosmo.comoving_distance()
        return self.cosmo.comoving_distance() * self.cosmo.comoving_distance() * self.cosmo.E(z) * 3e5 * self.cosmo.linear_power(k) * integrate.quad(integrate_mass_func_kappa, 1e12, 1e16) * integrate.quad(integrate_mass_func_y, 1e12, 1e16)


    def cl_2halo_ky(self):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        return integrate.quad(integrate_redshift_linear_power, 0, 2)

    self.cl_ky = self.cl_1halo_ky() + self.cl_2halo_ky()
