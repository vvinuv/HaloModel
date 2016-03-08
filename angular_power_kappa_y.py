import numpy as np
import scipy import integrate
import cosmology
import mass_function 
import convergence 


class HaloKappaSZ:
    '''
        Eq. 3.1 of Ma & Van Waerbeke 
    '''
    def __init__(self, ell):
        self.cosmo_dict = defaults.default_cosmo_dict
        self.ell = ell
        self.light_speed = 3.e5 #km/s

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
        Rvir = self.R_virial(Mvir)
        conc = se;f.concentration(Mvir)

        mass_array = np.array([Mvir*0.99, Mvir, Mvir*1.01])
        ln_mass_array = np.log(mass_array)
        nu_array = np.array([self.cosmo.nu_m(m) for m in mass_array]) 
        ln_mass_nu_spline = InterpolatedUnivariateSpline(ln_mass_array, nu_array)
        ln_mass_nu_derivative = ln_mass_nu_spline.derivatives(Mvir)[1] #1 for first derivate
        nu = self.cosmo.nu_m(Mvir)

        if ST99:
            nu_d = 0.707 * nu
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (nu_d / 2.)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            nuf = nuf_1 * nuf_2 * nuf_3        

        masss_function = nuf / nu * self.cosmo.rho_bar() * abs(ln_mass_nu_derivative) / Mvir
        return masss_function, Rvir, conc

    def integrate_mass_func_kappa_y(self, m, z, ell):
        '''
        Eq. 3.1 of Ma & Van Waerbeke. Here I took mass of the halo (i.e. m)
        as Mvir 
        and find Rvir and concentration accordingly. I need to ask Salman 
        about this assumption  
        '''
        masss_function, Rvir, conc = self.mass_func(m)
        kappa_l = convergence.kappa_l(z, m, Rvir, conc, ell)
        y_l = convergence.sz_l(z, m, Rvir, conc, ell)
        return masss_function * kappa_l * y_l

    def integrate_redshift(self, z, ell):
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
        return self.cosmo.comoving_distance() * self.cosmo.comoving_distance() * self.cosmo.E(z) * integrate.quad(integrate_mass_func_kappa_y, args=(z, ell), 1e12, 1e16)

    def cl_1halo_ky(self, ell):
        '''
        Eq. 3.1 of Ma & Van Waerbeke 
        '''
        return integrate.quad(integrate_redshift, 0, 2, args=(ell)) 

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
        nu = self.ps_cosmo.nu_m(m)
        bias = self.halo_bias(nu)
        masss_function = self.mass_func(m)
        kappa_l = convergence.kappa_l(m, z)
        return masss_function * bias * kappa_l 

    def integrate_mass_func_y(self, m, z):
        '''
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        nu = self.ps_cosmo.nu_m(m)
        bias = self.halo_bias(nu)
        mass_function = self.mass_func(m)
        y_l = convergence.sz_l(m, z)
        return masss_function * bias * y_l 

    def integrate_redshift_linear_power(self, z, ell):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        ''' 
        self.ps_cosmo = cosmology.SingleEpoch(z, cosmo_dict=self.cosmo_dict)
        k = ell / self.ps_cosmo.comoving_distance()
        return self.ps_cosmo.comoving_distance() * self.ps_cosmo.comoving_distance() * self.ps_cosmo.E(z) * self.ps_cosmo.linear_power(k) * integrate.quad(integrate_mass_func_kappa, args=(z), 1e12, 1e16) * integrate.quad(integrate_mass_func_y, args=(z), 1e12, 1e16)


    def cl_2halo_ky(self, ell):
        ''' 
        Eq. 3.5 of Ma & Van Waerbeke 
        '''
        return integrate.quad(integrate_redshift_linear_power, 0, 2, arg=(ell))

    self.cl_ky = self.cl_1halo_ky() + self.cl_2halo_ky()
