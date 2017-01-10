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
#from mytools import constants
#import fastcorr
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")


class CosmologyFunctions:

    """
    Calculate basic cosmological parameters.

    This is mostly using CHOMP cosmology.py script written by 
    Chris Morrison and Ryan Scranton and some changes by Vinu. 

    Most of the comments by Vinu 
    Input:
        redshift: float redshift at which to compute all 
                  cosmological values
        cosmo_dict: dictionary of float values that define 
                    the cosmology 
            (see config.py)
    """

    def __init__(self, redshift, cosmo_dict=None, with_bao=False, **kws):
        if redshift < 0.0:
            redshift = 0.0
        self._redshift = redshift

        if cosmo_dict is None:
            cosmo_dict = config.default_cosmo_dict

        self.cosmo_dict = cosmo_dict

        self._omega_m0 = cosmo_dict["omega_m0"]
        self._omega_b0 = cosmo_dict["omega_b0"]
        self._omega_l0 = cosmo_dict["omega_l0"]
        self._omega_r0 = cosmo_dict["omega_r0"]
        self._cmb_temp = cosmo_dict["cmb_temp"]
        self._h = cosmo_dict["h"]
        self._sigma_8 = cosmo_dict["sigma_8"]
        self._n = cosmo_dict["n_scalar"]
        self._w0 = cosmo_dict["w0"]
        self._wa = cosmo_dict["wa"]
        #For some reason H0 = 100 km/(s Mpc). 
        #However, h = 0.7 by using experiment. 
        #Therefore, H0/c = 100 Mpc^(-1). Since we need to 
        #write it in terms of experimatally varified 'h' 
        #then H0/c = 100 * h Mpc^(-1) = 70 Mpc^(-1)
        self.H0 = 100.0/(2.998*10**5)  # H0 / c in h Mpc^(-1)

        self._flat = True
        self._open = False
        self._closed = False

        self._k_min = config.default_limits['k_min']
        self._k_max = config.default_limits['k_max']
        #Bunn & White 1997 Eq. 29 gives delta_H
        self.delta_H = (
            1.94e-5*self._omega_m0**(-0.785 - 0.05*np.log(self._omega_m0))*
            np.exp(-0.95*(self._n - 1) - 0.169*(self._n - 1)**2))

        self._with_bao = with_bao

        self.a_growthIC = 0.001

        self._initialize_defaults()

    def _initialize_defaults(self):
        '''
        self._chi in the unit of Mpc/h
        self._growth unitless


        _sigma_norm is the normalization according to our cosmology parameter 
        sigma_8. i.e. if the given sigma_8 is 0.8 (lets say) then if you print 
        sigma_r(8.0) after this line that may not be 0.8 exactly. It may depend 
        on the power spectrum. Therefore, _sigma_norm is the ratio of the given 
        sigma_8 to the sigma_r(8.0) times the the growth factor. Growth factor 
        is 1 if z=0. After calculating _sigma_norm, _sigma_norm appropriatly 
        scales the linear power spectrum (if you need take a look at 
        linear_power() function). Therefore, printing sigma_r(8.0) gives 0.8 
        after the line _sigma_norm = _sigma_8*_growth/sigma_r(8.0)
        '''
        #For LCDM model the following work and self._w0=-1 and self._wa=0
        #Total LOS comoving distance
        self._chi = integrate.romberg(
            self.E, 0.0, self._redshift, vec_func=True,
            tol=config.default_precision["global_precision"],
            rtol=config.default_precision["cosmo_precision"],
            divmax=config.default_precision["divmax"])

        #Linear growth at a=1, i.e. present time
        growth_scale_1 = self.growth_factor_eval(1.0)

        #print 1, growth_scale_1
        #Linear growth at a=1/1+z
        a = 1.0 / (1.0 + self._redshift)
        growth = self.growth_factor_eval(a)
        
        #print 2, growth
        #Growth rate present to redshift z
        self._growth = growth / growth_scale_1

        #print 3, self._growth
        #print 4, self._sigma_8
        #sigma_8 at redshift z
        self._sigma_8z = self._sigma_8*self._growth 
   
        #print 5, self._sigma_8z
        #sigma_8 at redshift z. This is temporary self._sigma_norm. The correct 
        #self._sigma_norm = self._sigma_8*self._growth/self.sigma_r(8.0)  
        self._sigma_norm = 1.0

        #print 6, self._sigma_norm
        #print 'Previous ', self.sigma_r(8.0)
        #print self.sigma_r(8.0) #This does not match with sigma_8 on config.py

        #The following line was given in CHOMP and it doesn't give the correct
        #mass function.
        self._sigma_norm = self._sigma_8*self._growth/self.sigma_r(8.0)

        #print 'Previous ', self.sigma_r(8.0)

        #I think I am wrong here. Not having one more self._growth may be right
        #I checked kind of power spectrum generated by CAMB and by this code
        #Without two self._growth makes sense 
        #I added one more self._growth in the previous line and it kind of 
        #producing the mass function
        #print 4, self._sigma_8
        #self._sigma_norm = self._sigma_8*self._growth*self._growth/self.sigma_r(8.0)
        #print 7, self._sigma_norm
        #print  self._sigma_norm 
        #print 'Last ', self.sigma_r(8.0)
        #print self.sigma_r(8.0) #This matches with sigma_8 on config.py


    def E(self, redshift):
        """
        1/H(z). Used to compute cosmological distances.

        Args:
            redshift: redshift at which to compute E.
        Returns:
            Float value of 1/H at z=redshift.
            1/H(redshift) unit of Mpc h^(-1) 

        This function is similar to Hz in nfw.py but here the unit is Mpc h^(-1)
        (see the top of this class where the unit of H0 is h/Mpc and see the 
        details) instead of pc on nfw.py
        """
        return 1.0/(self.H0*np.sqrt(self.E0(redshift)))

    def E0(self, redshift):
        """
        (H(z)/H0)^2 aka Friedmen's equation. Used to compute various 
        redshift dependent cosmological factors.

        Args:
            redshift: redshift at which to compute E0.
        Returns:
            Float value of (H(z)/H0)^2 at z=redshift
            I think it returns unitless

        Only the 'if' part work here for LCDM model. This equation can also 
        found from Bryan & Norman 1998 (i.e. nfw.py). nfw.py script I write it 
        in redshift but this function convert that to scale factor 'a'
        """
        a = 1.0/(1.0 + redshift)
        return (self._omega_l0 + self._omega_m0/(a*a*a) + self._omega_r0/(a*a))
 
    def w(self, redshift):
        """
        Redshift dependent Dark Energy w

        Args:
            redshift: float array redshift
        Returns:
            float array Dark Energy w
            I think it returns unitless

        """
        a = 1.0/(1 + redshift)
        return self._w0 + self._wa*(1 - a)


    def _growth_integrand(self, a):
        """
        Integrand for growth factor as a function of scale

        Args:
            a: float array scale factor
        Returns:
            Integral of Eq. 4 of Eisenstein 97

        This returns unitless quantity. E0=(H(z)/H0)^2 is unitless  
        E = 1 / H(z)
        """
        redshift = 1.0/a - 1.0
        #return ((1.0 + redshift)/np.sqrt(self.E0(redshift)))**3
        return ((1.0 + redshift) * self.E(redshift))**3

    def _growth_factor_integral(self, a):
        '''
        Integral of Eq. 4 of Eisenstein 97

        E = 1 / H(z)
        This returns unitless quantity.
        '''
        redshift = 1. / a - 1.
        growth = integrate.romberg(
            self._growth_integrand, 1e-4, a, vec_func=True,
            tol=config.default_precision["global_precision"],
            rtol=config.default_precision["cosmo_precision"],
            divmax=config.default_precision["divmax"])
        #growth *= 2.5*self._omega_m0*np.sqrt(self.E0(redshift))
        growth *= (2.5*self._omega_m0 / self.E(redshift))
        return growth

    def growth_factor_eval(self, a):
        """
        Evaluate the linear growth factor for vector argument redshift.
            It returns unitless quantity
        """
        # a = 1. / (1. + redshift)
        if self._w0 == -1.0 and self._wa == 0.0:
            try:
                growth = np.zeros(len(a), dtype='float64')
                for idx in xrange(a.size):
                    growth[idx] = self._growth_factor_integral(a[idx])
            except TypeError:
                growth = self._growth_factor_integral(a)

        return growth

    def comoving_distance(self):
        """
        Comoving distance at redshift self._redshif in units Mpc/h.
        
        Returns:
            Float comoving distance
        """
        return self._chi

    def luminosity_distance(self):
        """
        Luminosity distance at redshift self._redshif in units Mpc/h.

        Returns:
            Float Luminosity Distance
        """
        return (1.0 + self._redshift) * self._chi

    def angular_diameter_distance(self):
        """
        Angular diameter distance at redshift self._redshif in 
        units Mpc/h.

        Returns:
            Float Angular diameter Distance
        """
        return self._chi/(1.0 + self._redshift)

    def redshift(self):
        """
        Getter for internal value _redshift

        Returns:
            float redshift
        """
        return self._redshift

    def growth_factor(self):
        """
        Linear growth factor, normalized to unity at z = 0.

        Returns:
            float growth factor
            It returns unitless
        """
        return self._growth

    def omega_m(self):
        """
        Total mass denisty at redshift z = _redshift.

        Returns:
            It returns unitless 

        float total mass density relative to 
        E0 = (H(z)/H0)^2 =omega_L + omega_m/a^3 + omega_k/a^2. 
        From Bryan&Norman paper it says that it is relative to critical 
        universe where Lambda=0 and curvature vasishes and the universe 
        density is equals to the critical density. 
        Read Schneider book p47. It ranges 0.3 to 1 at very high redshift  
        """
        return self._omega_m0*(1.0 + self._redshift)**3/self.E0(self._redshift)

    def omega_l(self):
        """
        Dark Energy denisty at redshift z = _redshift.

        Returns:
            float dark energy density relative to critical
            It returns unitless
        """
        return self._omega_l0/self.E0(self._redshift)

    def delta_c(self):
        """
        Over-density threshold for linear, spherical collapse a z=_redshift.

        Returns:
            Float threshold. I think it returns unitless 

        delta_c is the delta_sc in CS02 which 1.68 at z=0

        delta_c is unitless
        Fitting function taken from NFW97. This also gives 1.68 from CS02 when 
        usin eqn 53 with z=0.

        The weak dependence of Omega_m in p21 of CS02 or Eq. A14 
        (i.e. delta_crit^0(Omega)) of Navarro 1997
        """
        delta_c = 0.15*(12.0*np.pi)**(2.0/3.0)
        delta_c *= self.omega_m()**0.0055

        return delta_c

    def rho_crit(self):
        """
        Critical density in h^2 solar masses per cubic Mpc.

        rho_crit = 3 H(z)^2 / (8 pi G)
        Returns:
            float critical density (Msun h^2 Mpc^(-3))
        This is basically multiplied by h^2 using equation 75 of Schneider book. 
        This I have checked that with cosmolopy with returns density in 
        Solar mass per cubic megaparsec using redshift=0

        Unit of rho_crit is h^2 Msun Mpc^(-3)
        """
        return (1.879/(1.989)*3.086**3*1e10 * self.E0(self._redshift))

    def rho_bar(self):
        """
        Matter density in h^2 solar masses per cubic Mpc.

        Returns:
            float average matter desnity (Msun h^2 Mpc^(-3))
        """
        return self.rho_crit()*self.omega_m()

    def BryanDelta(self):
        '''
        Delta_c which include virial mass from Bryan & Norman 1998  
        '''
        x = self.omega_m() - 1.
        if 1: #self.OmegaR == 0:
            return 18 * np.pi * np.pi + 82. * x - 39. * x * x
        elif 0: #self.OmegaL == 0:
            return 18 * np.pi * np.pi + 62. * x - 32. * x * x
        else:
            raise ValueError('Given cosmology is not implemented')

    def _eh_transfer(self, k):
        """
        Eisenstein & Hu (1998) fitting function without BAO wiggles
        (see eqs. 26,28-31 in their paper)
        http://wwwmpa.mpa-garching.mpg.de/~komatsu/CRL/powerspectrum/nowiggle/eisensteinhu/eisensteinhu.f90
        Args:
            XXX: k: float in unit of [h/Mpc]
               array wave number at which to compute power spectrum.
        Returns:
            XXX: It returns unitless
            float array Transfer function T(k).
        """
 
        #Eq. 28. Unit of k = h/Mpc and q=k/(hMpc^-1) Theta^2_2.7/Gamma
        theta2 = (self._cmb_temp/2.7)**2. # Temperature of CMB_2.7

        Omh2 = self._omega_m0*self._h**2
        Omb2 = self._omega_b0*self._h**2
        omega_ratio = self._omega_b0/self._omega_m0
        s = 44.5*np.log(9.83/Omh2)/np.sqrt(1+10.0*(Omb2)**(3/4.))
        alpha = (1 - 0.328*np.log(431.0*Omh2)*omega_ratio +
                 0.38*np.log(22.3*Omh2)*omega_ratio**2)
        Gamma_eff = self._omega_m0*self._h*(
            alpha + (1-alpha)/(1+0.43*k*s)**4)
        q = k*theta2/Gamma_eff
        L0 = np.log(2*np.e + 1.8*q)
        C0 = 14.2 + 731.0/(1+62.5*q)
        return L0/(L0 + C0*q*q)


    def transfer_function(self, k):
        """
        Function for returning the CMB transfer function. Class variable 
        with_bao determines if the transfer function is the E+H98 fitting
        function with or without BAO wiggles.

        Args:
            XXX: k [h/Mpc]: float array wave number at which to compute 
            the transfer function
        Returns:
            XXX: float array CMB transfer function unitless
        """
        return self._eh_transfer(k)
    
    def delta_k_unsigma8(self, k):
        """
        k^3*P(k)/2*pi^2: dimensionless linear power spectrum. 

        Args:
            k: float array Wave number at which to compute power spectrum.
        Returns:
            float array dimensionless linear power spectrum k^3*P(k)/2*pi^2

        Eq. 26 of CS02 give this expression for delta_k 
        """
        delta_k = (self.delta_H**2*(k/self.H0)**(3 + self._n)*
                   self.transfer_function(k)**2) #/self._h
        return delta_k* self._growth*self._growth


    def linear_power_unsigma8(self, k):
        """
        Linear power spectrum P(k) in units Mpc^3/h^3 not normalized by sigma_8 

        Args:
            k: float array Wave number at which to compute power spectrum.
        Returns:
            float array linear power spectrum P(k)
        Eq. 26 of CS02 give this expression give P_lin(k)
        """
        return np.where(k > 1e-16, 
                        2.0*np.pi*np.pi*self.delta_k_unsigma8(k)/(k*k*k), 
                        1e-16)

 
    def delta_k(self, k):
        """
        k^3*P(k)/2*pi^2: dimensionless linear power spectrum normalized to by
        sigma_8 at scale 8 Mpc h^(-1). This means that if the sigma_8 = 0.78
        in config.py it will give 0.78 when I run sigma_r(8) at z=0. i.e.
      
        se = CosmologyFunctions(0)
        print se.sigma_r(8)

        Args:
            k: float array Wave number at which to compute power spectrum.

            XXX: k in unit of [h/Mpc]
        Returns:
            float array dimensionless linear power spectrum k^3*P(k)/2*pi^2

            XXX: return value Unitless!            
        Eq. 26 of CS02 give this expression for delta_k 
        """
        delta_k = (self.delta_H**2*(k/self.H0)**(3 + self._n)*
                   self.transfer_function(k)**2) #/self._h
        return delta_k*(
            self._growth*self._growth*self._sigma_norm*self._sigma_norm)

    def linear_power(self, k):
        """
        Linear power spectrum P(k) in units Mpc^3/h^3 normalized to by
        sigma_8 at scale 8 Mpc h^(-1). This means that if the sigma_8 = 0.78
        in config.py it will give 0.78 when I run sigma_r(8) at z=0. i.e.
      
        se = CosmologyFunctions(0)
        print se.sigma_r(8)

        Args:
            XXX: k: [h/Mpc] float array Wave number at which to compute 
            power spectrum.
        Returns:
            XXX: [Mpc/h]^3 float array linear power spectrum P(k)
        Eq. 26 of CS02 give this expression give P_lin(k)
        """
        return np.where(
            k > 1e-16,
            2.0*np.pi*np.pi*self.delta_k(k)/(k*k*k), 1e-16)

    def sigma_r(self, scale):
        """
        RMS power on scale in [Mpc/h]. sigma_8 is defined as sigma_r(8.0).
        
        Args:
            XXX: scale: unit [Mpc/h] length scale on which to compute RMS power
        Returns:
            XXX: Unitless float RMS power at scale
        These are eqn 18 of CS02 and read p10 
        """
        k_min = self._k_min
        k_max = self._k_max
        needed_k_min = 1.0/scale/10.0
        needed_k_max = 1.0/scale*14.0662 ### 4 zeros of the window function
        if (needed_k_min <= k_min and
            needed_k_min > self._k_min/100.0):
            k_min = needed_k_min
        elif (needed_k_min <= k_min and
              needed_k_min <= self._k_min/100.0):
            k_min = self._k_min/100.0
            # print "In cosmology.CosmologyFunctions.sigma_r:"
            # print "\tWARNING: Requesting scale greater than k_min."
            # print "\tExtrapolating to k_min=",k_min
        if (needed_k_max >= k_max and
              needed_k_max < self._k_max*100.0):
            k_max = needed_k_max
        elif (needed_k_max >= k_max and
              needed_k_max >= self._k_max*100.0):
            k_max = self._k_max*100.0
            # print "In cosmology.CosmologyFunctions.sigma_r:"
            # print "\tWARNING: Requesting scale greater than k_max."
            # print "\tExtrapolating to k_max=",k_max

        sigma2 = integrate.romberg(
            self._sigma_integrand, np.log(k_min),
            np.log(k_max), args=(scale,), vec_func=True,
            tol=config.default_precision["global_precision"],
            rtol=config.default_precision["cosmo_precision"],
            divmax=config.default_precision["divmax"])
        sigma2 /= 2.0*np.pi*np.pi

        return np.sqrt(sigma2)

    def _sigma_integrand(self, ln_k, scale):
        """
        Integrand to compute sigma_r

        Args:
            ln_k: float array natural log of wave number
            scale: float scale in Mpc/h
        Returns:
            These are eqn 18 of CS02 and read p10 

        """
        k = np.exp(ln_k)
        kR = scale*k

        W = 3.0*(np.sin(kR)/kR**3-np.cos(kR)/kR**2)

        return self.linear_power(k)*W*W*k*k*k
        #return self.linear_power_unsigma8(k)*W*W*k*k*k

    def sigma_m(self, mass):
        """
        RMS power on scale subtended by total mean mass in solar masses/h.

        Args:
            mass: float mean mass at which to compute RMS power
        Returns:
            float fluctuation for at mass

        These are eqn 18 of CS02 and read p10 

        """
        #scale = self._h*(3.0*mass/(4.0*np.pi*self.rho_bar()))**(1.0/3.0)
        scale = (3.0*mass/(4.0*np.pi*self.rho_bar()))**(1.0/3.0)

        return self.sigma_r(scale)

    def nu_r(self, scale):
        """
        Ratio of (delta_c/sigma(R))^2.

        Args:
            scale: float length scale on which to compute nu
        Returns:
            float normalized fluctuation
        """
        sqrt_nu = self.delta_c()/self.sigma_r(scale)
        return sqrt_nu*sqrt_nu

    def nu_m(self, mass):
        """
        Ratio of (delta_c/sigma(M))^2. Used as the integration variable for
        halo.py and mass.py. Determains at which mean mass a halo has
        collapsed.

        Args:
            mass: float mean mass at which to compute nu
        Returns:
            float normalized fluctuation

        This is nu = delta_sc^2/sigma(m)^2 in Eq. 57 of CS02 
        """
        sqrt_nu = self.delta_c()/ self._growth /self.sigma_m(mass)
        if mass > 1e55:
            print mass, self.delta_c(), self.delta_c()/self._growth, self._growth, sqrt_nu*sqrt_nu, self.sigma_m(mass)

        return sqrt_nu*sqrt_nu


@jit(nopython=True)
def integrate_rad(theta_radian, xi, ell, dlnt):
    cl = 0.0
    for ri, t in enumerate(theta_radian):
        cl += (t*t*np.sin(t*ell)*xi[ri]/t/ell)
    return cl*dlnt*2.*np.pi

@jit(nopython=True)
def integrate_ell(larr, cl, theta_rad, dlnl):
    xi = 0.0
    for i, l in enumerate(larr):
        xi += (l*l*np.sin(l*theta_rad)*cl[i]/l/theta_rad)
    return xi *dlnl / 2 /np.pi

@jit(nopython=True)
def integrate_splrad(theta_radian_spl, xi_spl, ell, dt):
    cl = 0.0
    for ri, t in enumerate(theta_radian_spl):
        cl += (t*np.sin(t*ell)*xi_spl[ri]/t/ell)
    return cl*dt*2.*np.pi

@jit(nopython=True)
def integrate_splell(larr_spl, cl_spl, theta_rad, dl):
    xi = 0.0
    for i, l in enumerate(larr_spl):
        xi += (l*np.sin(l*theta_rad)*cl_spl[i]/l/theta_rad)
    return xi *dl / 2 /np.pi


@jit(nopython=True)
def integrate_zdist(gangsarr, gchisarr, angl, Ns):
    #chi - in Eq. 2 of Waerbeke
    #fchi - chi in photometric redshift distribution 
    #angarr = np.linspace(chi, angH, 100)
    gint = 0.0
    i = 0
    for N in Ns:
        gint += ((gangsarr[i] - angl) * N / gangsarr[i]) 
        i += 1
    #gint *= (gangarr[1] - gangarr[0])
    gint *= (gchisarr[1] - gchisarr[0])
    return gint

@jit(nopython=True)
#def integrate_chi(zarr, chiarr, angarr, pkarr, Darr, constk, chiH, Ns):
def integrate_chi(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns):
    aarr = 1. / (1. + zlarr)
    Wk = constk * anglarr / aarr
    cl = 0.0
    for i, angl in enumerate(anglarr):
        gangsarr = angsarr[i+1:]
        gchisarr = chisarr[i+1:]
        gw = integrate_zdist(gangsarr, gchisarr, angl, Ns[i+1:])
        if gw <= 0:
            gw = 0.
        cl += (Wk[i] * Wk[i] *  gw * gw * pkarr[i] * Darr[i] * Darr[i] / angl / angl)
        #print Wk[i], gw, pkarr[i],Darr[i]
    cl *= (chilarr[1] - chilarr[0])
    #Wy = consty * bg * Te * ne
    return cl

if __name__=='__main__':
    #Write variables
    zmax_l = 1.0
    omega_m = 0.264
    h = 0.71
    H0 = h * 100 #(km/s)/Mpc
    #constants = constants()
    #c = constants.c_km_s
    c = 2.998e5 #km/s
    mpctocm = 3.085677581e24
    kB_kev_K = 8.617330e-8 #keV k^-1
    sigma_t_cm = 6.6524e-25 #cm^2
    rest_electron_kev = 511 #keV
    constk = 3. * omega_m * (H0 / c)**2. / 2. #Mpc^-2
    print constk
    consty = kB_kev_K  * sigma_t_cm / rest_electron_kev #cm^-2 
    const = constk * consty

    #Source redshift distribution
    f = np.genfromtxt('source_distribution.txt')
    zsarr = f[:,0][1:]
    Ns = f[:,1][1:]
    
    minz = zsarr.min()
    maxz = zsarr.max()
    zsspl = InterpolatedUnivariateSpline(zsarr, Ns, k=1)


    #These parameters will do nothing at this moment 
    compute = 1 #Whether the profile should be computed 
    fwhm = 10 #arcmin Doesn't work now
    rmin = 1 #Inner radius in arcmin 
    rmax = 1e2 #Outer radius in arcmin
    space = 10 #linear space between two points
    #Stop

    kmin = 1e-5
    kmax = 1e4
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
    karr = np.exp(lnkarr).astype(np.float64)

    chisarr, Darr = [], []
    for zi in zsarr:
        cosmo = CosmologyFunctions(zi)
        chisarr.append(cosmo.comoving_distance())
    chisarr = np.array(chisarr) / h #Mpc
    #pl.scatter(zarr, chiarr, c='k')

    chizspl = InterpolatedUnivariateSpline(chisarr, zsarr, k=1)
    chisarr = np.linspace(chisarr.min(), chisarr.max(), 1000)
    zsarr = chizspl(chisarr)
    conl = (zsarr > 0.0) & (zsarr < zmax_l)
    zlarr = zsarr[conl]
    chilarr = chisarr[conl]

    anglarr = chilarr / (1. + zlarr)
    angsarr = chisarr / (1. + zsarr)

    #pl.scatter(zarr, chiarr, c='r')
    #pl.show()

    #Interpolating source at individual redshift points
    Ns = zsspl(zsarr)

    for zi in zlarr:
        cosmo = CosmologyFunctions(zi)
        Darr.append(cosmo._growth)
    Darr = np.array(Darr)


	#pk_arr = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
        #pl.loglog(karr, pk_arr)
    #pl.show()
    #sys.exit() 
    #zint normalization of Eq 2 of Waerbeke
    ##zint = np.sum(Ns * (angarr[1] - angarr[0]))
    zint = np.sum((Ns[:-1] + Ns[1:]) * (chisarr[1] - chisarr[0])) / 2.
    Ns /= zint
    #pl.plot(zarr, Ns)
    #pl.yscale('log')
    #pl.show()

    cosmo = CosmologyFunctions(0.)

    #No little h
    pk_arr_z0 = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
    pkspl_z0 = InterpolatedUnivariateSpline(karr/cosmo._h, pk_arr_z0)

    fpk = np.genfromtxt('/media/luna1/vinu/software/FrankenEmu/output_pk/0_0.005.dat')
    karr, pk_arr_z0 = fpk[:,0], fpk[:,1]
    pkspl_z0 = InterpolatedUnivariateSpline(karr, pk_arr_z0)
    #pl.savetxt('pk_z0.dat', np.transpose((karr, pk_arr_z0)))
    #pl.loglog(karr, pk_arr_z0)
    #pl.show()
    #larr = np.linspace(1, 3000, 50)
    larr = np.arange(1, 4000, 10)
    larr = np.logspace(np.log10(1), np.log10(3000), 50)
    print larr 

    angH = angsarr.max()
    angL = angsarr.min()
    chiH = chisarr.max()
    chiL = chisarr.min()
 
    print angL, angH, const, zint
    cl_ky = [] 
    cl_kk = [] 
    for l in larr:
        #print l/angarr
        pkarr = pkspl_z0(l/anglarr)
        #pl.loglog(angarr, pkarr)
        #pl.show()
        #sys.exit()
        cl_kk.append(integrate_chi(zlarr, chilarr, chisarr, anglarr, angsarr, pkarr, Darr, constk, chiH, Ns))
    cl_kk = np.array(cl_kk)
    print cl_kk #* larr * (1+larr) / 2. / np.pi 
    np.savetxt('kk_power.txt', np.transpose((larr, cl_kk)))
    #np.savetxt('pk_%.1f.txt'%redshift, np.transpose((karr, pk_arr))) 
    pl.loglog(larr, cl_kk)
    pl.show()
    sys.exit()

