#Smoothing FWHM of kappa and sz 
fwhm_y = 0#10. #arcmin
fwhm_k = 0#1.0 #arcmin

#Should be use either kk or yy or ky
kk = False #Autocorrelation of WL
yy = True #Autocorrelation of tSZ
ky = False #Cross correlation of WL x tSZ
zsfile = 'source_distribution_new_z0p1.txt'

#Battaglia pressure parameters for Delta=200
P01 = 18.1
P02 = 0.154
P03 = -0.758
xc1 = 0.497
xc2 = -0.00865
xc3 = 0.731
beta1 = 4.35
beta2 = 0.0393
beta3 = 0.415

# Mass function
MF = 'Tinker' #Tinker or Bocquet
MassToIntegrate = 'virial' #virial or m200c or m200m (m200m is not working)
MassDef = 200 #Mass definition in Tinker MF in average mass density (critical * Omega_m(z)) if MassToIntegrate = 'virial'.Should be either 200 or 400. When using 400 it seems the power spectrum is more agreeable to Battaglia power spectrum  

#Integration limit for radius with respect to virial radius and bin width
#kRmax for convergence
#yRmax for SZ 
kRmax = 5.
kRspace = 100
yRmax = 4.
yRspace = 100

#ell limits
ellmin = 1
ellmax = 1e4
ellspace = 50 #log space

#limits and space of k, m, z
kmin = 1e-4 #1/Mpc
kmax = 1e4
kspace = 100

mmin = 1e11 #Msol
mmax = 5e15
mspace = 50

zmin = 0.07 #Based on Battaglia 2012 paper zmin=0.07 and zmax=5
zmax = 5
zspace = 51

#Saving output Cls
savefile = True

#Constants 
light_speed = 2.998e5 #km/s
mpctocm = 3.085677581e24
kB_kev_K = 8.617330e-8 #keV k^-1
sigma_t_cm = 6.6524e-25 #cm^2
rest_electron_kev = 511. #keV



"""
This module expresses the default values for the cosmology, halo model, and
the precision of all modules within the code.
"""
### parameters specifying a cosmology.
#Samuels simulation 
#default_cosmo_dict = {
#    "omega_m0": 0.2648,  
#    "omega_b0": 0.04479,
#    "omega_l0": 0.7352,  
#    "omega_r0": 0.,
#    "cmb_temp": 2.726,
#    "h"       : 0.71, 
#    "sigma_8" : 0.8, 
#    "n_scalar": 0.963, 
#    "w0"      : -1.0, 
#    "wa"      : 0.0 
#    }

#Battaglia cosmology
default_cosmo_dict = {
    "omega_m0": 0.25, 
    "omega_b0": 0.043,
    "omega_l0": 0.75, 
    "omega_r0": 0.,
    "cmb_temp": 2.726,
    "h"       : 0.7,
    "sigma_8" : 0.8, 
    "n_scalar": 0.96, 
    "w0"      : -1.0, 
    "wa"      : 0.0 
    }


#default_cosmo_dict = {
#    "omega_m0": 0.264, #0.278 - 4.15e-5/0.7**2, ### total matter density at z=0
#    "omega_b0": 0.04448, #0.046, ### baryon density at z=0
#    "omega_l0": 0.736, #0.722, ### dark energy density at z=0
#    "omega_r0": 0., #4.15e-5/0.7**2, ### radiation density at z=0
#    "cmb_temp": 2.726, ### temperature of the CMB in K at z=0
#    "h"       : 0.71, ### Hubble's constant at z=0 normalized to 1/100 km/s/Mpc
#    "sigma_8" : 0.8, ### over-density of matter at 8.0 Mpc/h
#    "n_scalar": 0.963, #0.9624, ### large k slope of the power spectrum
#    "w0"      : -1.0, ### dark energy equation of state at z=0
#    "wa"      : 0.0 ### varying dark energy equation of state. At a=0 the 
#                    ### value is w0 + wa.
#    }


### Default global integration limits for the code.
default_limits = {
    "k_min": 0.0001, ###k_min & k_max are in h Mpc^-1 
    "k_max": 10000.0,
    "mass_min": -1, ### If instead of integrating the mass function over a fixed
    "mass_max": -1  ### range of nu a fixed mass range is desired, set these
                    ### limits to control the mass integration range. Setting
                    ### These to hard limits can halve the running time of the
                    ### code at the expense of less integration range consitency
                    ### as a function of redshift and cosmology.
    }

### default precision parameters defining how many steps splines are evaluated
### for as well as the convergence of the Romberg integration used.
### If a user defines new derived classes it is recommended that they test if
### these values are still relevant to their modules. As a rule of thumb for
### the module returns: if a module has a quickly varying function use more
### n_points; if a module returns values of the order of the precision increase
### this variable. For highly discontinuous functions it is recommended that,
### instead of changing these variables, the integration method quad in
### scipy is used.
default_precision = {
    "corr_npoints": 50,
    "corr_precision": 1.48e-6,
    "cosmo_npoints": 50,
    "cosmo_precision": 1.48e-8,
    "dNdz_precision": 1.48e-8,
    "halo_npoints": 50,
    "halo_precision": 1.48e-5, ### The difference between e-4 and e-5 are at the
                               ### 0.1% level. Since this is the main slow down
                               ### in the calculation e-4 can be used to speed
                               ### up the code.
    "halo_limit" : 100,
    "kernel_npoints": 50,
    "kernel_precision": 1.48e-6,
    "kernel_limit": 100, ### If the variable force_quad is set in the Kernel 
                         ### class this value sets the limit for the quad
                         ### integration
    "kernel_bessel_limit": 8, ### Defines how many zeros before cutting off
                              ### the Bessel function integration in kernel.py
    "mass_npoints": 50,
    "mass_precision": 1.48e-8,
    "window_npoints": 100,
    "window_precision": 1.48e-6,
    "global_precision": 1.48e-32, ### Since the code has large range of values
                                  ### from say 1e-10 to 1e10 we don't want to 
                                  ### use absolute tolerances, instead using 
                                  ### relative tolerances to define convergence
                                  ### of our integrands
    "divmax":20,                  ### Maximum number of subdivisions for
                                  ### the romberg integration.
    #"epsabs":1e-4,
    #"epsrel":1e-4
    "epsabs":1e-1,
    "epsrel":1e-1
 
    }


