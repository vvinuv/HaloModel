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
import cosmology_vinu as cosmology
#import fastcorr
from halomodel_tSZ import CosmologyFunctions
 
__author__ = ("Vinu Vikraman <vvinuv@gmail.com>")

def halo_bias_st(sqnu):
    '''
    Eq. 8 in Sheth et al 2001
    '''
    common = 1./np.sqrt(0.707)/1.686
    fterm = np.sqrt(0.707) * 0.707 * sqnu #First term
    sterm = np.sqrt(0.707) * 0.5 * (0.707 * sqnu)**(1.-0.6) #second term
    ttermn = (0.707 * sqnu)**0.6 #numerator of third term
    ttermd = (0.707 * sqnu)**0.6 + 0.5 * (1.-0.6) * (1.-0.6/2.) #demoninator of third term
    tterm = ttermn / ttermd #third term
    blag = common * (fterm + sterm - tterm) #b_lag
    return 1+blag #b_eul
 
def bias_mass_func(Mvir, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=True):
    '''
    p21 and Eqns. 56-59 of CS02     

    m^2 n(m,z) * dm * nu = m * bar(rho) * nu f(nu) * dnu 
    n(m,z) = bar(rho) * nu f(nu) * (dlnnu / dlnm) / m^2
    n(z) = bar(rho) * nu f(nu) * (dlnnu / dlnm) / m

    Mvir -solar unit
    nu, nuf - unitless
    rho_norm - is the rho_crit * Omega_m * h^2 in the unit of solar  Mpc^(-3) at z=0
    mass_function -  Mpc^(-3)
    at redshift of z
 
    '''
    mass_array = np.logspace(np.log10(Mvir*0.9999), np.log10(Mvir*1.0001), 2)
    ln_mass_array = np.log(mass_array)
    ln_sigma_m_array = np.log(np.array([lnMassSigmaSpl(np.log(m)) for m in mass_array]))
    #spl = UnivariateSpline(ln_mass_array, ln_nu_array, k=3)
    #print spl.derivatives(np.log(Mvir))[1] 
    #Derivatives of dln_nu/dln_mass at ln_mass
    ln_sigma_m_ln_mass_derivative = abs((ln_sigma_m_array[1] - ln_sigma_m_array[0]) / (ln_mass_array[1] - ln_mass_array[0]))#1 for first derivate
    #print 'ln_sigma_m_ln_mass_derivative ', Mvir, ln_sigma_m_ln_mass_derivative

    #This is returns square of density^2_sc/sigma^2(m) 
    nu = cosmo.nu_m(Mvir)
    #print Mvir, nu
    nusq = np.sqrt(nu)
     
    if ST99:
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

    #print Mvir, rho_norm * cosmo._h * cosmo._h, cosmo.delta_c()/cosmo._growth, lnMassSigmaSpl(np.log(Mvir)),ln_sigma_m_ln_mass_derivative 
    mass_function = nuf * rho_norm * cosmo._h * cosmo._h * ln_sigma_m_ln_mass_derivative / Mvir
    #print Mvir, halo_bias(np.sqrt(nu)), halo_bias(nu), mass_function
    if bias:
        return mass_function * halo_bias_st(delta_c_sigma_m2)
    else:
        return mass_function
        #return anst


@jit((nb.float64)(nb.float64, nb.float64),nopython=True)
def concentration_maccio(Mvir, cosmo_h):
    '''Maccio 07 - Eq. 8. I also found that it is only within 1e9 < M < 5e13.
       What is this Delta_vir=98 in page 57
    '''
    Mvir = Mvir / cosmo_h
    conc = 10**(1.02 - 0.109 * (np.log10(Mvir) - 12.))
    return conc

@jit((nb.float64)(nb.float64, nb.float64, nb.float64),nopython=True)
def concentration_duffy(Mvir, z, cosmo_h):
    '''Duffy 2008'''
    conc = (5.72 / (1. + z)**0.71) * (Mvir * cosmo_h / 1e14)**-0.081
    return conc

@jit(nopython=True)
def Wk(zl, chil, zsarr, chisarr, Ns, constk):
    #zl = lens redshift
    #chil = comoving distant to lens
    #zsarr = redshift distribution of source
    #angsarr = angular diameter distance
    #Ns = Normalized redshift distribution of sources 
    al = 1. / (1. + zl)
    Wk = constk * chil / al
    gw = 0.0
    for i, N in enumerate(Ns):
        if chisarr[i] < chil:
            continue
        gw += ((chisarr[i] - chil) * N / chisarr[i])
    gw *= (zsarr[1] - zsarr[0])
    if gw <= 0:
        gw = 0.
    Wk = Wk * gw
    return Wk

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def f_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**3. / 3.) - (rho_s * Rs**3) * (np.log((Rs + Rfrac) / Rs) - Rfrac / (Rs + Rfrac))

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def df_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**2.) - (rho_s * Rs**3) * (Rfrac / (Rs + Rfrac)**2.)

@jit(nopython=True)
def MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h):
    '''Convert Mvir in solar mass to Rvir in Mpc, M200 in solar mass 
       R200 in Mpc
    '''
    #conc = concentration_maccio(Mvir, cosmo_h)
    conc = concentration_duffy(Mvir, z, cosmo_h)
    #print Mvir, conc
    Rvir = (Mvir / ((4 * np.pi / 3.) * BryanDelta * rho_critical))**(1/3.) #(Msun / Msun Mpc^(-3))1/3. -> Mpc    
    rho_s = rho_critical * (BryanDelta / 3.) * conc**3. / (np.log(1 + conc) - conc / (1 + conc)) #Msun / Mpc^3  
    Rs = Rvir / conc

    tolerance = 1e-6
    frac = 200.0

    # Using Newton - Raphson method. x1 = x0 - f(x0) / f'(x0) where x0 is
    # the initial guess, f and f' are the function and derivative

    # Intial guess is Rvir / 2. 
    x0 = Rvir / 2.0
    tol = Rvir * tolerance #tolerance
    x1 = tol * 10**6
    #print 1, x0, x1
    while abs(x0 - x1) > tol:
        #print abs(x0 - x1), tol
        x0 = x1 * 1.0
        x1 = x0 - f_Rfrac(x0, rho_s, Rs, rho_critical, frac) / df_Rfrac(x0, rho_s, Rs, rho_critical, frac)
        #print x0, x1
    Rfrac = x1
    Mfrac = (4. / 3.) * np.pi * Rfrac**3 * frac * rho_critical
    #print Mvir, Mfrac, Rvir, Rfrac
    return Mfrac, Rfrac, rho_s, Rs, Rvir 

@jit(nopython=True)
def battaglia_profile(x, Rs, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    Retrun: 
        Pressure profile in keV/cm^3 at radius r
    '''
    M200, R200, rho_s, Rs, Rvir = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
    #It seems R200 is in the physical distance, i.e. proper distance
    #Need to multiplied by (1+z) to get the comoving unit as I am giving r in
    #comoving unit.
    R200 *= (1. + z) #Comoving radius 
    r = x * (1. + z) * Rs
    x = r / R200 
    #print Mvir, M200, R200
    msolar = 1.9889e30 #kg
    mpc2cm = 3.0856e24 #cm 
    G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * omega_b0 * G * M200 / omega_m0 / 2. / (R200 / (1. + z)) #Msun km^2 / Mpc^3 / s^2

    P0 = 18.1 * ((M200 / 1e14)**0.154 * (1. + z)**-0.758)
    xc = 0.497 * ((M200 / 1e14)**-0.00865 * (1. + z)**0.731)
    beta = 4.35 * ((M200 / 1e14)**0.0393 * (1. + z)**0.415) 
    #print P0, xc, beta
    #print (P200*msolar * 6.24e18 * 1e3 / mpc2cm**3), P0, xc, beta
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta) #(km/s)^2 M_sun / Mpc^3

    #Joule = kg m^2 / s^2, Joule = 6.24e18 eV = 6.24e15 keV
    pth *= (msolar * 6.24e15 * 1e6 / mpc2cm**3) #keV/cm^3. 1e6 implies that I have converted km to m
    p_e = pth * 0.518 #For Y=0.24, Vikram, Lidz & Jain
    return p_e

@jit(nopython=True)
def integrate_halo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    dz = zarr[1] - zarr[0]
    cl1h = 0.0
    cl2h = 0.0
    mfj = 0
    for i, zi in enumerate(zarr):
        #print  zi, Wk(zi, chiarr[i], zsarr, angsarr, Ns, constk)
        kl_yl_multi = Wk(zi, chiarr[i], zsarr, chisarr, Ns, constk) * consty / chiarr[i] / chiarr[i] / rhobarr[i] 
        mint = 0.0
        mk2 = 0.0
        my2 = 0.0
        for mi in marr:
            kint = 0.0
            yint = 0.0
            Mfrac, Rfrac, rho_s, Rs, Rvir = MvirToMRfrac(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h)
            #Eq. 3.2 Ma et al
            rp = np.linspace(0, 5*Rvir, 100)
            for tr in rp:
                if tr == 0:
                    continue 
                kint += (tr * tr * np.sin(ell * tr / chiarr[i]) / (ell * tr / chiarr[i]) * rho_s / (tr/Rs) / (1. + tr/Rs)**2.)
            kint *= (4. * np.pi * (rp[1] - rp[0]))
            #Eq. 3.3 Ma et al
            xmax = 5 * Rvir / Rs / (1. + zi) #Ma et al paper says that Eq. 3.3 convergence by r=5 rvir. I didn't divided by 1+z because Battaglia model is calculated in comoving radial coordinate
            xp = np.linspace(0, xmax, 20)
            ells = chiarr[i] / (1. + zi) / Rs
            for x in xp:
                if x == 0:
                    continue 
                yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile(x, Rs, mi, zi, BDarr[i], rho_crit_arr[i], omega_b0, omega_m0, cosmo_h))
            yint *= (4 * np.pi * Rs * (xp[1] - xp[0]) / ells / ells)
            mint += (dlnm * mf[mfj] * kint * yint)
            mk2 += (dlnm * bias[mfj] * mf[mfj] * kint)
            my2 += (dlnm * bias[mfj] * mf[mfj] * yint)
            mfj += 1
        cl1h += (dVdzdOm[i] * kl_yl_multi * mint)
        cl2h += (dVdzdOm[i] * pk[i] * Darr[i] * Darr[i] * kl_yl_multi * mk2 * my2)
    cl1h *= dz
    cl2h *= dz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 

@jit(nopython=True)
def integrate_kkhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    dz = zarr[1] - zarr[0]
    cl1h = 0.0
    cl2h = 0.0
    mfj = 0
    for i, zi in enumerate(zarr):
        #print  zi, Wk(zi, chiarr[i], zsarr, angsarr, Ns, constk)
        kl_multi = Wk(zi, chiarr[i], zsarr, chisarr, Ns, constk) / chiarr[i] / chiarr[i] / rhobarr[i] 
        mint = 0.0
        mk2 = 0.0
        for mi in marr:
            kint = 0.0
            Mfrac, Rfrac, rho_s, Rs, Rvir = MvirToMRfrac(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h)
            #Eq. 3.2 Ma et al
            #limit_kk_Rvir.py tests the limit of Rvir. 
            rp = np.linspace(0, 5 * Rvir, 100)
            for tr in rp:
                if tr == 0:
                    continue 
                kint += (tr * tr * np.sin(ell * tr / chiarr[i]) / (ell * tr / chiarr[i]) * rho_s / (tr/Rs) / (1. + tr/Rs)**2.)
            kint *= (4. * np.pi * (rp[1] - rp[0]))
            mint += (dlnm * mf[mfj] * kint * kint)
            mk2 += (dlnm * bias[mfj] * mf[mfj] * kint)
            mfj += 1
        cl1h += (dVdzdOm[i] * kl_multi * kl_multi * mint)
        cl2h += (dVdzdOm[i] * pk[i] * Darr[i] * Darr[i] * kl_multi * kl_multi * mk2 * mk2)
    cl1h *= dz
    cl2h *= dz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 
@jit(nopython=True)
def integrate_yyhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty): 
    '''
    Eq. 3.1 Ma et al. 
    '''    
   
    dz = zarr[1] - zarr[0]
    cl1h = 0.0
    cl2h = 0.0
    mfj = 0
    for i, zi in enumerate(zarr):
        #print  zi, Wk(zi, chiarr[i], zsarr, angsarr, Ns, constk)
        mint = 0.0
        my2 = 0.0
        for mi in marr:
            yint = 0.0
            Mfrac, Rfrac, rho_s, Rs, Rvir = MvirToMRfrac(mi, zi, BDarr[i], rho_crit_arr[i], cosmo_h)
            #Eq. 3.3 Ma et al
            xmax = 5 * Rvir / Rs / (1. + zi) #Ma et al paper says that Eq. 3.3 convergence by r=5 rvir. I didn't divided by 1+z because Battaglia model is calculated in comoving radial coordinate 
            xp = np.linspace(0, xmax, 20)
            ells = chiarr[i] / (1. + zi) / Rs
            for x in xp:
                if x == 0:
                    continue 
                yint += (x * x * np.sin(ell * x / ells) / (ell * x / ells) * battaglia_profile(x, Rs, mi, zi, BDarr[i], rho_crit_arr[i], omega_b0, omega_m0, cosmo_h))
            yint *= (4 * np.pi * Rs * (xp[1] - xp[0]) / ells / ells)
            mint += (dlnm * mf[mfj] * yint * yint)
            my2 += (dlnm * bias[mfj] * mf[mfj] * yint)
            mfj += 1
        cl1h += (dVdzdOm[i] * consty * consty * mint)
        cl2h += (dVdzdOm[i] * pk[i] * Darr[i] * Darr[i] * consty * consty * my2 * my2)
    cl1h *= dz
    cl2h *= dz
    cl = cl1h + cl2h
    return cl1h, cl2h, cl
 


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


def wl_tsz_model(compute, fwhm, zsfile='source_distribution.txt', kk=False, yy=False, ky=True):
    '''
    Compute tSZ halomodel from the given mass and redshift
    '''
    fwhm = fwhm * np.pi / 2.355 / 60. /180. #angle in radian
    fwhmsq = fwhm * fwhm

    cosmo0 = CosmologyFunctions(0)
    omega_b0 = cosmo0._omega_b0
    omega_m0 = cosmo0._omega_m0
    cosmo_h = cosmo0._h

    light_speed = 2.998e5 #km/s
    mpctocm = 3.085677581e24
    kB_kev_K = 8.617330e-8 #keV k^-1
    sigma_t_cm = 6.6524e-25 #cm^2
    rest_electron_kev = 511 #keV
    constk = 3. * omega_m0 * (cosmo_h * 100. / light_speed)**2. / 2. #Mpc^-2
    consty = mpctocm * sigma_t_cm / rest_electron_kev 

    fz= np.genfromtxt(zsfile)
    zsarr = fz[:,0]
    Ns = fz[:,1]
    zint = np.sum(Ns) * (zsarr[1] - zsarr[0])
    Ns /= zint

    kmin = 1e-4 #1/Mpc
    kmax = 1e4
    mmin = 1e10
    mmax = 1e16
    dlnk = np.float64(np.log(kmax/kmin) / 100.)
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
    karr = np.exp(lnkarr).astype(np.float64)
    #No little h
    #Input Mpc/h to power spectra and get Mpc^3/h^3
    pk_arr = np.array([cosmo0.linear_power(k/cosmo0._h) for k in karr]).astype(np.float64) / cosmo0._h / cosmo0._h / cosmo0._h
    pkspl = InterpolatedUnivariateSpline(karr/cosmo0._h, pk_arr, k=2) 
    #pl.loglog(karr, pk_arr)
    #pl.show()

    mspace = 50
    dlnm = np.float64(np.log(mmax/mmin) / mspace)
    lnmarr = np.linspace(np.log(mmin), np.log(mmax), mspace)
    marr = np.exp(lnmarr).astype(np.float64)
    print 'dlnk, dlnm ', dlnk, dlnm
    #bias_mass_func(1e13, cosmo, ST99=True)
    #sys.exit()

    #No little h
    #Need to give mass * h and get the sigma without little h
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo0._h) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0, k=3)

    zarr = np.linspace(0.05, 1., 50)
    BDarr, rhobarr, chiarr, dVdzdOm, rho_crit_arr = [], [], [], [], []
    bias, Darr = [], []
    mf = []
    for zi in zarr:
        cosmo = CosmologyFunctions(zi)
        BDarr.append(cosmo.BryanDelta()) #OK
        rhobarr.append(cosmo.rho_bar() * cosmo._h * cosmo._h)
        chiarr.append(cosmo.comoving_distance() / cosmo._h)
        #Msun/Mpc^3 
        mf.append(np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm0, ST99=True, bias=False) for m in marr]).astype(np.float64))
        bias.append(np.array([halo_bias_st(cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(m)) / lnMassSigmaSpl(np.log(m))) for m in marr]))
        rho_crit_arr.append(cosmo.rho_crit() * cosmo._h * cosmo._h) #OK
        dVdzdOm.append(cosmo.E(zi) / cosmo._h) #Mpc/h, It should have (km/s/Mpc)^-1 but in the cosmology code the speed of light is removed  
        Darr.append(cosmo._growth)

    BDarr = np.array(BDarr)
    rhobarr = np.array(rhobarr)
    chiarr = np.array(chiarr)
    dVdzdOm = np.array(dVdzdOm) * chiarr * chiarr
    rho_crit_arr = np.array(rho_crit_arr)
    mf = np.array(mf).flatten()
    zchispl = InterpolatedUnivariateSpline(zarr, chiarr, k=2)
    chisarr = zchispl(zsarr)
    bias = np.array(bias).flatten()
    Darr = np.array(Darr)
    print mf.shape

    ellarr = np.linspace(1, 5001, 500)
    ellarr = np.logspace(0, np.log10(5001), 100)
    cl_arr, cl1h_arr, cl2h_arr = [], [], []
    for ell in ellarr:
        pk = pkspl(ell/chiarr)
        if ky: 
            cl1h, cl2h, cl = integrate_halo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        if kk:
            cl1h, cl2h, cl = integrate_kkhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        if yy:
            cl1h, cl2h, cl = integrate_yyhalo(ell, zarr, chiarr, dVdzdOm, marr, mf, BDarr, rhobarr, rho_crit_arr, bias, Darr, pk, zsarr, chisarr, Ns, dlnm, omega_b0, omega_m0, cosmo_h, constk, consty)
        cl_arr.append(cl)
        cl1h_arr.append(cl1h)
        cl2h_arr.append(cl2h)
        print ell, cl1h, cl2h, cl

    convolve = np.exp(-1 * fwhmsq * ellarr * ellarr)# i.e. the output is Cl by convolving by exp(-sigma^2 l^2)
    cl = np.array(cl_arr) * convolve
    cl1h = np.array(cl1h_arr) * convolve
    cl2h = np.array(cl2h_arr) * convolve
    
    if ky:
        np.savetxt('cl_ky.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')
    if kk:
        np.savetxt('cl_kk.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')
    if yy:
        np.savetxt('cl_yy.dat', np.transpose((ellarr, cl1h, cl2h, cl)), fmt='%.3e')


    if yy:
        #Convert y to \delta_T using 147 GHz. (g(x) TCMB)^2 = 7.2786
        cl *= 7.2786
        cl1h *= 7.2786
        cl2h *= 7.2786
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi, label='Cl')
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
        pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
        pl.xlabel(r'$\ell$')
        pl.ylabel(r'$c_\ell \ell (\ell + 1)/2/\pi \mu K^2$')
        pl.legend(loc=0)
    else:
        pl.plot(ellarr, ellarr * (ellarr+1) * cl / 2. / np.pi, label='Cl')
        pl.plot(ellarr, ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
        pl.plot(ellarr, ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
        pl.xlabel(r'$\ell$')
        pl.ylabel(r'$c_\ell \ell (\ell + 1)/2/\pi$')
        pl.legend(loc=0)

    pl.show()
    #No little h
    #Mass_sqnu = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(Mass)) / lnMassSigmaSpl(np.log(Mass))
    #hb = np.float64(halo_bias_st(Mass_sqnu))

    #bmf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=True) for m in marr]).astype(np.float64)

    sys.exit()

    #integrate_2halo(1, pk_arr, marr, karr, bmf, dlnk, dlnm, hb, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
    #sys.exit()
    #print rarr, cosmo.comoving_distance()
    #smooth_r = (fwhm/60.) * np.pi / 180 / np.sqrt(8 * np.log(2)) * cosmo.comoving_distance() #angle = arc/radius
    #print smooth_r
    #sys.exit()
    #Bk = np.exp(-karr*karr*smooth_r*smooth_r/2.)
    if logr:
        rarr = np.logspace(np.log10(rmin), np.log10(rmax), space).astype(np.float64)
    else:
        rarr = np.linspace(rmin, rmax, space).astype(np.float64)
    print rarr

if __name__=='__main__':
    #Write variables
    compute = 0 #Whether the profile should be computed 
    fwhm = 0 #arcmin Doesn't work now
    rmin = 1e-2 #Inner radius of pressure profile 
    rmax = 1e2 #Outer radius of pressure profile
    space = 50 #logarithmic space between two points
    #Stop

    if 1:
        wl_tsz_model(compute, fwhm, zsfile='source_distribution.txt', kk=1, yy=0, ky=0)
    
    if 0:
        z = 0.01
        Mvir = 1e15
        cosmo = CosmologyFunctions(0)
        omega_b = cosmo._omega_b0
        omega_m = cosmo._omega_m0
        cosmo_h = cosmo._h
        BryanDelta = cosmo.BryanDelta() 
        rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h
        Mfrac, Rfrac, rho_s, Rs, Rvir = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
        rp = np.linspace(0, 5 * Rvir, 100)
        xp = rp / Rs / (1 + z) #Ma et al paper says that Eq. 3.3 convergence by r=5 rvir 
        Pe = []
        for x in xp:
            Pe.append(battaglia_profile(x, Rs, Mvir, z, BryanDelta, rho_critical, omega_b, omega_m, cosmo_h))
        Pe = np.array(Pe) * rp * rp * 1e3 #keV/cm^3 to ev/cm^3
        pl.semilogy(rp, Pe)
        pl.show() 
