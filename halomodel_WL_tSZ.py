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
    rho_norm - is the rho_crit * Omega_m * h^2 in the unit of solar  Mpc^(-3)
    at redshift of z
    mass_function -  Mpc^(-3)
 
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
def concentration(Mvir, cosmo_h):
    '''Maccio 07 - Eq. 8. I also found that it is only within 1e9 < M < 5e13.
       What is this Delta_vir=98 in page 57
    '''
    Mvir = Mvir / cosmo_h
    conc = 10**(1.02 - 0.109 * (np.log10(Mvir) - 12.))
    return conc

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def f_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**3. / 3.) - (rho_s * Rs**3) * (np.log((Rs + Rfrac) / Rs) - Rfrac / (Rs + Rfrac))

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def df_Rfrac(Rfrac, rho_s, Rs, rho_critical, frac):
    return (frac * rho_critical * Rfrac**2.) - (rho_s * Rs**3) * (Rfrac / (Rs + Rfrac)**2.)

@jit(nb.typeof((1.0,1.0))(nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def MvirToMRfrac(Mvir, BryanDelta, rho_critical, cosmo_h):
    '''Convert Mvir in solar mass to Rvir in Mpc, M200 in solar mass 
       R200 in Mpc
    '''
    conc = concentration(Mvir, cosmo_h)
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
    return Mfrac, Rfrac 

@jit((nb.float64)(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython=True)
def battaglia_profile(r, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    Retrun: 
        Pressure profile in keV/cm^3 at radius r
    '''
    M200, R200 = MvirToMRfrac(Mvir, BryanDelta, rho_critical, cosmo_h)
    #It seems R200 is in the physical distance, i.e. proper distance
    #Need to multiplied by (1+z) to get the comoving unit as I am giving r in
    #comoving unit.
    R200 *= (1. + z) #Comoving radius 
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

#@jit(nb.typeof((1.0,1.0))(nb.float64, nb.float64, nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),nopython=True)
@jit(nopython=True)
def integrate_1halo(r, Mass, karr, dlnk, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, smooth): 
    '''
    Eq. 8 in Vikram, Lidz & Jain
    '''    
   
    #karr = np.exp(np.linspace(np.log(0.1), np.log(1000), 10000))
    #dlnk = np.log(10000./0.1) / 10000.
    rparr = np.linspace(0.1, 6., 100)
    drp = rparr[1] - rparr[0]
    intk = 0.0
    intksm = 0.0
    ki = 0
    #print 1, drp
    for k in karr:
        up = 0.0
        for rp in rparr:
            up += (4*np.pi*rp*np.sin(rp*k) * drp * battaglia_profile(rp, Mass, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h) / k)
            #print rp, up
            #print battaglia_profile(rp, m, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
        intk += (up * dlnk * k * k * np.sin(k * r) / r)
        intksm += (up * dlnk * k * k * np.sin(k * r) * np.exp(-smooth*smooth*k*k/2.) / r)
    #print r, intk / 2. / np.pi
    xi = intk / 2. / np.pi / np.pi
    xism = intksm / 2. / np.pi / np.pi
    return xi, xism
 

#@jit((nb.float64)(nb.float64, nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),nopython=True)
@jit(nopython=True)
def integrate_2halo(r, pk_arr, marr, karr, bmf, dlnk, dlnm, hb, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, smooth): 
    '''
    Eq. 8 in Vikram, Lidz & Jain
    '''    
    rparr = np.linspace(0.1, 6., 100)
    drp = rparr[1] - rparr[0]
    intk = 0.0
    intksm = 0.0
    ki = 0
    #print 1, dr
    for ki, pk in enumerate(pk_arr):
        k = karr[ki]
        intm = 0.0
        mi = 0
        for mi, m in enumerate(marr):
            up = 0.0
            for rp in rparr:
                up += (4*np.pi*rp*np.sin(rp*k) * drp * battaglia_profile(rp, m, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h) / k)
                #print rp, up
                #print battaglia_profile(rp, m, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
            intm += (bmf[mi] * up * dlnm)
            #intm += (bmf[mi] * dlnm)
        #print 1, m, intm, bmf[mi] 
        intk += (dlnk * k * k * np.sin(k * r) * pk * hb * intm / r)
        intksm += (dlnk * k * k * np.sin(k * r) * pk * hb * intm * np.exp(-smooth*smooth*k*k/2.) / r)
    #print r, intk / 2. / np.pi
    xi = intk / 2. / np.pi / np.pi
    xism = intksm / 2. / np.pi / np.pi
    return xi, xism
 
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


def tsz_model(redshift, Mass, compute, fwhm, rmin, rmax, space, logr=True, plot3d=False, plot_proj=False, plot_mf=False, plot_press_battaglia=False):
    '''
    Compute tSZ halomodel from the given mass and redshift
    '''
    if 0:
        pl.subplot(211)
        #for z in [0, 1, 2]:
        #    cosmo = CosmologyFunctions(z) 
        #    print z, cosmo.nu_m(1e8)
            #print cosmo.BryanDelta() 
            #print z, '%.2e %.2e'%(cosmo.rho_crit(), cosmo.rho_bar())
        #sys.exit()
        cosmo = CosmologyFunctions(redshift) 
        kmin = 1e-4
        kmax = 1e4
        mmin = 1e8
        mmax = 8e15
        dlnk = np.float64(np.log(kmax/kmin) / 100.)
        lnkarr = np.arange(np.log(kmin), np.log(kmax), dlnk)
        karr = np.exp(lnkarr).astype(np.float64)
        pk_arr = np.array([cosmo.linear_power(k) for k in karr]).astype(np.float64)
        pl.loglog(karr, pk_arr, c='b', label='z=0')
        cosmo = CosmologyFunctions(2.0) 
        pk_arr = np.array([cosmo.linear_power(k) for k in karr]).astype(np.float64)
        pl.loglog(karr, pk_arr, c='r', label='z=2.0')
        pl.legend(loc=0)

        pl.subplot(212)
        cosmo = CosmologyFunctions(0.) 
        marr = 10**(np.linspace(8, 16, 300))
        sigma_m = np.array([cosmo.sigma_m(m) for m in marr]) #/ cosmo._growth**(2/3.) 
        f = np.genfromtxt('/media/luna1/vinu/software/AdamSZ/sigma_v_mass_z0.dat')
        pl.plot(f[:,0], f[:,1], c='b', label='Adam')
        spl = InterpolatedUnivariateSpline(np.log10(marr), sigma_m)
        print spl(np.log10(10**14.3610*cosmo._h))
        int_sigma_m = spl(f[:,0]) 
        norm = f[:,1]/int_sigma_m
        pl.plot(f[:,0], sigma_m, c='r', label='Vinu')
        pl.plot(f[:,0], norm * sigma_m, c='g', ls='--', label='Normalzed Vinu')
        pl.plot(f[:,0], norm, c='g', ls='--', label='Normalzation')
        pl.legend(loc=0)
        pl.xlabel(r'$\log_{10}(M)$')
        pl.ylabel(r'$\sigma(M)$')
        pl.savefig('pk_masssigma.pdf', bbox_inches='tight')
        pl.show()
        #print cosmo.rho_crit(), cosmo.omega_m()
        sys.exit()
    cosmo = CosmologyFunctions(redshift) 
    BryanDelta = cosmo.BryanDelta() #OK
    #Msun/Mpc^3 
    rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h #OK
    #print BryanDelta, rho_critical
    omega_b0 = cosmo._omega_b0
    omega_m0 = cosmo._omega_m0
    cosmo_h = cosmo._h

    kmin = 1e-3 #1/Mpc
    kmax = 1e3
    mmin = 1e8
    mmax = 8e15
    dlnk = np.float64(np.log(kmax/kmin) / 100.)
    lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
    karr = np.exp(lnkarr).astype(np.float64)
    #No little h
    #Input Mpc/h to power spectra and get Mpc^3/h^3
    pk_arr = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
    #pl.loglog(karr, pk_arr)
    #pl.show()

    dlnm = np.float64(np.log(mmax/mmin) / 100.)
    lnmarr = np.linspace(np.log(mmin), np.log(mmax), 100)
    marr = np.exp(lnmarr).astype(np.float64)
    print 'dlnk, dlnm ', dlnk, dlnm
    #bias_mass_func(1e13, cosmo, ST99=True)
    #sys.exit()


    cosmo0 = CosmologyFunctions(0)
    #No little h
    #Need to give mass * h and get the sigma without little h
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo0._h) for m in marr])
    rho_norm = cosmo0.rho_bar()
    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0, k=3)

    if plot_mf:
        mf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=False) for m in marr])
        #print bmf
        alf = '/media/luna1/vinu/software/AdamSZ/amass_integrand_test_%.1f'%redshift
        if os.path.exists(alf):
            f = np.genfromtxt(alf)
            pl.scatter(f[:,1], f[:,3], c='r', label='Adam bias MF')
        pl.title('z=%.1f'%redshift)
        pl.loglog(marr, mf, c='b', label='Vinu bias MF')
        pl.legend(loc=0)
        pl.xlabel(r'$M_\odot$')
        pl.ylabel(r'Mpc$^{-3}$')
        pl.savefig('massfunc_%.1f.pdf'%redshift, bbox_inches='tight')
        pl.show()
        sys.exit()

    #No little h
    Mass_sqnu = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigmaSpl(np.log(Mass)) / lnMassSigmaSpl(np.log(Mass))
    hb = np.float64(halo_bias_st(Mass_sqnu))

    bmf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=True) for m in marr]).astype(np.float64)

    #battaglia_profile(10, 1e14, 0.1, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
    #sys.exit()

    if plot_press_battaglia:
        pre = np.array([battaglia_profile(r, Mass, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h) for r in np.linspace(0.1, 3, 100)] )
        f = np.genfromtxt('/media/luna1/vinu/software/AdamSZ/pressure_vs_z_test')
        pl.loglog(np.linspace(0.1, 3, 100), pre, label='Vinu:Battaglia prof: M=1e14, z=0.1')
        pl.loglog(f[:,0], f[:,1], c='r', label='Adam:Battaglia prof: M=1e14, z=0.1')
        pl.xlabel(r'$M_\odot$')
        pl.ylabel(r'keV cm$^{-3}$')
        pl.legend(loc=0)
        pl.show()
        sys.exit()

    #integrate_2halo(1, pk_arr, marr, karr, bmf, dlnk, dlnm, hb, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
    #sys.exit()
    #print rarr, cosmo.comoving_distance()
    smooth_r = (fwhm/60.) * np.pi / 180 / np.sqrt(8 * np.log(2)) * cosmo.comoving_distance() #angle = arc/radius
    #print smooth_r
    #sys.exit()
    #Bk = np.exp(-karr*karr*smooth_r*smooth_r/2.)
    if logr:
        rarr = np.logspace(np.log10(rmin), np.log10(rmax), space).astype(np.float64)
    else:
        rarr = np.linspace(rmin, rmax, space).astype(np.float64)
    print rarr

    fy3d = 'yprof3d_%.1f_%.1f.txt'%(np.log10(Mass), redshift)
    fy2d = 'yproj_%.1f_%.1f.txt'%(np.log10(Mass), redshift)
    if compute:
        xi1h, xi2h, xi = [], [], [] 
        xi1hsm, xi2hsm, xism = [], [], [] 
        trarr = np.linspace(1e-4, 200, 1000)
        dtr = trarr[1] - trarr[0]
        h11 = np.array([battaglia_profile(r, Mass, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h) for r in trarr])
        print smooth_r, dtr
        h1sm = gaussian_filter(h11, smooth_r/dtr, mode='constant', cval=0.) 
        h1smSpl = InterpolatedUnivariateSpline(trarr, h1sm, k=1) 
        #pl.semilogx(trarr, h11)
        #pl.semilogx(trarr, h1smSpl(trarr))
        #pl.show()
        #sys.exit()
        print 'Comoving R 1halo 2halo'
        for r in rarr:
            h1 = battaglia_profile(r, Mass, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h)
            h1sm = h1smSpl(r)
            #h1, h1sm = integrate_1halo(r, Mass, karr, dlnk, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, smooth_r)
            h2, h2sm = integrate_2halo(r, pk_arr, marr, karr, bmf, dlnk, dlnm, hb, redshift, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, smooth_r)
            #h2 = 0.
            xi1h.append(h1)
            xi2h.append(h2)
            xi.append(h1 + h2)
            xi1hsm.append(h1sm)
            xi2hsm.append(h2sm)
            xism.append(h1sm + h2sm)
 
            print '%.2e %.2e %.2e %.2e %.2e'%(r, h1, h1sm, h2, h2sm)
        np.savetxt(fy3d, np.transpose((rarr, xi1h, xi2h, xi, xi1hsm, xi2hsm, xism)))

    theta_arcmin_arr = 60 * (180. / np.pi) * rarr / cosmo.comoving_distance()
    theta_radian_arr = rarr / cosmo.comoving_distance()
    larr = 1 / theta_radian_arr
 
    f = np.genfromtxt(fy3d)
    rarr, xi1h, xi2h, xi = f[:,0], f[:,1], f[:,2], f[:,3]
    xi1hsm, xi2hsm, xism = f[:,4], f[:,5], f[:,6]

    if plot3d: 
        f = np.genfromtxt('/media/luna1/vinu/software/AdamSZ/twop_press_test_2')
        pl.loglog(rarr, xi1h, c='r', label='1halo-Vinu')
        pl.loglog(rarr, xi2h, c='g', label='2halo-Vinu')
        pl.loglog(rarr, xi, c='k', label='Total-Vinu')
        pl.loglog(f[:,0], f[:,1], c='r', ls='--', label='1halo-Adam')
        pl.loglog(f[:,0], f[:,2], c='g', ls='--', label='2halo-Adam')
        pl.legend(loc=0)
        pl.xlabel('r (Mpc)')
        pl.ylabel(r'$\xi_{y,g}(r)$')
        pl.savefig('compareAdamVinu3d.pdf', bbox_inches='tight')
        pl.show()
        #sys.exit()
 
    xi1h_spl = InterpolatedUnivariateSpline(rarr, xi1h, k=3)
    xi2h_spl = InterpolatedUnivariateSpline(rarr, xi2h, k=3)
    xi_spl = InterpolatedUnivariateSpline(rarr, xi, k=3)
    xi1hsm_spl = InterpolatedUnivariateSpline(rarr, xi1hsm, k=3)
    xi2hsm_spl = InterpolatedUnivariateSpline(rarr, xi2hsm, k=3)
    xism_spl = InterpolatedUnivariateSpline(rarr, xism, k=3)
 
    #trarr = np.logspace(np.log10(0.1), np.log10(50), 100)
    #pl.loglog(trarr, xi1h_spl(trarr))
    #pl.show()

    sigma_t=6.6524e-25 #cm^2
    rest_electron_kev=511. #keV
    mpc2cm = mpc_to_cm=3.0856e24
    rmin = .0
    #rmax = 50.
    #space = 150. #rmin=0, rmax=50 and space=150 is consistent with Adam's model
    #dr = (rmax - rmin) / (space - 1)
    dr = 0.33557047
    space = int(1 + (rmax - rmin) / dr)
    sqxi = np.linspace(rmin, rmax, space)**2.
    xi1h_proj, xi2h_proj, xi_proj = [], [], [] 
    xi1hsm_proj, xi2hsm_proj, xism_proj = [], [], [] 
    for r in rarr:
        R = np.sqrt(r**2. + sqxi)
        R = R[R < rmax]
        xi1h_proj.append(xi1h_spl(R).sum()*dr)
        xi2h_proj.append(xi2h_spl(R).sum()*dr)
        xi_proj.append(xi_spl(R).sum()*dr)
        xi1hsm_proj.append(xi1hsm_spl(R).sum()*dr)
        xi2hsm_proj.append(xi2hsm_spl(R).sum()*dr)
        xism_proj.append(xism_spl(R).sum()*dr)
 
           
    xi1h_proj = 2. * np.array(xi1h_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift) 
    xi2h_proj = 2. * np.array(xi2h_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift)
    xi_proj = 2. * np.array(xi_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift)
    
    ##larr = np.linspace(1, 1e6, 10000)
    ##Bl = np.exp(-larr*(1.+larr)*smooth_r*smooth_r/2.)
    ##hh = 0.002
    ##NN = 20
    ##clsm_proj = fastcorr.calc_corr(larr, theta_radian_arr, xi_proj, N=NN, h=hh)
    ##clsm_proj = np.nan_to_num(clsm_proj) * Bl
    ##xism_proj = fastcorr.calc_corr(theta_radian_arr, larr, clsm_proj, N=NN, h=hh)
    ##xism_proj = np.nan_to_num(xism_proj)
    ##print xism_proj

    xi1hsm_proj = 2. * np.array(xi1hsm_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift) 
    xi2hsm_proj = 2. * np.array(xi2hsm_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift)
    xism_proj = 2. * np.array(xism_proj) * sigma_t * mpc2cm / rest_electron_kev / (1. + redshift)
    


    #xi1h_proj_sm = gaussian_filter(xi1h_proj, smooth_r)
    #xi2h_proj_sm = gaussian_filter(xi2h_proj, smooth_r)
    #xi_proj_sm = gaussian_filter(xi_proj, smooth_r)
    #print rarr
    #print xi1h_proj
    #print xi2h_proj
    #print xi_proj
    np.savetxt(fy2d, np.transpose((theta_arcmin_arr, rarr, xi1h_proj, xi2h_proj, xi_proj, xi1hsm_proj, xi2hsm_proj, xism_proj)), fmt='%.6e', header='R(arcmin) R(Comovin-Mpc) 1halo 2halo Total Smoothed-1halo Smoothed-2halo Smoothed-Total')

    if plot_proj:    
        alf = '/media/luna1/vinu/software/AdamSZ/yproj_test_%.1f'%redshift
        if os.path.exists(alf):
            f = np.genfromtxt(alf)
            pl.loglog(f[:,0], f[:,1], c='r', ls='--', label='1halo-Adam')
            pl.loglog(f[:,0], f[:,2], c='g', ls='--', label='2halo-Adam')
            pl.loglog(f[:,0], f[:,3], c='k', ls='--', label='Total-Adam')
        pl.loglog(rarr, xi1h_proj, c='r', label='1halo-Vinu')
        pl.loglog(rarr, xi2h_proj, c='g', label='2halo-Vinu')
        pl.loglog(rarr, xi_proj, c='k', label='Total-Vinu')
        #pl.loglog(rarr, xism_proj, c='k', label='Smoothed total-Vinu')
        pl.legend(loc=0)
        pl.xlabel('r (Mpc)')
        pl.ylabel(r'$\xi_{y,g}(r)$')
        pl.savefig('compareAdamVinu_%.1f.pdf'%redshift, bbox_inches='tight')
        pl.show()

if __name__=='__main__':
    #Write variables
    redshift = 0.5 #Redshift of the halo
    Mass = 1e14 #mass of the halo
    compute = 0 #Whether the profile should be computed 
    fwhm = 0 #arcmin Doesn't work now
    rmin = 1e-2 #Inner radius of pressure profile 
    rmax = 1e2 #Outer radius of pressure profile
    space = 50 #logarithmic space between two points
    #Stop

    if 0:
        mmin = 1e8
        mmax = 8e15
        dlnm = np.float64(np.log(mmax/mmin) / 9.)
        lnmarr = np.linspace(np.log(mmin), np.log(mmax), 10)
        marr = np.exp(lnmarr).astype(np.float64)
        colors = pl.cm.jet(np.linspace(0, 1, marr.shape[0])) 
        for i, m in enumerate(marr):
            sigma_m = np.array([CosmologyFunctions(z).sigma_m(m) for z in np.linspace(0,2,10)])
            pl.plot(np.linspace(0,2,10), sigma_m, label='M=%.2e'%m, c=colors[i])
        pl.legend(loc=0)
        pl.xlabel('z')
        pl.ylabel(r'$\sigma_m$')
        pl.show()
        sys.exit()
    if 0:
        cosmo = CosmologyFunctions(redshift)
        mmin = 1e8
        mmax = 8e15
        dlnm = np.float64(np.log(mmax/mmin) / 100.)
        
        lnmarr = np.linspace(np.log(mmin), np.log(mmax), 100)
        marr = np.exp(lnmarr).astype(np.float64)
        cosmo0 = CosmologyFunctions(0)
        #Byt giving m * h to sigm_m gives the sigma_m at z=0 
        sigma_m0 = np.array([cosmo0.sigma_m(m*cosmo0._h) for m in marr])
        rho_norm = cosmo0.rho_bar()
        lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0) 
        #pl.plot(lnmarr, sigma_m0)
        #pl.plot(lnmarr, lnMassSigmaSpl(lnmarr))
        #pl.show()

        mf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=0) for m in marr])
        bmf = np.array([bias_mass_func(m, cosmo, lnMassSigmaSpl, rho_norm, ST99=True, bias=1) for m in marr])
        np.savetxt('%.1f.txt'%redshift, np.transpose((marr, mf, bmf)))
        #f = np.genfromtxt('/media/luna1/vinu/software/AdamSZ/amass_integrand_test_%.f'%redshift)
        #pl.scatter(f[:,1], f[:,3], c='r', label='Adam bias MF')
        #pl.loglog(marr, mf)
        #pl.show()
        sys.exit()
    if 0:
        cosmo = CosmologyFunctions(redshift)
        kmin = 1e-4
        kmax = 1e4
        mmin = 1e8
        mmax = 8e15
        dlnk = np.float64(np.log(kmax/kmin) / 100.)
        lnkarr = np.linspace(np.log(kmin), np.log(kmax), 100)
        karr = np.exp(lnkarr).astype(np.float64)
        #No little h
        pk_arr = np.array([cosmo.linear_power(k/cosmo._h) for k in karr]).astype(np.float64) / cosmo._h / cosmo._h / cosmo._h
        np.savetxt('pk_%.1f.txt'%redshift, np.transpose((karr, pk_arr))) 
        #pl.loglog(karr, pk_arr)
        #pl.show()
        sys.exit()
    tsz_model(redshift, Mass, compute, fwhm, rmin, rmax, space, logr=True, plot3d=False, plot_proj=1, plot_mf=False, plot_press_battaglia=False)

