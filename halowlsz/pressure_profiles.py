import os, sys
import numpy as np
from numba import jit
import pylab as pl
from CosmologyFunctions import CosmologyFunctions
from convert_NFW_RadMass import MfracToMvir, MvirToMRfrac

@jit(nopython=True)
def battaglia_profile_2d(x, y, Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h, P01, P02, P03, xc1, xc2, xc3, beta1, beta2, beta3):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    x = r/Rs where r and Rs in angular diameter distance 
    Retrun: 
        Pressure profile in keV/cm^3 at radius r in angular comoving distance
    
    This result is confirmed by using Adam's code
    '''    
    #Rs & R200 are in the physical distance, i.e. angular comoving distance
    x = np.sqrt(x**2. + y**2)
    r = x * Rs
    x = r / R200
    msolar = 1.9889e30 #kg
    mpc2cm = 3.0856e24 #cm 
    G = 4.3e-9 #Mpc Mo^-1 (km/s)^2 
    alpha = 1.0
    gamma = -0.3
    P200 = 200. * rho_critical * omega_b0 * G * M200 / omega_m0 / 2. / R200 #Msun km^2 / Mpc^3 / s^2

    #Delta=200
    P0 = P01 * ((M200 / 1e14)**P02 * (1. + z)**P03)
    xc = xc1 * ((M200 / 1e14)**xc2 * (1. + z)**xc3)
    beta = beta1 * ((M200 / 1e14)**beta2 * (1. + z)**beta3)
    #Delta=500
    #P0 = 7.49 * ((M200 / 1e14)**0.226 * (1. + z)**-0.957)
    #xc = 0.710 * ((M200 / 1e14)**-0.0833 * (1. + z)**0.853)
    #beta = 4.19 * ((M200 / 1e14)**0.0480 * (1. + z)**0.615)
    #Shock Delta=500
    #P0 = 20.7 * ((M200 / 1e14)**-0.074 * (1. + z)**-0.743)
    #xc = 0.438 * ((M200 / 1e14)**0.011 * (1. + z)**1.01)
    #beta = 3.82 * ((M200 / 1e14)**0.0375 * (1. + z)**0.535)


    #print P0, xc, beta
    #print (P200*msolar * 6.24e18 * 1e3 / mpc2cm**3), P0, xc, beta
    pth = P200 * P0 * (x / xc)**gamma * (1. + (x/xc))**(-1. * beta) #(km/s)^2 M_sun / Mpc^3

    #Joule = kg m^2 / s^2, Joule = 6.24e18 eV = 6.24e15 keV
    pth *= (msolar * 6.24e15 * 1e6 / mpc2cm**3) #keV/cm^3. 1e6 implies that I have converted km to m
    p_e = pth * 0.518 #For Y=0.24, Vikram, Lidz & Jain
    return p_e


@jit(nopython=True)
def battaglia_profile_proj(x, y, Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h):
    '''Projected to circle'''
    M = np.sqrt(xmax**2 - x**2)
    N = int(M / 0.01)
    if N == 0:
        return 2. * battaglia_profile_2d(x, 0., Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h)
    else:
        xx = np.linspace(0, M, N)
        f = 0.0
        for x1 in xx:
            f += battaglia_profile_2d(x, x1, Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h)
        f *= (2 * (xx[1] - xx[0]))
        return f


@jit(nopython=True)
def ks2002(x, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, nu=150.):
    '''
    Output is pgas3d in unit of eV/cm^3
    '''
    Mvir, Rvir, M500, R500, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h, frac=500.)
    conc = Rvir / Rs
    #print conc 
    #Eq. 18
    eta0 = 2.235 + 0.202 * (conc - 5.) - 1.16e-3 * (conc - 5.)**2
    #Eq. 17
    gamma = 1.137 + 8.94e-2 * np.log(conc/5.) - 3.68e-3 * (conc - 5.)
    #Eq. 16
    B = 3 / eta0 * (gamma - 1.) / gamma / (np.log(1.+conc) / conc - 1./(1.+conc))
    #print conc, gamma, eta0, B
    #Eq. 15
    ygasc = (1. - B *(1. - np.log(1.+conc) / conc))**(1./(gamma-1.))
    #Eq. 21 of KS 2002
    rhogas0 = 7.96e13 * (omega_b0 * cosmo_h * cosmo_h/omega_m0) * (Mvir*cosmo_h/1e15)/ Rvir**3 / cosmo_h**3 * conc * conc / ygasc / (1.+conc)**2/(np.log(1.+conc)-conc/(1.+conc)) #In the CRL code it is multiplied by square of conc. However, I think it should be multiplied by only with concentration
    #Eq. 19
    Tgas0 = 8.80 * eta0 * Mvir / 1e15 / Rvir #keV. This is really kBT
    Pgas0 = 55.0 * rhogas0 / 1e14 * Tgas0 / 8.

    #x = 8.6e-3
    pgas3d = Pgas0 * (1. - B *(1. - np.log(1.+x) / x))**(gamma/(gamma-1.))  
    #print x,gamma, eta0, B, Pgas0, (1. - B *(1. - np.log(1.+x) / x))**(gamma/(gamma-1.)), pgas3d
    pgas2d = 0.0 
    txarr = np.linspace(x, 5*Rvir/Rs, 100)
    for tx in txarr:
        if tx <= 0:
            continue
        pgas2d += (1. - B *(1. - np.log(1.+tx) / tx))**(gamma/(gamma-1.))
    pgas2d = 2. * Pgas0 * pgas2d * (txarr[1] - txarr[0])
    
    #h = 6.625e-34
    #kB = 1.38e-23
    #Kcmb = 2.725
    #x = h * nu * 1e9 / kB / Kcmb
    #y_factor = Kcmb * (x / np.tanh(x / 2.) - 4)
    p_e = pgas3d*0.518 #eV/cm^3
    return x*Rs/Rvir, pgas3d, p_e #pgas3d*0.518, pgas2d, y_factor * pgas2d

def bprofile(r, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, mtype='vir'):
    '''
    Using Battaglia et al (2012). 
    Eq. 10. M200 in solar mass and R200 in Mpc
    mtype: Definition of mass provided. mtype=vir or frac
    Retrun: 
        Pressure profile in eV/cm^3 at radius r
    '''
    if mtype == 'vir':
        Mvir, Rvir, M200, R200, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)
    if mtype == 'frac':
        Mvir, Rvir, M200, R200, rho_s, Rs = MRfracToMvir(Mvir, z, BryanDelta, rho_critical, cosmo_h)
    print(M200, R200)
    #It seems R200 is in the physical distance, i.e. proper distance
    #Need to multiplied by (1+z) to get the comoving unit as I am giving r in
    #comoving unit.
    R200 *= (1. + z) #Comoving radius 
    #r = x * (1. + z) * Rs
    #r = x * Rs
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
    return x*R200/(1.+z)/Rvir, pth, p_e


#@jit(nopython=True)
def arnaud_profile(x, y, Mvir, zi, BD, rho_crit, hz, omega_b0, omega_m0, cosmo_h):
    Mvir, Rvir, M500, R500, rho_s, Rs = MvirToMRfrac(Mvir, zi, BD, rho_crit, cosmo_h, frac=500.0)
    print(M500, R500)
    r = x * R500
    x = np.sqrt(x**2. + y**2.)
    #Eq. 11, 12, 13
    P0 = 8.403 * (0.7/cosmo_h)**1.5 
    c500 = 1.177
    gamma = 0.3081
    alpha = 1.0510
    beta = 5.4905
    px = P0 / (c500 * x)**gamma / (1. + (c500 * x)**alpha)**((beta - gamma) / alpha)
    #alpha_p=0.12 and alpha'_p(x) can be ignored from first approximation 
    pr = 1.65 * 1e-3 * hz**(8./3.)*(M500/3.e14/0.7)**(2./3.+0.12) * px * 0.7**2 #keV/cm^-3
    return pr / 0.518 


@jit(nopython=True)
def arnaud_profile_2d(x, y, Rs, M500, R500, zi, rho_crit, hz, omega_b0, omega_m0, cosmo_h):

    r = x * R500
    x = np.sqrt(x**2. + y**2.)
    #Eq. 11, 12, 13
    P0 = 8.403 * (0.7/cosmo_h)**1.5 
    c500 = 1.177
    gamma = 0.3081
    alpha = 1.0510
    beta = 5.4905
    px = P0 / (c500 * x)**gamma / (1. + (c500 * x)**alpha)**((beta - gamma) / alpha)
    #alpha_p=0.12 and alpha'_p(x) can be ignored from first approximation 
    pr = 1.65 * 1e-3 * hz**(8./3.)*(M500/3.e14/0.7)**(2./3.+0.12) * px * 0.7**2 #keV/cm^-3
    return pr / 0.518 

@jit(nopython=True)
def arnaud_profile_proj(x, Rs, M500, R500, zi, rho_crit, hz, xmax, omega_b0, omega_m0, cosmo_h):
    M = np.sqrt(xmax**2 - x**2)
    N = int(M / 0.01)
    if N == 0:
        return 2. * arnaud_profile_2d(x, 0, Rs, M500, R500, zi, rho_crit, hz, omega_b0, omega_m0, cosmo_h)
    else:
        xx = np.linspace(0, M, N)
        f = 0.0
        for x1 in xx:
            f += arnaud_profile_2d(x, x1, Rs, M500, R500, zi, rho_crit, hz, omega_b0, omega_m0, cosmo_h)
        #print xx
        f *= (2 * (xx[1] - xx[0]))
        return f

if __name__=='__main__':
    from scipy.interpolate import interp1d
    z = 1. #0.0231
    cosmo = CosmologyFunctions(z, 'wlsz.ini', 'battaglia')
    omega_b0 = cosmo._omega_b0
    omega_m0 = cosmo._omega_m0
    cosmo_h = cosmo._h
    BryanDelta = cosmo.BryanDelta()
    rho_critical = cosmo.rho_crit() * cosmo._h * cosmo._h

    rarr = np.logspace(-3, 3, 100)
    Mvir = 1.e15 #/ cosmo_h

    Mvir, Rvir, M200, R200, rho_s, Rs = MvirToMRfrac(Mvir, z, BryanDelta, rho_critical, cosmo_h)

    print('%.2e %.2f %.2e %.2f %.2e %.2f'%(Mvir, Rvir, M200, R200, rho_s, Rs))
    M200 = 8.915e14
    R200 = 1.392
    Rs = 0.53
    xarr = rarr / Rs
    pe_ba = np.array([battaglia_profile_2d(x, 0., Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h) for x in xarr])
    pl.subplot(121)
    pl.loglog(np.logspace(-3, 3, 100), pe_ba, label='Vinu')
    spl = interp1d(np.logspace(-3, 3, 100), pe_ba, fill_value='extrapolate')
    #This file contains the angular radial bins NOT comoving radial bins and the 3d pressure profile from Adam's code. This is implemented lines between ~130 to 150
    fa = np.genfromtxt('/media/luna1/vinu/software/AdamSZ/pressure3d_z_1_M_1e15') 
    pl.loglog(fa[:,0], fa[:,1], label='Adam')
    pl.legend(loc=0)
    pl.subplot(122)
    pl.scatter(fa[:,0], fa[:,1]/spl(fa[:,0]))
    pl.show()
    sys.exit()
    #ks2002(1.34e-2, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, nu=150.)
    #sys.exit()

    rrvirarr, rrvirarr1, pgas3d_ksarr, pgas3d_baarr = [], [], [], []
    pe_ba_arr, pe_ks_arr = [], []
    for rrs in np.logspace(-2, np.log10(20), 30):
        rrvir, pgas3d_ks, pe_ks = ks2002(rrs, Mvir, z, BryanDelta, rho_critical, omega_b0, omega_m0, cosmo_h, 150.)
        rrvirarr.append(rrvir)
        pe_ba = battaglia_profile_2d(rrs, 0., Rs, M200, R200, z, rho_critical, omega_b0, omega_m0, cosmo_h) 
        rrvirarr1.append(rrs/Rvir)
        pgas3d_ksarr.append(pgas3d_ks)
        pgas3d_baarr.append(pe_ba * 1e3 / 0.518)
        pe_ba_arr.append(pe_ba * 1e3)
        pe_ks_arr.append(pe_ks)


    
    pl.subplot(121)
    pl.loglog(rrvirarr1, pgas3d_baarr, c='k', label='Battaglia')
    pl.loglog(rrvirarr, pgas3d_ksarr, c='g', label='KS')
    f = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/clusters/battagliaprofile/battaglia/xvir_pgas_tsz.txt')
    pl.loglog(f[:,0], f[:,1], c='r', label='CRL Battaglia')
    f = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/clusters/komatsuseljakprofile/ks/xvir_pgas_tsz.txt')
    pl.loglog(f[:,0], f[:,1], c='m', label='CRL KS')
    pl.legend(loc=0)

    f = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/clusters/komatsuseljakprofile/ks/xvir_pgas_tsz.txt')
    pl.subplot(122)
    pl.loglog(rrvirarr1, pe_ba_arr, c='k', label='Battaglia electron')
    pl.loglog(rrvirarr, pe_ks_arr, c='g', label='KS electron')
    f = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/clusters/battagliaprofile/battaglia/xvir_pgas_tsz.txt')
    pl.loglog(f[:,0], f[:,1]*0.518, c='r', label='CRL Battaglia electron')
    f = np.genfromtxt('/media/luna1/vinu/software/komastu_crl/clusters/komatsuseljakprofile/ks/xvir_pgas_tsz.txt')
    pl.loglog(f[:,0], f[:,1]*0.518, c='m', label='CRL KS electron')
    pl.legend(loc=0)
    pl.show()
    sys.exit()

    pl.subplot(133)
    pl.loglog(np.array(rrvirarr), pgas2darr, c='k', label='Vinu')
    pl.loglog(f[:,0], f[:,2], c='r', label='KS')
    pl.legend(loc=0)

    pl.show()

