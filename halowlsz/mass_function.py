import os
import sys
import numpy as np
import pylab as pl
from scipy.interpolate import interp1d
from CosmologyFunctions import CosmologyFunctions
from scipy.interpolate import InterpolatedUnivariateSpline
from convert_NFW_RadMass import MvirToMRfrac, MfracToMvir
import warnings
 
class MassFunctionSingle:
    '''
    This class consists of mass function  and halo bias 
    Eqs. 56-59 of Cooray & Sheth 2002
    Input:
        Virial mass Mvir (not with little h)
        Redshift 
        bias: 0/1/2 if it returns only mass function, only halo bias or
              mass function weighted by halo bias
        mass function: ST99, CS02, PS74 

    Output:
        Mass function in 1/Mpc^3
    '''
    def __init__(self, Mvir, redshift, bias, mf='ST99'):
        '''
        cosmological parameters 
        '''
        self.Mvir = Mvir
        self.redshift = redshift
        self.bias = bias
        self.mf = mf
        self.rho_norm, self.lnMassSigmaSpl = self.RhoNorm(self.Mvir)
        self.massfunction() #self.Mvir, self.redshift, self.lnMassSigmaSpl, self.rho_norm, self.mf)
 
    def RhoNorm(self, Mvir):
        '''
        Return the normaliztion for mass function
        ''' 
        lnmarr = np.linspace(np.log(Mvir * 0.99), np.log(Mvir * 1.01), 10)
        marr = np.exp(lnmarr).astype(np.float64)
        cosmo0 = CosmologyFunctions(0)
        #Byt giving m * h to sigm_m gives the sigma_m at z=0 
        sigma_m0 = np.array([cosmo0.sigma_m(m*cosmo0._h) for m in marr])
        rho_norm = cosmo0.rho_bar()
        lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0)
        return rho_norm, lnMassSigmaSpl

    def halo_bias_st(self, nu):
	'''
	Eq. 8 in Sheth et al 2001
	'''
	common = 1./np.sqrt(0.707)/1.686
	fterm = np.sqrt(0.707) * 0.707 * nu #First term
	sterm = np.sqrt(0.707) * 0.5 * (0.707 * nu)**(1.-0.6) #second term
	ttermn = (0.707 * nu)**0.6 #numerator of third term
	ttermd = (0.707 * nu)**0.6 + 0.5 * (1.-0.6) * (1.-0.6/2.) #demoninator of third term
	tterm = ttermn / ttermd #third term
	blag = common * (fterm + sterm - tterm) #b_lag
	return 1+blag #b_eul

    def massfunction(self):
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
        cosmo = CosmologyFunctions(self.redshift)
        mass_array = np.logspace(np.log10(self.Mvir*0.9999), np.log10(self.Mvir*1.0001), 2)
        ln_mass_array = np.log(mass_array)
        ln_sigma_m_array = np.log(np.array([self.lnMassSigmaSpl(np.log(m)) for m in mass_array]))
        #spl = UnivariateSpline(ln_mass_array, ln_nu_array, k=3)
        #print spl.derivatives(np.log(Mvir))[1] 
        #Derivatives of dln_nu/dln_mass at ln_mass
        ln_sigma_m_ln_mass_derivative = abs((ln_sigma_m_array[1] - ln_sigma_m_array[0]) / (ln_mass_array[1] - ln_mass_array[0]))#1 for first derivate
        #print 'ln_sigma_m_ln_mass_derivative ', Mvir, ln_sigma_m_ln_mass_derivative

        if self.mf=='ST99':
            #This is (delta_c/sigma(m))^2. Here delta_c is slightly dependence on 
            #Omega_m across redshift. cosmo._growth is the growth fucntion
            #This means delta_c increases as a function of redshift
            #lnMassSigmaSpl returns the sigma(m) at z=0 when gives the log(Mvir) 
            nu = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / self.lnMassSigmaSpl(np.log(self.Mvir)) / self.lnMassSigmaSpl(np.log(self.Mvir))
            nu_d = 0.707 * nu
            nuf_1 = (1. + 1. / nu_d**0.3)
            nuf_2 = (2. * nu_d)**0.5
            nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
            nuf = 0.322 * nuf_1 * nuf_2 * nuf_3
        if self.mf=='CS02':
            #This is from CS02 paper not from ST99 paper. The halo mass function implimented in CHOMP is from ST99 paper
            p = 0.3
            A = 0.3222
            q = 0.75
            nuf_1 = A * (1. + (q * self.nu)**-0.3) 
            nuf_2 = (q * self.nu / (2. * np.pi))**0.5
            nuf_3 = np.exp(-q * self.nu / 2.)
            #nuf = nu*f(nu) 
            nuf = nuf_1 * nuf_2 * nuf_3
        if self.mf=='PS74':
            nuf = np.sqrt(1. / 2. / np.pi / self.nu) * np.exp(-self.nu / 2.)

        #print Mvir, rho_norm * cosmo._h * cosmo._h, cosmo.delta_c()/cosmo._growth, lnMassSigmaSpl(np.log(Mvir)),ln_sigma_m_ln_mass_derivative 
        mass_function = nuf * self.rho_norm * cosmo._h * cosmo._h * ln_sigma_m_ln_mass_derivative / self.Mvir
        if self.bias == 0:
	    return mass_function
        elif self.bias == 1:
	    return self.halo_bias_st(nu)
        elif self.bias == 2: 
            return mass_function * self.halo_bias_st(nu)
        else:
            return 


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


def bias_mass_func_st(redshift, lMvir, uMvir, mspace, bias=True, marr=None):
    '''
    Sheth & Torman 1999 & Eq. 56-50 of CS02 in p21
    Output:
         MF = dn/dlnMvir (1/Mpc^3)
         mass = Solar mass
    '''
    cosmo0 = CosmologyFunctions(0)
    cosmo_h = cosmo0._h

    if marr is not None:
        lnmarr = np.log(marr)
    else:
        dlnm = np.float64(np.log(uMvir/lMvir) / mspace)
        lnmarr = np.linspace(np.log(lMvir), np.log(uMvir), mspace)
        marr = np.exp(lnmarr).astype(np.float64)
    #print 'dlnm ', dlnm

    #No little h
    #Need to give mass * h and get the sigma without little h
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo0._h) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    #print marr, sigma_m0
    lnMassSigma0Spl = InterpolatedUnivariateSpline(lnmarr, sigma_m0, k=3)

    cosmo = CosmologyFunctions(redshift)

    mf,nuarr,fnuarr = [],[],[]
    for m in marr:
        mlow = m * 0.99
        mhigh = m * 1.01
        mass_array = np.logspace(np.log10(mlow), np.log10(mhigh), 2)
        ln_mass_array = np.log(mass_array)
        ln_sigma_m_array = np.log(np.array([lnMassSigma0Spl(np.log(m1)) for m1 in mass_array]))

        #Derivatives of dln_nu/dln_mass at ln_mass
        lnSigma_m_lnM_derivative = abs((ln_sigma_m_array[1] - ln_sigma_m_array[0]) / (ln_mass_array[1] - ln_mass_array[0]))#1 for first derivate

        nu = cosmo.nu_m(m)
 
        #This is (delta_c/sigma(m))^2. Here delta_c is slightly dependence on 
        #Omega_m across redshift. cosmo._growth is the growth fucntion
        #This means delta_c increases as a function of redshift
        #lnMassSigmaSpl returns the sigma(m) at z=0 when gives the log(Mvir) 
        delta_c_sigma_m2 = cosmo.delta_c() * cosmo.delta_c() / cosmo._growth / cosmo._growth / lnMassSigma0Spl(np.log(m)) / lnMassSigma0Spl(np.log(m))
        nu_d = 0.707 * delta_c_sigma_m2
        nuf_1 = (1. + 1. / nu_d**0.3)
        nuf_2 = (2. * nu_d)**0.5
        nuf_3 = np.exp(-nu_d / 2.) / np.sqrt(np.pi)
        nuf = 0.322 * nuf_1 * nuf_2 * nuf_3

        if bias:
            mf.append(halo_bias_st(delta_c_sigma_m2) * nuf * rho_norm0 * cosmo._h * cosmo._h * lnSigma_m_lnM_derivative / m)
        else:
            mf.append(nuf * rho_norm0 * cosmo._h * cosmo._h * lnSigma_m_lnM_derivative / m)
    mf = np.array(mf) 
    return marr, mf


def halo_bias_tinker(Delta, nu):
    '''
    Tinker et al 2010. Table 2 and Eq. 6
    '''
    #raise("Work in progress")
    y = np.log10(Delta)
    A = 1.0 + 0.24 * y * np.exp(-(4/y)**4)
    alpha = 0.44 * y - 0.88
    B = 0.183
    beta = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4/y)**4)
    ceta = 2.4
    delta = 1.686
    bias = 1. - A*(nu**alpha)/(nu**alpha + delta**alpha) + \
           B* nu**beta + C* nu**ceta
    return bias
 
def bias_mass_func_tinker(redshift, lM, uM, mspace, bias=True, Delta=200, mtune=False, marr=None, reduced=False):
    '''
    Wrote on Jan 26, 2017
    Mass function of mass determined from mean density
    of the Universe (i.e. omega_m(z) critical density)

    redshift : Redshift of mass function
    lM : Lower limit of marr
    uM : Upper limit of marr
    mspace : mass space
    bias : if weighted by ST bias (Doesn't work now)
    marr : Solar mass/h
    M200 -solar unit/h

    Both the output have no little h
    mass function in dn/dlnM200 in 1/Mpc^3
    marr solar unit / h 

    '''

    DeltaTinker = np.log((200.,300.,400.,600.,800.,1200.,1600.,2400.,3200.))

    # A
    ATinker = (1.858659e-01, 1.995973e-01, 2.115659e-01, 2.184113e-01, 2.480968e-01, 2.546053e-01, 2.600000e-01, 2.600000e-01, 2.600000e-01)
    Aspl = interp1d(DeltaTinker, ATinker, kind='cubic') #fill_value='extrapolate')

    # a
    aTinker = (1.466904e+00, 1.521782e+00, 1.559186e+00, 1.614585e+00, 1.869936e+00, 2.128056e+00, 2.301275e+00, 2.529241e+00, 2.661983e+00)
    aspl = interp1d(DeltaTinker, aTinker, kind='cubic') #fill_value='extrapolate')

    # b
    bTinker = (2.571104e+00, 2.254217e+00, 2.048674e+00, 1.869559e+00, 1.588649e+00, 1.507134e+00, 1.464374e+00, 1.436827e+00, 1.405210e+00)
    bspl = interp1d(DeltaTinker, bTinker, kind='cubic') #fill_value='extrapolate')

    # c
    cTinker = (1.193958e+00, 1.270316e+00, 1.335191e+00, 1.446266e+00, 1.581345e+00, 1.795050e+00, 1.965613e+00, 2.237466e+00, 2.439729e+00)
    cspl = interp1d(DeltaTinker, cTinker, kind='cubic') #fill_value='extrapolate')

    cosmo0 = CosmologyFunctions(0)
    cosmo_h = cosmo0._h

    if mtune:
        lnmarr = np.linspace(np.log(lM), np.log(1e13), 30)
        marr13 = np.exp(lnmarr).astype(np.float64)
        marr14 = np.linspace(2e13, 1e14, 10)
        marr15 = np.linspace(1.1e14, 1e15, 30)
        #marr16 = np.linspace(1.2e15, 1e16, 30)
        marr16 = np.linspace(1.1e15, 1e16, 30)
        #marr17 = np.linspace(2**16, 10**17, 10)
        marr = np.hstack([marr13, marr14, marr15, marr16])
        lnmarr = np.log(marr)
    elif marr is not None:
        lnmarr = np.log(marr)
    else:
        dlnm = np.float64(np.log(uM/lM) / mspace)
        lnmarr = np.linspace(np.log(lM), np.log(uM), mspace)
        marr = np.exp(lnmarr).astype(np.float64)
    #print 'dlnm ', dlnm
    #Should give mass in Msun/h to sigma_m() in CosmologyFunctions.py.
    #See the note in that function
    sigma_m0 = np.array([cosmo0.sigma_m(m) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    #print marr, sigma_m0
    if redshift >= 3:
        cosmo = CosmologyFunctions(3.)
    else:
        cosmo = CosmologyFunctions(redshift)

    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0*cosmo._growth, k=3)
    if 0:#Delta == 200:
        A = 0.186 * (1. + cosmo.redshift())**-0.14
        a = 1.47 * (1. + cosmo.redshift())**-0.06
        alpha = 10**(-(0.75/np.log10(200./75.))**1.2) 
        b = 2.57    * (1. + cosmo.redshift())**-alpha
        c = 1.19
    if 0:#Delta == 400:
        A = 0.212 * (1. + cosmo.redshift())**-0.14
        a = 1.56 * (1. + cosmo.redshift())**-0.06
        alpha = 10**(-(0.75/np.log10(400./75.))**1.2) 
        b = 2.05 * (1. + cosmo.redshift())**-alpha
        c = 1.34

    if Delta > 0:
        A = Aspl(np.log(Delta))
        a = aspl(np.log(Delta))
        b = bspl(np.log(Delta))
        c = cspl(np.log(Delta))
    else:
        A = 0.1 * np.log10(Delta) - 0.05
        a = 1.43 + (np.log10(Delta) - 2.3)**1.5
        b = 1.0 + (np.log10(Delta) - 1.6)**-1.5
        c = 1.2 + (np.log10(Delta) - 2.35)**1.6
    alpha = 10**(-(0.75/np.log10(Delta/75.))**1.2) 
    mf,sarr,fsarr = [],[],[]
    for tm in marr:
        mlow = tm * 0.99
        mhigh = tm * 1.01
        slow = lnMassSigmaSpl(np.log(mlow))
        shigh = lnMassSigmaSpl(np.log(mhigh))
        ds_dm = (shigh - slow) / (mhigh - mlow) #this will have one h
        sigma = lnMassSigmaSpl(np.log(tm))
        #print '%.2e %.2f %.2e'%(tm, sigma, ds_dm)

        fsigma = A * np.exp(-c / sigma**2.) * ((sigma/b)**-a + 1.)
        #print '%.2e %.2e %.2f %.2f %.2f %.2f %.2f'%(tm, fsigma, A, a, b, c, sigma)
        if reduced:
            mf.append(-1 * fsigma * rho_norm0 * ds_dm / sigma) #if need h^2/Mpc^3
        else:
            mf.append(-1 * fsigma * rho_norm0 * cosmo._h**3 * ds_dm / sigma) #It will have h^3/Mpc^3 if the dn/dlnM has M in Msol unit
        sarr.append(sigma)
        fsarr.append(fsigma)

    mf = np.array(mf)
    sarr = np.array(sarr)
    fsarr = np.array(fsarr)
    if 0:
        return marr, mass_function * halo_bias_st(delta_c_sigma_m2), sigma, fsigma
    else:
        return marr, mf, sarr, fsarr

def bias_mass_func_bocquet(redshift, lM200, uM200, mspace, bias=True, Delta=200, mtune=False, marr=None):
    '''
    Wrote on Feb 20, 2017
    Mass function of mass determined from critical density
    of the Universe 

    redshift : Redshift of mass function
    lM200 : Lower limit of M200c, i.e. critical M200
    uM200 : Upper limit of M200c
    mspace : mass space
    bias : if weighted by ST bias (Doesn't work now)

    M200 -solar unit/h

    Both the output have no little h
    mass function in dn/dlnM200 in 1/Mpc^3
    marr solar unit/h

    '''

    cosmo0 = CosmologyFunctions(0)
    cosmo_h = cosmo0._h

    g0 = 3.54e-2 + cosmo0._omega_m0**0.09
    g1 = 4.56e-2 + 2.68e-2/cosmo0._omega_m0
    g2 = 0.721 + 3.50e-2/cosmo0._omega_m0
    g3 = 0.628 + 0.164/cosmo0._omega_m0
    d0 = -1.67e-2 + 2.18e-2 * cosmo0._omega_m0
    d1 = 6.52e-3 - 6.86e-3 * cosmo0._omega_m0
    dd = d0 + d1 * redshift
    gg = g0 + g1 * np.exp(-1. * ((g2-redshift)/g3)**2.)

    if mtune:
        lnmarr = np.linspace(np.log(lM200), np.log(1e13), 30)
        marr13 = np.exp(lnmarr).astype(np.float64)
        marr14 = np.linspace(2e13, 1e14, 10)
        marr15 = np.linspace(1.1e14, 1e15, 30)
        #marr16 = np.linspace(1.2e15, 1e16, 30)
        marr16 = np.linspace(1.1e15, 1e16, 30)
        #marr17 = np.linspace(2**16, 10**17, 10)
        marr = np.hstack([marr13, marr14, marr15, marr16])
        lnmarr = np.log(marr)
    elif marr is not None:
        marr = marr / cosmo_h
        lnmarr = np.log(marr)
    else:
        dlnm = np.float64(np.log(uM200/lM200) / mspace)
        lnmarr = np.linspace(np.log(lM200), np.log(uM200), mspace)
        marr = np.exp(lnmarr).astype(np.float64)
    #print 'dlnm ', dlnm
    #No little h
    #Mass is in the unit of Solar/h and get the sigma without little h
    sigma_m0 = np.array([cosmo0.sigma_m(m * cosmo_h) for m in marr])
    rho_norm0 = cosmo0.rho_bar()
    #print marr, sigma_m0

    cosmo = CosmologyFunctions(redshift)
    lnMassSigmaSpl = InterpolatedUnivariateSpline(lnmarr, sigma_m0*cosmo._growth, k=3)
    if Delta == 200:
        A = 0.222 * (1. + cosmo.redshift())**0.269
        a = 1.71 * (1. + cosmo.redshift())**0.321
        b = 2.24 * (1. + cosmo.redshift())**-0.621
        c = 1.46 * (1. + cosmo.redshift())**-0.153
    if Delta == 500:
        raise Exception('Not implemeted')
        A = 0.241 * (1. + cosmo.redshift())**0.370
        a = 2.18 * (1. + cosmo.redshift())**0.251
        b = 2.35 * (1. + cosmo.redshift())**-0.698
        c = 2.02 * (1. + cosmo.redshift())**-0.310

    mf,sarr,fsarr = [],[],[]
    for M200 in marr:
        M1dM = gg + dd * np.log(M200)
        mlow = M200 * 0.99
        mhigh = M200 * 1.01
        slow = lnMassSigmaSpl(np.log(mlow))
        shigh = lnMassSigmaSpl(np.log(mhigh))
        ds_dm = (shigh - slow) / (mhigh - mlow)
        sigma = lnMassSigmaSpl(np.log(M200))
        #print '%.2e %.2f %.2e'%(M200, sigma, ds_dm)

        fsigma = A * np.exp(-c / sigma**2.) * ((sigma/b)**-a + 1.)
        #print '%.2e %.2e %.2f %.2f %.2f %.2f %.2f'%(M200, fsigma, A, a, b, c, sigma)
        mf.append(-1 * M1dM * fsigma * rho_norm0 * cosmo._h * cosmo._h * ds_dm / sigma)
        ##mf.append(-1 * fsigma * rho_norm0 * ds_dm / sigma) #if need h^2/Mpc^3
        sarr.append(sigma)
        fsarr.append(fsigma)

    mf = np.array(mf)
    sarr = np.array(sarr)
    fsarr = np.array(fsarr)
    if 0:
        return marr, mass_function * halo_bias_st(delta_c_sigma_m2), sigma, fsigma
    else:
        return marr, mf, sarr, fsarr


if __name__=='__main__':
    mf = []
    redshift = 1.
    mlow = 1e11
    mhigh = 5e15
    mspace = 100
    marr1 = np.logspace(np.log10(mlow), np.log10(mhigh), mspace)
    marr = np.logspace(np.log10(mlow), np.log10(mhigh), mspace)
    #for m in marr:
    #    mf.append(MassFunctionSingle(m, redshift, 0, mf='ST99').massfunction())

    #pl.loglog(marr, mf, label='Class')
    #pl.show()
    #marr, mf = bias_mass_func_st(redshift, mlow, mhigh, mspace, bias=True)
    #pl.loglog(marr, mf, label='Function')

    '''
    M200 = []
    cosmo = CosmologyFunctions(redshift)
    cosmo_h = cosmo._h
    BryanDelta = cosmo.BryanDelta() 
    rho_critical = cosmo.rho_crit() * cosmo_h * cosmo_h
    for m in marr:
        M200.append(MvirToMRfrac(m, redshift, BryanDelta, cosmo.omega_m() * rho_critical * cosmo._h * cosmo._h, cosmo_h, frac=200.0)[2])
    M200 = np.array(M200)
    print M200.min(), M200.max()
    m2001 = np.logspace(np.log10(M200.min()), np.log10(M200.max()), mspace)
    '''
    M200 = marr.copy()
    marr, mf, sarr, fsarr = bias_mass_func_tinker(redshift, M200.min(), M200.max(), mspace, marr=marr, reduced=0) #np.array([1e11, 1e12, 1e13, 1e14, 1e15]))
    print marr.min(), marr.max()
    np.savetxt('../data/vmf.dat', np.transpose((marr, mf/marr)))
    print marr[0], mf[0] 
    #I think my code give M/h and the code from Jeremy Tinker and HMF is also agrees with my code. However, it should be checked, i.e. not very sure whether the mass in this function bias_mass_func_tinker() gives M*h or M 
    #pl.loglog(marr1, mf, label='Tinker-Vinu1')
    #pl.loglog(m2001, mf, label='Tinker-Vinu2001', lw=3)
    pl.loglog(marr, mf, label='Tinker-Vinu', lw=3)
    #marr, mf, sarr, fsarr = bias_mass_func_bocquet(redshift, M200.min(), M200.max(), mspace, marr=None) #np.array([1e11, 1e12, 1e13, 1e14, 1e15]))
    #pl.loglog(marr, mf, label='Bocquet-Vinu', lw=3)
    f = np.genfromtxt('../hmf/mVector_PLANCK-SMT z: 0.5.txt')
    pl.loglog(f[:,0], f[:,6]*0.70**3, label='HMF')
    f = np.genfromtxt('testz1p0.dndM')
    mspl = InterpolatedUnivariateSpline(f[:,0], f[:,0]*f[:,1]*0.71**2) 
    pl.loglog(f[:,0], f[:,0]*f[:,1]*0.7**2, label='Jeremy')
    pl.xlabel(r'$M_\odot/h$')
    pl.ylabel(r'$Mpc^{-3}$')
    pl.legend(loc=0)
    pl.savefig('../figs/compare_mf_jeremy_vinu.png', bbox_inches='tight')
    pl.show()
