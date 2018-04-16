import numpy as np
from numba import jit

@jit(nopython=True)
def Wk_one(zl, chil, zsarr, chisarr, Ns, constk):
    #zl = lens redshift
    #chil = comoving distant to lens
    #zsarr = redshift distribution of source
    #Ns = Normalized redshift distribution of sources 
    al = 1. / (1. + zl)
    Wk = constk * chil / al
    gw = 0.0
    if chisarr < chil  or  chisarr == 0:
        pass
    else:
        gw += ((chisarr - chil) * Ns / chisarr)
    Wk = Wk * gw
    return Wk

@jit(nopython=True)
def Wkcom(zl, chil, zsarr, chisarr, Ns, constk):
    ''' zl = lens redshift
        chil = comoving distant to lens
        zsarr = redshift distribution of source
        Ns = Normalized redshift distribution of sources 
    '''
    al = 1. / (1. + zl)
    Wk = constk * chil / al
    gw = 0.0
    for i, N in enumerate(Ns):
        if chisarr[i] < chil  or  chisarr[i] == 0:
            continue
        gw += ((chisarr[i] - chil) * N / chisarr[i])
    if i == 0: 
       pass
    else:
        gw *= (zsarr[1] - zsarr[0])
    #gw *= (chisarr[1] - chisarr[0])
    Wk = Wk * gw
    return Wk


@jit(nopython=True)
def Wkcom_ang(zl, chil, zsarr, chisarr, Ns, constk):
    ''' zl = lens redshift
        chil = comoving distant to lens
        zsarr = redshift distribution of source
        Ns = Normalized redshift distribution of sources 
    '''
    al = 1. / (1. + zl)
    angl = chil * al
    angsarr = chisarr / (1. + zsarr)
    Wk = constk * angl / al
    gw = 0.0
    for i, N in enumerate(Ns):
        if chisarr[i] < chil  or  chisarr[i] == 0:
            continue
        gw += ((chisarr[i] - chil) * N / angsarr[i])
    if i == 0: 
       pass
    else:
        gw *= (zsarr[1] - zsarr[0])
    #gw *= (chisarr[1] - chisarr[0])
    Wk = Wk * gw
    return Wk

def waerbeke():
    import pylab as pl
    from CosmologyFunctions import CosmologyFunctions
    '''
    Weight function should use the angular coordinate which matches with the 
    plot from Waerbeke et al, 2014
    '''
    cosmo = CosmologyFunctions(0., 'wlsz.ini', 'battaglia')
    omega_m0 = cosmo._omega_m0
    cosmo_h = cosmo._h
    light_speed = 3e5 #km/s
    constk = 3. * omega_m0 * (cosmo_h * 100. / light_speed)**2. / 2.

    zsarr, W = np.genfromtxt('waerbeke_weight.dat', unpack=True)
    pl.plot(zsarr, W, label='CFHT')

    zsarr, Ns = np.genfromtxt('waerbeke_zdist.dat', unpack=True)
    dz = zsarr[1] - zsarr[0]
    Nsum = Ns.sum() * dz
    Ns /= Nsum

    zlarr = zsarr.copy()
    chilarr = np.array([CosmologyFunctions(zi, 'wlsz.ini', 'battaglia').comoving_distance() for zi in zlarr]) / 0.7

    chisarr = chilarr.copy()
    
    Wk = np.array([Wkcom(zl, chil, zsarr, chisarr, Ns, constk) for zl, chil in zip(zlarr, chilarr)])
    Wksum = Wk.sum() * dz
    Wk /= Wksum
    pl.plot(zlarr, Wk, label='Comoving distance')

    Wk = np.array([Wkcom_ang(zl, chil, zsarr, chisarr, Ns, constk) for zl, chil in zip(zlarr, chilarr)])
    Wksum = Wk.sum() * dz
    Wk /= Wksum
    pl.plot(zlarr, Wk, label='Comoving - Angular distance')


    pl.legend(loc=0)
    pl.show()


def des():
    import pylab as pl
    from CosmologyFunctions import CosmologyFunctions
    from scipy.interpolate import interp1d
    import mytools

    mytools.matrc_small()
    cosmo = CosmologyFunctions(0., 'wlsz.ini', 'battaglia')
    omega_m0 = cosmo._omega_m0
    cosmo_h = cosmo._h
    light_speed = 3e5 #km/s
    constk = 3. * omega_m0 * (cosmo_h * 100. / light_speed)**2. / 2.
    colors = pl.cm.jet(np.linspace(0, 1, 10)) 
    mz = np.linspace(0.001, 1.5, 100)
    dz = mz[1] - mz[0]
    mchi = np.array([CosmologyFunctions(zi, 'wlsz.ini', 'battaglia').comoving_distance() for zi in mz]) / cosmo_h
    splzchi = interp1d(mz, mchi)
    ax = pl.subplot(111) 
    zdict = {0:'z0p1', 1:'z0p2', 2:'z0p3', 3:'z0p4', 4:'z0p5', 5:'z0p6', 6:'z0p7', 7:'z0p8', 8:'z0p9', 9:'z1p0'}
    #zdict = {0:'z0p4'}
    for i in np.arange(10):
        zs, N = np.genfromtxt('source_distribution_new_%s.txt'%zdict[i], unpack=1)
        N = N/( N.sum() * (zs[1] - zs[0]))
        chis = splzchi(zs)
        Wk = np.array([Wkcom(zl, chil, zs, chis, N, constk) for zl, chil in zip(mz, mchi)])
        Wksum = Wk.sum() * dz
        print Wksum
        Wk /= Wksum

        pl.plot(zs, N, label=r'$z > %.1f$'%float(zdict[i][1:].replace('p', '.')), ls='--', c=colors[i])
        pl.plot(mz, Wk, ls='-', c=colors[i])
    pl.xlabel(r'$z$')
    pl.ylabel(r'$N(z)$') 
    pl.legend(loc=0)
    ax2 = ax.twinx()
    pl.ylabel(r'$W(z)$')  
    #ax.yaxis.set_label_position("right")
    pl.savefig('fig_zdist_weighting.pdf', bbox_inches='tight')
    pl.show()
            

def lens_astropy():
    import pylab as pl
    from astropy.cosmology import WMAP9
    zl = np.linspace(0.01, 1.9, 100)
    zs = 2.
    wa = []
    wc = []
    angs = WMAP9.angular_diameter_distance(zs).value
    chis = WMAP9.comoving_distance(zs).value
    for zi in zl:
        angl = WMAP9.angular_diameter_distance(zi).value
        angls = WMAP9.angular_diameter_distance_z1z2(zi,zs).value
        chil = WMAP9.comoving_distance(zi).value
        chils = WMAP9.comoving_distance(zs).value-WMAP9.comoving_distance(zi).value
        wa.append(angl * angls / angs)
        wc.append(chil * chils / chis / (1+zi))
    pl.plot(zl, wa)
    pl.plot(zl, wc, ls='--')
    pl.show()

if __name__=='__main__':
    #waerbeke()
    des()
    #lens_astropy()
