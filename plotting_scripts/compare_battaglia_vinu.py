import numpy as np
import pylab as pl
from  astropy.io import fits
from scipy.interpolate import interp1d

sigma_y = 0 * np.pi / 2.355 / 60. /180. #angle in radian
sigmasq = sigma_y * sigma_y

#f = fits.open('/media/luna1/flender/projects/gasmod/maps/OuterRim/cl_tsz150_Battaglia_c05_R13.fits')[1].data
#l = np.arange(10000)
#pl.semilogx(l, l*(l+1)*f['TEMPERATURE'][1:]/2./np.pi, label='Simulation')
bl, bcl = np.genfromtxt('/media/luna1/vinu/github/HaloModel/data/battaglia_analytical.csv', delimiter=',', unpack=True)
Bl = np.exp(-bl*bl*sigmasq)
bclsm = bcl*Bl
bclsm = bclsm *2*np.pi/ bl / (bl+1) /6.7354
#pl.semilogx(bl, bclsm, label='Battaglia')
pl.loglog(bl, bclsm, label='Battaglia')
vl, vcl1, vcl2, vcl = np.genfromtxt('/media/luna1/vinu/github/HaloModel/data/cl_yy.dat', unpack=True)
Dl = vl*(1.+vl)*vcl1*1e12*6.7354/2./np.pi
Dl = vcl1*1e12
Bl = np.exp(-vl*vl*sigmasq)
spl = interp1d(vl, Dl*Bl)
pl.figure(1)
#pl.semilogx(vl, Dl*Bl, label='Vinu')
pl.loglog(vl, Dl*Bl, label='Vinu')
pl.xlim(500,10000)
pl.xlabel(r'$\ell$')
pl.ylabel(r'$D_\ell$')
pl.legend(loc=0)
pl.savefig('../figs/compare_battaglia_vinu_simulation.png', bbox_inches='tight')
pl.figure(2)
pl.plot(bl, (bclsm-spl(bl))/spl(bl), label='Battaglia/Vinu')
pl.xlabel(r'$\ell$')
pl.ylabel('Battaglia/Vinu')
pl.show()
