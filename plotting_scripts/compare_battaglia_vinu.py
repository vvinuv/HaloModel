import numpy as np
import pylab as pl
from  astropy.io import fits

f = fits.open('/media/luna1/flender/projects/gasmod/maps/OuterRim/cl_tsz150_Battaglia_c05_R13.fits')[1].data
l = np.arange(10000)
pl.semilogx(l, l*(l+1)*f['TEMPERATURE'][1:]/2./np.pi, label='Simulation')
b = np.genfromtxt('/media/luna1/vinu/github/HaloModel/data/battaglia_analytical.csv', delimiter=',')
pl.semilogx(b[:,0], b[:,1], label='Battaglia')
v = np.genfromtxt('/media/luna1/vinu/github/HaloModel/data/cl_yy.dat')
pl.semilogx(v[:,0], v[:,0]*(1.+v[:,0])*v[:,1]*1e12*6.7354/2./np.pi, label='Vinu')
pl.xlim(500,10000)
pl.xlabel(r'$\ell$')
pl.ylabel(r'$D_\ell$')
pl.legend(loc=0)
pl.savefig('../figs/compare_battaglia_vinu_simulation.png', bbox_inches='tight')
pl.show()
