import numpy as np
import pylab as pl
from halomodel_cl_WL_tSZ import cl_WL_tSZ
import config

gx =6.7354 #For 150 GHz
ellarr, cl1h, cl2h, cl = cl_WL_tSZ(config.fwhm, config.kk, config.yy, config.ky, config.zsfile, odir='../data')
Dl = 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi
b = np.genfromtxt('../data/battaglia_analytical.csv', delimiter=',')
pl.plot(b[:,0], b[:,1], c='r', label='Battaglia')
pl.plot(ellarr, Dl*gx, c='k', label='This code')
pl.legend(loc=0)
pl.xlabel(r'$\ell$')
pl.ylabel(r'$D_\ell$')
pl.savefig('demo_battaglia_this.png', bbox_inches='tight')
pl.show()
