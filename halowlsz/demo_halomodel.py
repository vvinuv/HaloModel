import numpy as np
import pylab as pl
from halomodel_cl_WL_tSZ import cl_WL_tSZ
import config

gx =6.7354 #For 150 GHz
ellarr, cl1h, cl2h, cl = cl_WL_tSZ(config.fwhm, config.kk, config.yy, config.ky, config.zsfile, odir='../data')

#Battaglia el al
b = np.genfromtxt('../data/battaglia_analytical.csv', delimiter=',')
pl.plot(b[:,0], b[:,1], c='r', label='Battaglia')

#Using mass function in config.py
Dl = 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi
pl.plot(ellarr, Dl*gx, c='k', label='This code')

#Using M400m as mass function
f4 = np.genfromtxt('../data/cl_yy_virial_mf_400m.dat')
ellarr = f4[:,0]
Dl400m = f4[:,3] * 1e12 * ellarr * (ellarr+1) / 2. / np.pi
pl.plot(ellarr, Dl400m*gx, c='g', label='This code MF=400m')

pl.legend(loc=0)
pl.xlabel(r'$\ell$')
pl.ylabel(r'$D_\ell$')
pl.savefig('demo_battaglia_this.png', bbox_inches='tight')
pl.show()
