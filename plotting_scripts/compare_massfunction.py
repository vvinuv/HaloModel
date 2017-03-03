import numpy as np
import pylab as pl

pl.figure(figsize=[12,6])
cm = np.genfromtxt('/media/luna1/vinu/software/cosmosis/mft_output/mass_function/m_h.txt')
cmf = np.genfromtxt('/media/luna1/vinu/software/cosmosis/mft_output/mass_function/dndlnmh.txt')
h = 0.71

pl.subplot(121)
pl.loglog(cm, cmf[0], label='COSMOSIS z=0')

hf = np.genfromtxt('../hmf/mVector_PLANCK-SMT z: 0.0.txt')
pl.loglog(hf[:,0], hf[:,7], label='HMF z=0 dn/dlog10m')
pl.xlabel(r'$M_\odot/h$')
pl.ylabel(r'dn/dlnm $h^3 Mpc^{-3}$')
pl.xlim([1e10,1e16])
pl.ylim([1e-14,1e1])
pl.legend(loc=0)


pl.subplot(122)
pl.loglog(cm, cmf[0], label='COSMOSIS z=0')
hf = np.genfromtxt('../hmf/mVector_PLANCK-SMT z: 0.0.txt')
pl.loglog(hf[:,0], hf[:,6], label='HMF z=0 dn/dlnm')
pl.xlabel(r'$M_\odot/h$')
pl.ylabel(r'dn/dlnm $h^3 Mpc^{-3}$')
pl.xlim([1e10,1e16])
pl.ylim([1e-14,1e1])
pl.legend(loc=0)
pl.savefig('../figs/compare_hmf_cosmosis_tinker.png', bbox_inches='tight')
pl.show()
