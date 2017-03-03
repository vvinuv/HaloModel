import numpy as np
import pylab as pl

#Here I used zarr = np.linspace(0.05, 1, 50) which is consistent with Nan's simulation
colors = pl.cm.jet(np.linspace(0, 1, 8))
fn = np.genfromtxt('../data/psd_emulator.dat')
pl.loglog(fn[:,0], fn[:,1], c=colors[0], label='Nan power')
for r in np.arange(1, 8):
    f = np.genfromtxt('../data/cl_kk_%dRvir.txt'%r)
    pl.loglog(f[:,0], f[:,3], c=colors[r], label='Rvir=%d'%r)
pl.legend(loc=0)
pl.ylabel(r'$C_\ell$')
pl.xlabel(r'$\ell$')
pl.savefig('limit_kk_Rvir.png', bbox_inches='tight')


#Convert y to \delta_T using 147 GHz. (g(x) TCMB)^2 = 7.2786
files = ['cl_yy_1Rvir_mmin5e13.dat', 'cl_yy_2Rvir_mmin5e13.dat', 'cl_yy_3Rvir_mmin5e13.dat', 'cl_yy_4Rvir_mmin5e13.dat', 'cl_yy_5Rvir_mmin5e13_dx_20.dat','cl_yy_5Rvir_mmin5e13_dx_20.dat', 'cl_yy_6Rvir_mmin5e13.dat', 'cl_yy_7Rvir_mmin5e13.dat', 'cl_yy_8Rvir_mmin5e13_dx_100.dat']


pl.figure(2)
pl.subplot(131)
pl.title('M=5e10-1e16')
colors = pl.cm.jet(np.linspace(0, 1, 10))
b = np.genfromtxt('battaglia_analytical.csv', delimiter=',')
pl.plot(b[:,0], b[:,1], label='Battaglia Analytical')
Rvir = [1, 2, 3, 4, 5, 5, 6, 7, 8]
i = 1
for r,fi in zip(Rvir, files):
    f = np.genfromtxt(fi)
    ellarr = f[:,0]
    cl1h = f[:,1]
    cl2h = f[:,2]
    cl = f[:,3]
    cl *= 7.2786
    cl1h *= 7.2786
    cl2h *= 7.2786
    pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi, label='Rvir=%d'%r, c=colors[i])
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
    i += 1
pl.xlabel(r'$\ell$')
pl.ylabel(r'$c_\ell \ell (\ell + 1)/2/\pi \mu K^2$')
pl.legend(loc=0)

#Convert y to \delta_T using 147 GHz. (g(x) TCMB)^2 = 7.2786
files = ['cl_yy_1Rvir_mmin10.dat', 'cl_yy_2Rvir_mmin10.dat', 'cl_yy_3Rvir_mmin10.dat', 'cl_yy_4Rvir_mmin10.dat', 'cl_yy_5Rvir_mmin10_dx_20.dat','cl_yy_5Rvir_mmin10_dx_20.dat', 'cl_yy_6Rvir_mmin10.dat']
pl.subplot(132)
pl.title('M=1e10-1e16')
pl.plot(b[:,0], b[:,1], label='Battaglia Analytical')
Rvir = [1, 2, 3, 4, 5, 5, 6]
i = 1
for r,fi in zip(Rvir, files):
    f = np.genfromtxt(fi)
    ellarr = f[:,0]
    cl1h = f[:,1]
    cl2h = f[:,2]
    cl = f[:,3]
    cl *= 7.2786
    cl1h *= 7.2786
    cl2h *= 7.2786
    pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi, label='Rvir=%d'%r, c=colors[i])
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
    i += 1

pl.xlabel(r'$\ell$')
pl.ylabel(r'$c_\ell \ell (\ell + 1)/2/\pi \mu K^2$')
pl.legend(loc=0)

#Convert y to \delta_T using 147 GHz. (g(x) TCMB)^2 = 7.2786
files = ['cl_yy_1Rvir_mmin5e13_dx_0.5.dat', 'cl_yy_2Rvir_mmin5e13_dx_0.5.dat', 'cl_yy_3Rvir_mmin5e13_dx_0.5.dat', 'cl_yy_4Rvir_mmin5e13_dx_0.5.dat', 'cl_yy_5Rvir_mmin5e13_dx_0.5.dat','cl_yy_5Rvir_mmin5e13_dx_0.1.dat']


pl.subplot(133)
pl.title('M=5e10-1e16')
colors = pl.cm.jet(np.linspace(0, 1, 10))
b = np.genfromtxt('battaglia_analytical.csv', delimiter=',')
pl.plot(b[:,0], b[:,1], label='Battaglia Analytical')
Rvir = [1, 2, 3, 4, 5, 5, 6, 7, 8]
i = 1
for r,fi in zip(Rvir, files):
    f = np.genfromtxt(fi)
    ellarr = f[:,0]
    cl1h = f[:,1]
    cl2h = f[:,2]
    cl = f[:,3]
    cl *= 7.2786
    cl1h *= 7.2786
    cl2h *= 7.2786
    pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl / 2. / np.pi, label='Rvir=%d'%r, c=colors[i])
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl1h / 2. / np.pi, label='Cl1h')
    #pl.plot(ellarr, 1e12 * ellarr * (ellarr+1) * cl2h / 2. / np.pi, label='Cl2h')
    i += 1
pl.xlabel(r'$\ell$')
pl.ylabel(r'$c_\ell \ell (\ell + 1)/2/\pi \mu K^2$')
pl.legend(loc=0)



pl.savefig('limit_yy_Rvir.png', bbox_inches='tight')

pl.show()
