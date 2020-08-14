import os
import numpy as np
import correlation

#This models kappa-SZ halo model correlation for different redshift distributions
pdict = {'s8':0.8, 'om':0.25, 'P01':18.1, 'P02':0.154, 'P03':-0.758, 'xc1':0.497, 'xc2':-0.00865, 'xc3':0.731, 'beta1':4.35, 'beta2':0.0393, 'beta3':0.415}

for key, value in pdict.items():
    for i, v in enumerate([0.95*value, 1.05*value, value]):
        pdict[key] = v
        print(key, pdict[key])
        oxifile = 'xi_ky_%s_%d_5rvir.dat'%(key, i)
        oclfile = 'cl_ky_%s_%d_5rvir.dat'%(key, i)
        rarcmin, xi1h, xi2h, xi = correlation.xi_wl_tsz(config_file='wlsz.ini',
                                  cosmology='battaglia', rmin=1e-2, rmax=150,
                                  rbin=100, fwhm_k=1., fwhm_y=10.,
                                  kk=False, yy=False, ky=True,
                                  zsfile='source_distribution_new_z0p4.txt',
                                  P01=pdict['P01'], P02=pdict['P02'], 
                                  P03=pdict['P03'], 
                                  xc1=pdict['xc1'], xc2=pdict['xc2'], 
                                  xc3=pdict['xc3'], 
                                  beta1=pdict['beta1'], beta2=pdict['beta2'], 
                                  beta3=pdict['beta3'], 
                                  omega_m0=pdict['om'], sigma_8=pdict['s8'],
                                  odir='../data', oxifile=oxifile, 
                                  oclfile=oclfile)
