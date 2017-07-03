import os
import numpy as np
import correlation

#This models kappa-SZ halo model correlation for different redshift distributions
if 1:
    for zsfile in os.listdir("."):
        if zsfile.startswith("source_distribution_new_"):
            zstr = zsfile[-8:-4]
            print zstr 
            if zstr=='z0p4':
                ofile = 'xi_ky_%s_5rvir.dat'%zstr
                #ofile = 'xi_ky_BH_%s.dat'%zstr
                rarcmin, xi1h, xi2h, xi = correlation.xi_wl_tsz(rmin=1e-2, 
                    rmax=150, rbin=100, fwhm_k=1., fwhm_y=10., kk=False, 
                    yy=False, ky=True, zsfile=zsfile, omega_m0=0.25, 
                    sigma_8=0.8, P01=18.1, P02=0.154, P03=-0.758, xc1=0.497, 
                    xc2=-0.00865, xc3=0.731, beta1=4.35, beta2=0.0393, 
                    beta3=0.415, default_pp=False, paramsfile='wlxtsz.ini', 
                    odir='../data',  ofile=ofile) 
if 0:
    zsfile = 'source_distribution_cori.txt'
    ofile = 'xi_ky_cori.dat'
    rarcmin, xi1h, xi2h, xi = correlation.xi_wl_tsz(rmin=1e-2, rmax=150,
                                                    rbin=100, fwhm_k=0., 
                                                    fwhm_y=0.,
                                                    kk=False, yy=False, 
                                                    ky=True, zsfile=zsfile, 
                                                    ofile=ofile) 
