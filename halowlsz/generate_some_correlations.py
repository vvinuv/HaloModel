import os
import numpy as np
import correlation

#This models kappa-SZ halo model correlation for different redshift distributions
if 1:
    for zsfile in os.listdir("."):
        if zsfile.startswith("source_distribution_new_"):
            zstr = zsfile[-8:-4]
            print zstr 
            if not zstr=='1z0p4':
                ofile = 'xi_ky_%s_2rvir.dat'%zstr
                #ofile = 'xi_ky_BH_%s.dat'%zstr
                rarcmin, xi1h, xi2h, xi = correlation.xi_wl_tsz(rmin=1e-2, rmax=150, 
                                                            rbin=100, fwhm_k=1., 
                                                            fwhm_y=10.,
                                                            kk=False, yy=False, 
                                                            ky=True, zsfile=zsfile, 
                                                            ofile=ofile) 
if 0:
    zsfile = 'source_distribution_cori.txt'
    ofile = 'xi_ky_cori.dat'
    rarcmin, xi1h, xi2h, xi = correlation.xi_wl_tsz(rmin=1e-2, rmax=150,
                                                    rbin=100, fwhm_k=0., 
                                                    fwhm_y=0.,
                                                    kk=False, yy=False, 
                                                    ky=True, zsfile=zsfile, 
                                                    ofile=ofile) 
