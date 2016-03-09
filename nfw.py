import numpy as np
import pylab as pl
import config

class NFW:
    '''This class estimates NFW parameters. It uses Bryan & Norman 1998 
       Maccio et al. (2007) and Coe 2010    
    '''

    def __init__(self, redshift, mass, NM=False, print_mode=False):
        self.initilize(redshift, mass, NM, print_mode)    

    def initilize(self, redshift, mass, NM, print_mode):
        self.redshift = redshift
        self.Mvir = mass
        self.h = config.default_cosmo_dict['h']
        self.G = 4.302e-9 / self.h #Mpc h^-1 Mo^-1 (km/s)^2 
        self.H0 = 100. #h km/(Mpc s)
        self.rho_c = (3 * self.H0**2.) / (8 * np.pi * self.G) # This is the critical density 3*H_0^2/(8*pi*G) (Mo h^3 / Mpc^3)
        self.Omega0 = config.default_cosmo_dict['omega_m0'] # nonrelativistic matter density over critical density 
        self.OmegaR = 0.0 # Radius of curvature
        self.OmegaL = config.default_cosmo_dict['omega_l0'] # Lambda 
        self.rho_m = self.Omega0 * self.rho_c # nonrelativistic matter density (solar h^3 / Mpc^3)

        self.GetRvir()
        self.concentration()            
        #central dentsity rho_s
        self.rho_s = self.rho_c * (self.Delta_c() / 3.) * self.conc**3. / (np.log(1 + self.conc) - self.conc / (1 + self.conc)) #Solar h^3 / Mpc^3
        self.Rs = self.Rvir / self.conc #Mpc h^(-1) 

        if NM:
            self.M500, self.R500 = self.RvirToRNM(NM)
        else:
            self.M500, self.R500 = self.RvirToR500()

        if print_mode:
            print ' h = %2.2f \n G=%2.2e Mpc Mo^-1 (km/s)^2'%(self.h, self.G)
            print ' H0=%2.2e km/(Mpc s) \n rho_c=%2.2e Mo / Mpc^3'%(self.H0, self.rho_c)
    def Ez(self):
        return np.sqrt(self.Omega0 * (1 + self.redshift)**3. + \
                       self.OmegaR * (1 + self.redshift)**2. + self.OmegaL)

    def Hz(self):
        return self.H0 * self.Ez()

    def Omegaz(self):
        return self.Omega0 * (1 + self.redshift)**3. / self.Ez()**2.

    def Delta_c(self):
        x = self.Omegaz() - 1.
        if self.OmegaR == 0:
            return 18 * np.pi * np.pi + 82. * x - 39. * x * x
        elif self.OmegaL == 0:
            return 18 * np.pi * np.pi + 62. * x - 32. * x * x
        else:
            raise ValueError('Given cosmology is not implemented')

    def concentration(self):
        '''Maccio 07 - Eq. 8. I also found that it is only within 1e9 < M < 5e13. What is this Delta_vir=98 in page 57'''
        Mvir = self.Mvir / self.h
        self.conc = 10**(1.02 - 0.109 * (np.log10(Mvir) - 12.))

    def GetRvir(self):
        '''Rvir is in Mpc h^(-1) and Mvir in solar mass'''
        self.Rvir = (self.Mvir / ((4 * np.pi / 3.) * self.Delta_c() * self.rho_c))**(1/3.)

    def RvirToR500(self):
 
        f_R500 = lambda R500: (500 * self.rho_c * R500**3. / 3.) - \
                              (self.rho_s * self.Rs**3) * \
                              (np.log((self.Rs + R500) / self.Rs) - \
                               R500 / (self.Rs + R500))
        df_R500 = lambda R500: (500 * self.rho_c * R500**2.) - \
                               (self.rho_s * self.Rs**3) * \
                               (R500 / (self.Rs + R500)**2.)

        # Using Newton - Raphson method. x1 = x0 - f(x0) / f'(x0) where x0 is
        # the initial guess, f and f' are the function and derivative
 
        # Intial guess is Rvir / 2. 
        x0 = self.Rvir / 2.0
        tol = self.Rvir * 1.0e-6 #tolarence
        x1 = tol * 10**6

        while abs(x0 - x1) > tol:
            x1 = x0 - f_R500(x0) / df_R500(x0)
            x0 = x1 * 1.0
        R500 = x1
        M500 = (4. / 3.) * np.pi * R500**3 * 500. * self.rho_c
        return M500, R500

    def RvirToRNM(self, NM):
        '''Function to find radius included mass of NM times average 
           density. 
        '''

        f_RNM = lambda RNM: (NM * self.rho_c * RNM**3. / 3.) - \
                              (self.rho_s * self.Rs**3) * \
                              (np.log((self.Rs + RNM) / self.Rs) - \
                               RNM / (self.Rs + RNM))
        df_RNM = lambda RNM: (NM * self.rho_c * RNM**2.) - \
                               (self.rho_s * self.Rs**3) * \
                               (RNM / (self.Rs + RNM)**2.)

        # Using Newton - Raphson method. x1 = x0 - f(x0) / f'(x0) where x0 is
        # the initial guess, f and f' are the function and derivative
 
        # Intial guess is Rvir / 2. 
        x0 = self.Rvir / 2.0
        tol = self.Rvir * 1.0e-6 #tolarence
        x1 = tol * 10**6

        while abs(x0 - x1) > tol:
            x1 = x0 - f_RNM(x0) / df_RNM(x0)
            x0 = x1 * 1.0
        RNM = x1
        MNM = (4. / 3.) * np.pi * RNM**3 * 500. * self.rho_c
        return MNM, RNM



if __name__=='__main__':
    redshift = 0.1
    mass = 1e13
    n = NFW(redshift, mass, NM=False, print_mode=False) 

    print 'rho_crit > %.2e Solar h^3 Mpc^(-3)'%(n.rho_c)
    print 'M500 > %.2e Solar R500 > %.2e h^-1 Mpc'%(n.M500, n.R500)
    print 'Rs > %.2e h^-1 Mpc rho_s > %.2e Solar h^3 Mpc^-3'%(n.Rs, n.rho_s)

