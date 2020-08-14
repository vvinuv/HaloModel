from setuptools import setup

setup(name='pyhalowlsz',
      version='0.1',
      description='Estimating the correlation function of lensing vs SZ using halo model',
      url='https://github.com/vvinuv/HaloModel',
      author='Vinu Vikraman, Samuel Flender',
      author_email='vvinuv@gmail.com',
      license='MIT',
      packages=['halowlsz'],
#      scripts=['halowlsz/wlsz_corr.py', 'halowlsz/halomodel_cl_WL_tSZ.py', 'halowlsz/mass_function.py', 'halowlsz/convert_NFW_RadMass.py', 'halowlsz/CosmologyFunctions.py', 'halowlsz/lensing_efficiency.py', 'halowlsz/pressure_profiles.py'],
      zip_safe=False)

