import os
import sys

from setuptools import setup


# Check if pyhalowlsz could run on the given system
if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: {}.'.format(sys.platform)
    )

if sys.version_info < (3, 5):
    raise ValueError(
        'Unsupported Python version {}.{}.{} found. pyhalowlsz requires Python 3.5 or higher.'.format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, 'requirements.txt')) as f:
    install_reqs = [r.rstrip() for r in f.readlines()]

with open("__version__") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

with open('README.md') as f:
    long_description = f.read()


setup(name='halowlsz',
      version=version,
      author='Vinu Vikraman, Samuel Flender',
      author_email='vvinuv@gmail.com',
      description='Estimating the correlation function of lensing vs SZ using halo model',
      install_requires=install_reqs,
      packages=find_packages(),
      include_package_data=True,
      url='https://github.com/vvinuv/halowlsz',
      license='MIT',
      packages=['halowlsz'],
          platforms=['Linux'],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: Linux/MacOS Independent",
        "Topic :: Scientific/Engineering :: Experimental Analysis",
        'Programming Language :: Python :: 3.6',
    ],
    zip_safe=False,
    python_requires='>=3.6.*'
    )

