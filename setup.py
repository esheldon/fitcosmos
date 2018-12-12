import os
from glob import glob
from distutils.core import setup

scripts=glob('bin/*')
scripts = [s for s in scripts if '~' not in s]


setup(
    name="fitcosmos", 
    version="0.1.0",
    description="Code to fit models to objects in COSMOS+DES",
    license = "GPL",
    author="Erin Scott Sheldon",
    author_email="erin.sheldon@gmail.com",
    scripts=scripts,
    packages=['fitcosmos'],
)
