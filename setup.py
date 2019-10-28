from distutils.core import setup

setup(
	name='pyobdm',
	version='1.2',
	description='OBDM Observers Based Data Modeling',
	author='Fares Meghdouri',
	author_email='fares.meghdouri@tuwien.ac.at',
	url='https://github.com/CN-TU/pyobdm',
	packages=['pyobdm'],
	requires=['numpy',  'scipy']
)