from distutils.core import setup

setup(
	name='pyodm',
	version='3',
	description='ODM Observers-based Data Modeling',
	author='Fares Meghdouri',
	author_email='fares.meghdouri@tuwien.ac.at',
	url='https://github.com/CN-TU/pyodm',
	packages=['pyodm'],
	requires=['numpy',  'scipy', 'joblib', 'multiprocess']
)