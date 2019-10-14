from distutils.core import setup

setup(
	name='pydmc',
	version='1.0',
	description='DMC Data Modeling and Compression',
	author='Fares Meghdouri',
	author_email='fares.meghdouri@tuwien.ac.at',
	url='https://github.com/CN-TU/pydmc',
	packages=['pydmc'],
	requires=['numpy', 'scikit_learn', 'scipy', 'pandas', 'matplotlib']
)