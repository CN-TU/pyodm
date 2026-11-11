from setuptools import find_packages, setup

setup(
    name="pyodm",
    version="3.1.0",
    description="ODM Observers-based Data Modeling",
    author="Fares Meghdouri",
    author_email="fares.meghdouri@tuwien.ac.at",
    url="https://github.com/CN-TU/pyodm",
    packages=find_packages(exclude=("tests", "experiements", "venv", ".venv")),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "scipy>=1.8", "joblib>=1.1"],
    extras_require={"plot": ["matplotlib>=3.5"], "test": ["pytest>=7"]},
)
