from setuptools import setup, find_packages

setup(
    name="bgc-utils",
    version="1.0",
    description="Tools to summarize output from BGC (bayesian genomic clines)",
    url="http://github.com/kerrycobb/bgc_utils",
    author="Kerry A Cobb",
    author_email="cobbkerry@gmail.com",
    license="MIT",
    requires=["h5py", "numpy", "pandas", "numpyro", "plotly", "fire"],
    packages=find_packages(),
    scripts=["bgc_utils/vcf2bgc.py"],
)