from setuptools import setup

setup(
    name="bgc",
    version="1.0",
    description="Tools to summarize output from BGC (bayesian genomic clines)",
    url="http://github.com/kerrycobb/bgc_summary",
    author="Kerry A Cobb",
    author_email="cobbkerry@gmail.com",
    license="MIT",
    requires=["h5py", "numpy", "pandas", "numpyro", "plotly", "fire"],
    scripts=["vcf2bgc.py"]
)