from setuptools import setup

setup(name="nncell",
      version="0.1",
      url="http://github.com/swarchal/NN_cell",
      description="ImageXpress preprocessing for Keras",
      author="Scott Warchal",
      license="MIT",
      packages=["nncell"],
      tests_require=["pytest"],
      dependency_links=["https://github.com/swarchal/parserix/tarball/master#egg=parserix-0.1"],
      install_requires=["pandas>=0.16",
                        "numpy>=1.0",
                        "scikit-image>=0.12",
                        "parserix>=0.1",
                        "joblib>=0.10.0"],
      zip_safe=False)
