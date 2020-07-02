import setuptools


setuptools.setup(
    name="tnt-learn",
    install_requires=[
        "scikit-learn",
        "dill",
        "numpy",
        "scipy",
        "nearset>=0.0.4",
        "blist"
    ],
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="Fast search in text data using cluster-pruning k-nearest neighbor search",
    url="https://github.com/kerighan/tnt",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
