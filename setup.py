from setuptools import setup, find_packages

setup(
    name="bannane",
    version="0.1.0",
    description="Bayesian Neural Network for Atomic and Nuclear Emulation",
    author="Jose M Munoz",
    url="https://github.com/munozariasjm/bannane",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "pyro-ppl",
        "scikit-learn",
        "matplotlib",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
