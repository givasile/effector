from setuptools import setup, find_packages

setup(
    name="effector",
    version="0.0.261",
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "tqdm",
        "shap"
    ],
    # Other metadata such as author, description, etc.
)