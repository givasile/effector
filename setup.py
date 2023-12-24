from setuptools import setup, find_packages

setup(
    name='effector',
    version='0.1.0',
    description='Effector: A Python package for feature effect estimation',
    author='Vasilis Gkolemis',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy'
        'matplotlib',
        'tqdm',
        'shap'
        ]
)

