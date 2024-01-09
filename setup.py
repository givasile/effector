from setuptools import setup, find_packages

setup(
    name='effector',
    version='0.0.1',
    description='Effector: A Python package for regional effects',
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

