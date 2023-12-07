from setuptools import setup, find_packages

setup(
    name='ada',
    version='1.0',
    packages=find_packages(include=['ada', 'ada.*']),
    install_requires=[
        'scikit-learn==1.0.2',
        'torch==1.13.0',
        'torchvision==0.14.0',
        'tqdm==4.64.1'       
    ]
)