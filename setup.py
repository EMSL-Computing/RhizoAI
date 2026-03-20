from setuptools import setup, find_packages

setup(
    name='oct-classification-project',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for OCT image classification using Random Forest methods.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'opencv-python',
        'matplotlib',
        'scikit-image',
        'joblib',
        'PyYAML'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)