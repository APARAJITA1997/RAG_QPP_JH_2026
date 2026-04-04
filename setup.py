"""
setup.py — makes RAG-QPP installable as a Python package.

    pip install -e .

After this, you can do:
    from src.features import extract_features
    from src.models   import train_model
    etc.
"""
from setuptools import setup, find_packages

setup(
    name='rag-qpp',
    version='1.0.0',
    description=(
        'Adaptive Query Performance Prediction for '
        'Retrieval-Augmented Generation'
    ),
    author='Aparajita Sinha, Kunal Chakma',
    author_email='aparajitas824@gmail.com',
    url='https://github.com/YOUR_USERNAME/rag-qpp',
    packages=find_packages(exclude=['notebooks', 'scripts', 'data', 'outputs']),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'xgboost>=1.7.0',
        'lightgbm>=4.0.0',
        'torch>=2.0.0',
        'transformers>=4.38.0',
        'sentence-transformers>=2.6.0',
        'rouge-score>=0.1.2',
        'bert-score>=0.3.13',
        'datasets>=2.16.0',
        'tqdm>=4.66.0',
        'pyyaml>=6.0',
    ],
    extras_require={
        'gpu':     ['faiss-gpu>=1.7.4'],
        'cpu':     ['faiss-cpu>=1.7.4'],
        'colbert': ['colbert-ai>=0.2.19'],
        'dev':     ['pytest>=7.0', 'black', 'isort', 'jupyter'],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Indexing',
    ],
)
