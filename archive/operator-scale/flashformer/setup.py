from setuptools import setup, find_packages

setup(
    name="flashformer",
    version="0.2.0",
    description="A high-performance low batch CUDA kernel library",
    author="Aniruddha Nrusimha",
    author_email="anin@mit.edu",
    packages=find_packages(),
    install_requires=[
        "torch==2.5.1",
        "numpy",
        "cheetah>=0.1.0",
    ],
    python_requires=">=3.10",
)
