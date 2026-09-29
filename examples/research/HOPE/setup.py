from setuptools import setup, find_packages

setup(
    name="hope-pruning",
    version="0.1.0",
    description="HOPE: Higher-Order Pruning of Experts for MoE Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "transformers==5.10.1",
        "accelerate",
        "numpy",
        "scipy",
        "h5py",
        "click",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "hope=hope.cli:cli",
        ],
    },
)
