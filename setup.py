from setuptools import setup, find_packages

setup(
    name="osint-cib-detector",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={"cib": ["static/*"]},
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "networkx>=3.1",
        "fastapi>=0.104",
        "uvicorn>=0.24",
        "pandas>=2.0",
        "python-dateutil>=2.8",
        "python-multipart>=0.0.6",
        "tqdm>=4.65",
    ],
    entry_points={
        "console_scripts": [
            "cib=cib.cli:main",
        ],
    },
)
