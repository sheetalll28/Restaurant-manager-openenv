from setuptools import setup, find_packages

setup(
    name="restaurant-manager-openenv",
    version="1.1.0",
    description="AI restaurant shift management environment - OpenEnv compatible",
    author="OpenEnv Contributors",
    author_email="help.openenvhackathon@scaler.com",
    url="https://github.com/sheetalll28/Restaurant-manager-openenv",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "openenv-core>=0.2.0",
        "fastapi==0.111.0",
        "uvicorn[standard]==0.29.0",
        "pydantic==2.7.1",
        "openai==1.30.1",
        "httpx==0.27.0",
        "python-multipart==0.0.9",
    ],
    entry_points={
        "console_scripts": [
            "restaurant-manager-server=server.app:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
