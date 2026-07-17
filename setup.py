from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="acoustics_gpu",
    version="1.0.2",
    author="Evg5xa",
    author_email="zhmeldov@yandex.ru",
    description="GPU-ускоренное моделирование акустики помещений: Ray Tracing + Image Source",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Evg5xa/-Application-of-GPU-optimization-for-physical-models",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
    ],
)
