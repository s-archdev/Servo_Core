from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="video-color-palette-analyzer",
    version="0.1.0",
    author="Video Palette Team",
    author_email="example@example.com",
    description="A tool for analyzing color palettes in videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-color-palette-analyzer",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "video-palette=main:main",
        ],
    },
)