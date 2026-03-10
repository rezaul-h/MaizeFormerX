from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
README_PATH = ROOT / "README.md"
REQUIREMENTS_PATH = ROOT / "requirements.txt"


def read_readme() -> str:
    if README_PATH.exists():
        return README_PATH.read_text(encoding="utf-8")
    return "MaizeFormerX: lightweight multi-scale vision transformer for maize disease classification."


def read_requirements() -> list[str]:
    if not REQUIREMENTS_PATH.exists():
        return []

    requirements: list[str] = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


setup(
    name="maizeformerx",
    version="0.1.0",
    description="MaizeFormerX: lightweight multi-scale vision transformer for maize disease classification",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="MaizeFormerX Authors",
    python_requires=">=3.10",
    packages=find_packages(
        include=[
            "src",
            "src.*",
        ]
    ),
    package_dir={"": "."},
    include_package_data=True,
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4",
            "pytest-cov>=4.1",
        ],
    },
    entry_points={
        "console_scripts": [
            "maizeformerx=src.main:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License ::  MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "maize",
        "plant disease",
        "vision transformer",
        "deep learning",
        "computer vision",
        "agriculture ai",
    ],
    zip_safe=False,
)