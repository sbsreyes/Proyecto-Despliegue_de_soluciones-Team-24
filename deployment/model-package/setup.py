#!/usr/bin/env python
from pathlib import Path
from setuptools import find_packages, setup

NAME            = "dengue-model"
DESCRIPTION     = "AI-lerta — Sistema de Alerta Temprana de Dengue Colombia."
EMAIL           = "team24@uniandes.edu.co"
AUTHOR          = "Team 24 — MAIA"
REQUIRES_PYTHON = ">=3.10.0"

ROOT_DIR         = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / "requirements"
PACKAGE_DIR      = ROOT_DIR / "model"

about = {}
with open(PACKAGE_DIR / "VERSION") as f:
    about["__version__"] = f.read().strip()


def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return [l for l in fd.read().splitlines() if l and not l.startswith("#")]


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(exclude=("tests",)),
    package_data={
        "model": [
            "VERSION",
            "config.yml",
            "datasets/*.csv",
            "trained/*.pkl",
        ]
    },
    install_requires=list_reqs(),
    include_package_data=True,
    license="BSD-3",
)
