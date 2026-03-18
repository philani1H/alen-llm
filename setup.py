"""
ALAN v4 — Package Setup
ECONX GROUP (PTY) LTD
"""

from setuptools import setup, find_packages

setup(
    name="alan-llm",
    version="4.0.0",
    description="ALAN v4 — Adaptive Learning and Awareness Network",
    author="ECONX GROUP (PTY) LTD",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0.0",
        "tqdm>=4.65.0",
    ],
    extras_require={},
    entry_points={
        "console_scripts": [
            "alan-train=alan.scripts.train:main",
            "alan-evaluate=alan.scripts.evaluate:main",
            "alan-serve=alan.api.server:main",
        ],
    },
)
