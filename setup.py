"""
COSMOS - COordinated Safety On Manifold for multi-agent Systems

Setup script for pip installation.
"""

from setuptools import setup, find_packages

setup(
    name="cosmos-marl",
    version="0.1.0",
    description="Safe Multi-Agent RL with Constraint Manifold Projection",
    author="COSMOS Team",
    python_requires=">=3.8",
    packages=find_packages(include=["cosmos", "cosmos.*", "formation_nav", "formation_nav.*"]),
    install_requires=[
        "numpy>=1.20",
        "scipy>=1.7",
        "torch>=1.12",
        "gymnasium>=0.28",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "matplotlib>=3.5",
        "tqdm>=4.60",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.10",
        ],
        "logging": [
            "wandb>=0.13",
            "tensorboard>=2.10",
        ],
        "envs": [
            "safety-gymnasium>=0.4",
            "vmas>=1.2",
            "mujoco>=2.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "cosmos-train=cosmos.train:main",
        ],
    },
)
