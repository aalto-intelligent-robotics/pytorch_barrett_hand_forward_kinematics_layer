from setuptools import find_packages, setup
import warnings

DEPENDENCY_PACKAGE_NAMES = ["trimesh", "torch", "numpy"]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(
            'Missing dependencies: {}. We recommend you follow '
            'the installation instructions at '
            'https://github.com/hassony2/manopth#installation'.format(
                missing_dependencies))


check_dependencies()

setup(
    name="barrett_kinematics",
    version="0.0.1",
    author="Jens Lundell",
    author_email="jens.lundell@aalto.fi",
    packages=find_packages(exclude=('tests', )),
    package_data={"barrett_kinematics": ['meshes/barrett_hand/*']},
    python_requires=">=3.5.0",
    description="PyTorch Barrett Hand layer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
