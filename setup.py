from importlib import util as import_util
from pathlib import Path

from setuptools import find_packages
from setuptools import setup

spec = import_util.spec_from_file_location('_metadata', 'robolfd/_metadata.py')
_metadata = import_util.module_from_spec(spec)
spec.loader.exec_module(_metadata)

def parse_requirements_file(path):
    return [line.rstrip() for line in open(path, "r")]

# Requirements
reqs_main = parse_requirements_file("requirements/main.txt")
reqs_dev = parse_requirements_file("requirements/dev.txt")

env_requirements = [
    'gym>=0.18.0',
]

# Get the version from metadata.
version = _metadata.__version__

# If we're releasing a nightly/dev version append to the version string.
if '--nightly' in sys.argv:
  sys.argv.remove('--nightly')
  version += '.dev' + datetime.datetime.now().strftime('%Y%m%d')

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="robolfd",
    version=version,
    author="Mohan Kumar S",
    description="A Python library for learning from demonstration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mohanksriram/lfd",
    license='Apache License, Version 2.0',
    keywords='reinforcement-learning python machine learning',
    packages=find_packages(),
    install_requires=reqs_main,
    extras_require = {
        "dev": reqs_main + reqs_dev,
        "envs": env_requirements,
    },
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)