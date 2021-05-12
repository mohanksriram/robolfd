# Bash settings: fail on any error and display all commands being run.
set -e
set -x

# Set up a virtual environment.
conda create -n robolfd python=3.8
conda activate robolfd

# Python must be 3.7 or higher.
python --version

# Install dependencies.
pip install --upgrade pip setuptools
pip --version
pip install .
pip install .[dev]
pip install .[envs]