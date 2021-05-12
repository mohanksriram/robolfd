"""Package metadata for robolfd.
This is kept in a separate module so that it can be imported from setup.py, at
a time when robolfd's dependencies may not have been installed yet.
"""

# We follow Semantic Versioning (https://semver.org/)
_MAJOR_VERSION = '0'
_MINOR_VERSION = '0'
_PATCH_VERSION = '0'

# Example: '0.4.2'
__version__ = '.'.join([_MAJOR_VERSION, _MINOR_VERSION, _PATCH_VERSION])