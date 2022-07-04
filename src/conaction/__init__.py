from . import estimators
from . import numparam
from . import symparam

with open('VERSION', 'r') as f:
    __version__ = f.read()

__version_info__ = __version__.split('.')
