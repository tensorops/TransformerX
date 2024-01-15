VERSION = (0, 0, 1)

if len(VERSION) < 3:
    raise ValueError("VERSION must have at least three elements")

__version__ = ".".join(str(v) for v in VERSION[:3])
if len(VERSION) > 3:
    __version__ += "-" + ".".join(str(v) for v in VERSION[3:])
# version_str += "-dev"
print(__version__)
