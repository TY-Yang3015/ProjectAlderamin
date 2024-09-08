import attr
import periodictable as prd


@attr.s(frozen=True)
class Electron:
    position: (float, float, float) = attr.ib()

    charge: int = -1
    mass: int = 1
