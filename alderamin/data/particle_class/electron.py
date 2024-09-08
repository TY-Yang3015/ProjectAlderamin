import attr


def spin_validator(instance, attribute, value):
    if value not in [1, -1]:
        return ValueError('spin must be either -1 or 1.')


@attr.s
class Electron:
    spin: int = attr.ib(validator=spin_validator)

    charge: int = -1
    mass: int = 1
