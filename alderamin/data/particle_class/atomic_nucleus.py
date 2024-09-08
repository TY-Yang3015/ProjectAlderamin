import attr
from attrs.validators import instance_of
import periodictable as prd


def is_valid_atom(instance, attribute, value):
    try:
        prd.elements.symbol(value)
    except ValueError:
        raise ValueError(f"'{value}' is not a valid atom.")


@attr.s(frozen=True)
class AtomicNucleus:
    name: str = attr.ib(validator=is_valid_atom)
    position: tuple[float, float, float] = attr.ib(validator=instance_of(tuple))

    @property
    def charge(self) -> int:
        return prd.elements.symbol(self.name).number

    spin: int = 0
