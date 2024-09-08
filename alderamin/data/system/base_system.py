import attr
import abc


class QuantumSystem(abc.ABC):
    system_member: list

    @property
    def total_spin(self) -> int:
        spin = 0
        for member in self.system_member:
            spin += member.spin
        return spin

    @property
    def spin(self) -> int:
        return self.total_spin

    @property
    def charge(self) -> int:
        return self.total_charge

    @property
    def total_charge(self) -> int:
        charge = 0
        for member in self.system_member:
            charge += member.charge
        return charge

    @property
    @abc.abstractmethod
    def summary(self) -> dict:
        pass
