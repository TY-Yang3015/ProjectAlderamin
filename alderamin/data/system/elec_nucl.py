import attr

from alderamin.data.particle_class import AtomicNucleus, Electron
from alderamin.data.system import QuantumSystem


@attr.s
class ElectronNucleusSystem(QuantumSystem):
    system_nucleus: AtomicNucleus = attr.ib(validator=attr.validators.instance_of(AtomicNucleus))
    num_electrons: int = attr.ib(validator=attr.validators.instance_of(int))

    system_member: list = attr.ib(factory=list)

    def initialize_system(self):
        self.system_member.append(self.system_nucleus)
        current_spin = 1
        for i in range(self.num_electrons):
            self.system_member.append(Electron(current_spin))
            current_spin *= -1
        return self.validate_system()

    def validate_system(self):
        total_spin = self.total_spin
        electron_counter = self.num_electrons
        for member in self.system_member:
            if isinstance(member, AtomicNucleus):
                pass
            elif isinstance(member, Electron):
                pass
            else:
                raise ValueError("unknown system element type.")

        if electron_counter == 0:
            raise ValueError("no electron given.")
        elif electron_counter % 2 == 0:
            if total_spin != 0:
                raise ValueError("total spin must be zero for the ground "
                                 "state of even-number electron system.")
        elif electron_counter % 2 == 1:
            if abs(total_spin) != 1:
                raise ValueError("total spin must be one for the ground "
                                 "state of odd-number electron system.")

        return self

    @property
    def total_electrons(self) -> int:
        return len(self.electrons_list)

    @property
    def total_nucleus(self) -> int:
        return len(self.nucleus_list)

    @property
    def nucleus_list(self) -> list[AtomicNucleus]:
        nucleus_list = []
        for member in self.system_member:
            if isinstance(member, AtomicNucleus):
                nucleus_list.append(member)
        return nucleus_list

    @property
    def electrons_list(self) -> list[Electron]:
        electrons_list = []
        for member in self.system_member:
            if isinstance(member, Electron):
                electrons_list.append(member)
        return electrons_list

    @property
    def summary(self) -> dict:
        return {"total_spin": self.total_spin,
                "total_charge": self.total_charge,
                "total_electrons": self.total_electrons,
                "total_nucleus": self.total_nucleus,
                "nucleus_list": self.nucleus_list,
                "electrons_list": self.electrons_list}

# a = AtomicNucleus('Li', (0, 0, 0))
# c = ElectronNucleusSystem(system_nucleus=a,
#                          num_electrons=3).initialize_system()
# print(c.summary)

# b = AtomicNucleus('H', (0, 0, 1))

# d = ElectronNucleusSystem(system_nucleus=b,
#                          num_electrons=1).initialize_system()
