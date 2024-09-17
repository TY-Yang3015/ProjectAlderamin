from psiflax.data.system import QuantumSystem, ElectronNucleusSystem
from psiflax.data.particle_class import Electron, AtomicNucleus

import attr


@attr.s(frozen=True)
class GlobalSystem(QuantumSystem):
    system_member: list[ElectronNucleusSystem] = attr.ib(factory=list)

    @property
    def electrons_list(self) -> list[Electron]:
        electrons_list = []
        for member in self.system_member:
            electrons_list.extend(member.electrons_list)
        return electrons_list

    @property
    def nucleus_list(self) -> list[AtomicNucleus]:
        nucleus_list = []
        for member in self.system_member:
            nucleus_list.extend(member.nucleus_list)
        return nucleus_list

    @property
    def total_electrons(self) -> int:
        return len(self.electrons_list)

    @property
    def total_nucleus(self) -> int:
        return len(self.nucleus_list)

    @property
    def summary(self) -> dict:
        return {
            "total_spin": self.total_spin,
            "total_charge": self.total_charge,
            "total_electrons": self.total_electrons,
            "total_nucleus": self.total_nucleus,
            "nucleus_list": self.nucleus_list,
            "electrons_list": self.electrons_list,
        }

    @property
    def electron_to_nucleus(self) -> list:
        electron_to_nucleus = []
        for i in range(self.total_nucleus):
            for j in range(self.system_member[i].num_electrons):
                electron_to_nucleus.append(i)
        return electron_to_nucleus

    def initialize_system(self):
        if abs(self.total_spin) not in [0, 1]:
            spin_list = []
            for member in self.system_member:
                if abs(member.total_spin) == 1:
                    spin_list.append(member.total_spin)

            num_plus_one = spin_list.count(1)
            num_minus_one = spin_list.count(-1)

            flip_count = abs(num_plus_one - num_minus_one) // 2

            i = 0
            if num_minus_one > num_plus_one:
                for member in self.system_member:
                    if member.total_spin == -1:
                        for electron in member.electrons_list:
                            electron.spin *= -1
                        i += 1
                        if i == flip_count:
                            break
            else:
                for member in self.system_member:
                    if member.total_spin == 1:
                        for electron in member.electrons_list:
                            electron.spin *= -1
                        i += 1
                        if i == flip_count:
                            break
            return self
        else:
            return self


# a = AtomicNucleus('Li', (0, 0, 0))
# c = ElectronNucleusSystem(system_nucleus=a,
#                          num_electrons=3).initialize_system()
# b = AtomicNucleus('H', (0, 0, 1))
# d = ElectronNucleusSystem(system_nucleus=b,
#                          num_electrons=1).initialize_system()
# f = AtomicNucleus('Ga', (0, 0, 1))
# G = ElectronNucleusSystem(system_nucleus=b,
#                          num_electrons=10).initialize_system()
# e = GlobalSystem(system_member=[c, d, G]).initialize_system()
# print(e.summary)
