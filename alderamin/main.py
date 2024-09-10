import hydra

from alderamin.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from alderamin.config import PsiFormerConfig
from alderamin.trainer import PsiFormerTrainer
from absl import app
import logging

a = AtomicNucleus('H', (0, 0, 1))
c = ElectronNucleusSystem(system_nucleus=a,
                          num_electrons=1).initialize_system()
b = AtomicNucleus('H', (0, 0, 1.398))
d = ElectronNucleusSystem(system_nucleus=b,
                          num_electrons=1).initialize_system()

e = GlobalSystem(system_member=[c, d]).initialize_system()
p = PsiFormerConfig()
p.sampler_spec.system = e


@hydra.main(version_base=None)
def execute(config: PsiFormerConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    config = config.initialize()
    trainer = PsiFormerTrainer(config)
    trainer.train()


if __name__ == "__main__":
    execute(p)
