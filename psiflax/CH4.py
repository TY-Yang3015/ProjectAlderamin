import hydra
import jax

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

from omegaconf import DictConfig


c = AtomicNucleus("C", (0, 0, 0))
c = ElectronNucleusSystem(system_nucleus=c, num_electrons=6).initialize_system()
h1 = AtomicNucleus("H", (1.18886, 1.18886, 1.18886))
h1 = ElectronNucleusSystem(system_nucleus=h1, num_electrons=1).initialize_system()
h2 = AtomicNucleus("H", (-1.18886, -1.18886, 1.18886))
h2 = ElectronNucleusSystem(system_nucleus=h2, num_electrons=1).initialize_system()
h3 = AtomicNucleus("H", (1.18886, -1.18886, -1.18886))
h3 = ElectronNucleusSystem(system_nucleus=h3, num_electrons=1).initialize_system()
h4 = AtomicNucleus("H", (-1.18886, 1.18886, -1.18886))
h4 = ElectronNucleusSystem(system_nucleus=h4, num_electrons=1).initialize_system()
e = GlobalSystem(system_member=[c, h1, h2, h3, h4]).initialize_system()


@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config: DictConfig) -> None:
    jax.config.update("jax_traceback_filtering", "off")

    trainer = PsiFormerTrainer(config, e)
    trainer.train()


if __name__ == "__main__":
    execute()
