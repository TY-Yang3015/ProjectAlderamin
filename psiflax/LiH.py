import hydra
import jax

from psiflax.data import GlobalSystem, AtomicNucleus, ElectronNucleusSystem
from psiflax.trainer import PsiFormerTrainer

from omegaconf import DictConfig


li = AtomicNucleus("Li", (0, 0., 0.))
li = ElectronNucleusSystem(system_nucleus=li, num_electrons=3).initialize_system()
h = AtomicNucleus("H", (3.015, 0., 0.))
h = ElectronNucleusSystem(system_nucleus=h, num_electrons=1).initialize_system()
e = GlobalSystem(system_member=[li, h]).initialize_system()

@hydra.main(version_base=None, config_path="./config", config_name="base_config")
def execute(config: DictConfig) -> None:
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
    jax.config.update("jax_traceback_filtering", 'off')

    trainer = PsiFormerTrainer(config, e)
    trainer.train()


if __name__ == "__main__":
    execute()
